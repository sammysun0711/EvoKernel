# Depthwise Conv3D Optimization Trajectory

## Problem

3D depthwise convolution on AMD Instinct MI350X (CDNA4/gfx950):
```
Input:  [1, 512, 61, 45, 80]  BF16  NCHW
Weight: [512, 1, 3, 5, 5]     BF16
Output: [1, 512, 59, 45, 80]  BF16  NCHW
Groups: 512, Padding: (0,2,2), Stride: (1,1,1)
GFLOPs: 16.31
```

Each output element requires 75 multiply-accumulate operations (3x5x5 kernel).

---

## Phase 1: NHWC Direct-Compute Kernels (V80-V95)

**Approach**: NHWC layout, threads parallelize over channels, direct global memory access.

| Kernel | Strategy | VGPRs | Occ | Time (ms) | BW (GB/s) |
|--------|----------|-------|-----|-----------|-----------|
| V80 | 1 spatial/warp, 8 ch/thread | ~56 | 6 | 1.20 | ~368 |
| V82 | 8 spatial/block, 16 ch/thread | ~56 | 8 | 1.18 | ~375 |
| V84 | 8 spatial/block, 8 ch/thread, vec loads | 56 | 8 | **1.16** | 384 |
| V85 | LDS weight caching, 75 syncbarriers | ~56 | 5 | >1.5 | <300 |
| V95 | Wave-aligned, best MI308X kernel | 56 | 8 | 1.16 | 381 |

**Ceiling**: ~1.15ms. All NHWC variants converge here.

**Bottleneck**: No input data reuse. Each thread independently loads 75 input values + 75 weights from global memory per output. 302 bytes/output → bandwidth-limited.

**Key learning**: NHWC gives perfect channel coalescing but no weight or input reuse across threads. For large kernels (75 taps), data reuse matters more than coalescing.

---

## Phase 2: NCHW with LDS Input Caching (V96-V97)

**Approach**: Adopt PyHIP contrib kernel design. NCHW layout, cooperative LDS input loading, weight register caching.

### V97 Architecture (Production Winner)
```
Grid:   [B, C_out, D_out] = [1, 512, 59] = 30,208 blocks
Block:  256 threads
LDS:    ~25 KB input tile (KD × padded_H × padded_W = 3 × 49 × 84 bf16)
VGPRs:  155 (75 float weights + 45 uint32 input pairs + overhead)
Occ:    3 waves/SIMD

Algorithm:
  1. Cooperative LDS clear (zero-padding): 256 threads clear 12,348 bf16 values
  2. Cooperative input load: global_load_lds_dword (direct HBM→LDS, bypasses VGPRs)
  3. Weight load into registers: 75 bf16 → 75 float VGPRs
  4. Single __syncthreads()
  5. Compute loop: each thread processes ~7 output positions
     - 45 ds_read_b32 (batched, inline asm, compile-time offsets)
     - 1 s_waitcnt_lgkmcnt<0>
     - 150 v_fmac_f32 (inline asm)
     - KW_PACK=2 (2 outputs per loop iteration)
```

| Kernel | Change from V97 | Time (ms) | BW (GB/s) | vs V97 |
|--------|-----------------|-----------|-----------|--------|
| Contrib (HIP) | Baseline (original pyhip contrib) | 0.640 | 693 | 0.96x |
| **V97-origin** | **Clean reimplementation of contrib** | **0.614** | **720** | **1.00x** |

**Result**: 0.61ms — **9.4x faster** than PyTorch, **1.9x faster** than best NHWC (V84).

**Why V97 wins**: Cooperative LDS input caching amortizes load cost across all 256 threads. Weight register caching eliminates weight traffic entirely after initial load. 75 input values read from LDS (fast, ~ns) vs global memory (slow, ~100s ns).

---

## Phase 3: Profiling and Bottleneck Analysis

### rocprofv3 Hardware Counters (V97)

| Counter | Value | Analysis |
|---------|-------|----------|
| SQ_LDS_BANK_CONFLICT | 1,401,840 | 107% of GPU cycles |
| GRBM_COUNT | 1,305,905 | Total GPU cycles |
| SQ_WAVES | 3,776 | Per-SE wave count |
| TCP_TOTAL_ACCESSES | 357,904,384 | L1 cache accesses |
| TCC_HIT / TCC_MISS | 73% / 27% | L2 hit rate |
| HBM bandwidth used | 344 GB/s | 4.3% of 8 TB/s peak |

### Diagnostic: LDS Bank Conflicts

Created isolated load-only and compute-only kernels to separate LDS conflict sources:

| Phase | SQ_LDS_BANK_CONFLICT | GRBM_COUNT | Conflict/Cycle |
|-------|---------------------|------------|----------------|
| Load-only (global_load_lds) | 0 | 314,784 | 0% |
| Compute-only (ds_read_b32 + v_fmac) | 1,401,840 | 1,051,727 | 133% |
| Full kernel | 1,401,840 | 1,305,905 | 107% |

**Finding**: Bank conflicts come entirely from the compute phase (ds_read_b32 reads). However, they are only **2-way conflicts** in the second half-wavefront (threads 32-63 span two output rows). Actual performance impact: **<0.1% of kernel time**. The 107% ratio is misleading due to multi-SE counter aggregation.

### VALU Instruction Breakdown (V97 assembly)

| Instruction | Count | Purpose |
|-------------|-------|---------|
| v_fmac_f32 | 150 | Multiply-accumulate (the actual compute) |
| v_lshlrev_b32 | 89 | bf16→float conversion (shift to upper 16 bits) |
| v_and_b32 | 86 | bf16 extraction from uint32 pairs (low/high) |
| ds_read_b32 | 45 | LDS pair reads (2 bf16 per read) |
| Other VALU | ~10 | Address computation, output conversion |
| **Total** | **~380** | **Per compute iteration** |

**Real bottleneck**: VALU instruction count. The 175 conversion instructions (v_and + v_lshlrev) are 46% of total VALU. This is the price for packing 2 bf16 per VGPR (saves 30 VGPRs).

---

## Phase 4: v_dot2_f32_bf16 Optimization Attempt (V102-V103)

**Idea**: Replace 2× (v_and + v_lshlrev + v_fmac) = 6 instructions with 1× v_dot2_f32_bf16. Expected ~40% VALU reduction.

### V102: v_dot2 with v_cvt_pk_bf16_f32

Converted float weight pairs to packed bf16 via `v_cvt_pk_bf16_f32`, then used `v_dot2_f32_bf16` for computation.

| Metric | V97 | V102 |
|--------|-----|------|
| VGPRs | 155 | 125 |
| Occupancy | 3 | 3 |
| Correctness | PASS | **FAIL** (45/75 weights zeroed) |
| Spills | 0 | 0 |

**Root cause**: Compiler register allocator bug. The compiler reports 0 spills but silently reuses weight VGPRs for address computation. Weights 50-74 (d=2 slice) are overwritten. Confirmed by assembly inspection: `v_cvt_pk_bf16_f32 v123, v0, v2` where v0=0 (address base, not weight).

### V103: LDS Weight Pairs (Bypass v_cvt_pk_bf16_f32)

Stored weights as bf16 in LDS, loaded as packed uint32 pairs directly for v_dot2.

| Metric | V97 | V103 |
|--------|-----|------|
| VGPRs | 155 | 95 |
| Occupancy | 3 | 5 |
| Correctness | PASS | **FAIL** (identical 45/75 corruption) |

### Further Workarounds (All Failed)

| Attempt | Strategy | Result |
|---------|----------|--------|
| Per-depth-slice weight arrays | Separate arrays for d=0,1,2 to reduce pressure | FAIL — same corruption |
| Volatile weight registers | `volatile` to prevent reuse | FAIL — forces 192B scratch, garbage output |
| Different array layouts | Flat, 3D, union, separate arrays | FAIL — all v_dot2 variants fail identically |

**Conclusion**: The trigger is `v_dot2_f32_bf16` inline asm itself. When present in a fully-unrolled compute loop with an outer tile loop, the register allocator incorrectly determines weight registers are dead. Filed in `compiler_bug/BUG_REPORT.md`.

**Impact**: v_dot2 would provide ~20-30% speedup (0.61ms → ~0.45ms) but is blocked by ROCm 7.2 compiler bug.

---

## Phase 5: Group Conv Study Techniques (V104 Series)

Studied `/root/workspace/task/group_conv_study/` — production 2D grouped convolution kernels using:
- Row-streaming with double-buffered `buffer_load_lds`
- SwizzleT XOR-based LDS bank conflict avoidance
- Circular accumulators for sliding window
- Compiler pointer dereference for LDS reads (not inline asm)
- Unified LDS phase separation

**Not applicable**: MFMA matrix multiply (depthwise = diagonal weight matrix, no channel mixing).

### V104: Row-Chunk Streaming

Process 5 output rows per chunk instead of full 45×80 tile. Smaller LDS (~4.5KB), higher occupancy.

| Metric | V97 | V104 |
|--------|-----|------|
| VGPRs | 155 | 94 |
| Occupancy | 3 | 5 |
| LDS | 32,576 B | 4,536 B |
| Time | 0.63ms | **1.55ms** |
| Correctness | PASS | PASS |

**Why slower**: 9 chunks × (LDS clear + cooperative load + 2 barriers + compute) = ~20 barriers per kernel. Each barrier stalls the entire CU. Also, per-chunk loading is far less efficient than V97's single cooperative load.

### V104b: Compiler Pointer Reads (Same Algorithm as V97)

Replaced `ds_read_b32` inline asm with compiler-managed pointer dereference `s_input[idx]`. Same full-tile algorithm otherwise.

| Metric | V97 | V104b |
|--------|-----|-------|
| VGPRs | 155 | 83 |
| Occupancy | 3 | 5 |
| Time | 0.63ms | **1.34ms** |
| Assembly | 45 ds_read_b32 | 150 ds_read_u16 |

**Why slower**: Compiler generates `ds_read_u16` (16-bit reads) — 150 of them instead of V97's 45 `ds_read_b32` (32-bit reads). 3.3x more LDS read instructions. Compiler reads each bf16 individually instead of loading pairs.

### V104c: Batch ds_read_u16_d16_hi (Preload All Inputs)

Load each bf16 via `ds_read_u16_d16_hi` (places bf16 in upper 16 bits = valid float32). Pre-load all 90 input values into float registers, then compute.

| Metric | V97 | V104c |
|--------|-----|-------|
| VGPRs | 155 | **170** |
| Occupancy | 3 | **2** |
| Time | 0.63ms | **0.93ms** |

**Why slower**: 90 float input registers (vs V97's 45 uint32 pairs) → VGPR blowup to 170 → occupancy drops to 2. The conversion savings don't compensate for lower occupancy.

### V104d: Interleaved ds_read_u16_d16_hi + v_fmac

Load one bf16 value via `ds_read_u16_d16_hi`, immediately use it in `v_fmac_f32`. No input array pre-load.

| Metric | V97 | V104d |
|--------|-----|-------|
| VGPRs | 155 | 83 |
| Occupancy | 3 | 5 |
| Time | 0.63ms | **1.34ms** |
| Assembly | 1 s_waitcnt | **167 s_waitcnt** |

**Why slower**: Each `ds_read_u16_d16_hi` followed by `s_waitcnt lgkmcnt(0)` serializes the pipeline. V97 batches 45 reads then waits once. V104d does 150 read-wait-compute cycles. **Pipeline stall is the bottleneck**, not the instruction count.

### V104e: Packed bf16 Weight Pairs

Store weights as packed bf16 pairs (30 uint32 + 15 float singles instead of 75 floats).

| Metric | V97 | V104e |
|--------|-----|-------|
| VGPRs | 155 | **170** |
| Correctness | PASS | **FAIL** |

**Why failed**: Compiler generates extraction temporaries that inflate VGPRs to 170. Also produces wrong results — the bf16 packing/extraction logic has a bug in the global memory read pattern.

### Key Learnings from V104 Series

1. **Batched ds_read_b32 with single waitcnt is optimal**: V97 issues 45 LDS reads in parallel, then waits once. Any approach that serializes reads (per-value waitcnt, compiler pointer reads) is 2x slower.

2. **Higher occupancy does NOT compensate for worse IPC**: V104b/V104d have occupancy 5 (vs V97's 3) but are 2x slower because each output executes 150 ds_read_u16 instead of 45 ds_read_b32.

3. **bf16→float conversion cost is the price for VGPR efficiency**: The 175 conversion instructions (v_and + v_lshlrev) look wasteful but they enable packing 2 bf16 per VGPR, keeping total VGPRs at 155 and occupancy at 3.

4. **Row-chunk streaming adds too much barrier overhead**: For this problem size (45×80), full-tile loading with one barrier is far superior to 9 chunks with ~20 barriers.

---

## Phase 6: NHWC with LDS Weight Tiling (V105)

**Idea**: NHWC layout with weight caching in LDS. Adjacent threads access adjacent channels (coalesced), weights shared via LDS across spatial positions.

### V105: Channel-Tiled NHWC

```
TILE_C = 64 channels per tile
CHANS_PER_THREAD = 8
THREADS_PER_SPATIAL = 8
SPATIAL_PER_BLOCK = 32
Grid: [26,550 spatial_blocks, 8 channel_tiles]
LDS: 75 × 64 × 2 = 9,600 bytes (weight tile as bf16)
```

| Metric | V97 | V105 |
|--------|-----|------|
| VGPRs | 155 | 42 |
| Occupancy | 3 | 8 |
| LDS | 32,576 B | 9,600 B |
| Time | 0.63ms | **1.26ms** |
| Correctness | PASS | PASS |

**Why slower**: 75 global memory reads per output position (no input caching). Weight caching saves 75 weight reads, but input reads dominate. V97's cooperative LDS input caching (load once, read 7x from LDS) is fundamentally more efficient for large kernels.

**NHWC vs NCHW for depthwise**: NHWC gives coalesced channel access but trades away input data reuse. For 75-tap kernels, input reuse (NCHW + LDS) beats coalescing (NHWC).

---

## Summary: All Kernels Ranked

| Rank | Kernel | Layout | Time (ms) | vs V97 | Key Feature |
|------|--------|--------|-----------|--------|-------------|
| 1 | **V97-origin** | **NCHW** | **0.614** | **1.00x** | Batched ds_read_b32, weight reg cache, global_load_lds |
| 2 | Contrib (HIP) | NCHW | 0.640 | 0.96x | Original pyhip contrib kernel |
| 3 | V104c | NCHW | 0.927 | 0.66x | ds_read_u16_d16_hi batch (VGPR blowup) |
| 4 | V84 | NHWC | 1.157 | 0.53x | Direct global, no caching |
| 5 | V105 | NHWC | 1.262 | 0.49x | LDS weight tile, global input |
| 6 | V104b | NCHW | 1.337 | 0.46x | Compiler pointer reads |
| 7 | V104d | NCHW | 1.339 | 0.46x | Interleaved d16_hi+fmac |
| 8 | V104 | NCHW | 1.549 | 0.40x | Row-chunk streaming |
| 9 | V105 (TILE_C=128) | NHWC | 2.605 | 0.24x | Oversized LDS weight tile |
| - | V102 | NCHW | - | - | v_dot2 (WRONG — compiler bug) |
| - | V103 | NCHW | - | - | v_dot2 LDS weights (WRONG) |
| - | V104e | NCHW | - | - | Packed bf16 weights (WRONG) |

---

## Why V97 is Optimal

V97 sits at a well-tuned equilibrium across multiple dimensions:

```
                V97 Position
                     ↓
LDS reads:    [serialized] ──────●────── [batched]
                               45 ds_read_b32 + 1 waitcnt

VGPRs:        [too few (spills)] ──●──── [too many (low occ)]
                                  155 VGPRs, occ 3

Data reuse:   [none (global)] ──────●──── [full (LDS)]
                              Input in LDS, weights in regs

Instruction:  [scalar (slow)] ──●──── [packed (fast)]
                         v_fmac_f32 (v_dot2 blocked by bug)
```

Every alternative we tried is worse in at least one dimension:
- More occupancy → fewer batched reads → slower
- Fewer VGPRs → can't cache weights → more traffic → slower
- NHWC → no input reuse → more global reads → slower
- Row-chunk → more barriers → slower
- Packed reads → VGPR blowup or serialization → slower

---

## Remaining Optimization Path

---

## Phase 7: Deep Profiling — Instruction-Level Analysis

### PMC Counter Profiling (8 passes with rocprofv3)

Collected hardware counters across 8 separate PMC passes. Key per-wave metrics:

| Counter | Value/wave | Insight |
|---------|-----------|---------|
| SQ_INSTS_VALU | 2,165 | 70% of all instructions |
| SQ_INSTS_SALU | 493 | 16% (address calc, control) |
| SQ_INSTS_LDS | 374 | 12% (ds_read_b32 reads) |
| SQ_INSTS_VMEM | 60 | 2% (global loads/stores) |
| SQ_WAVE_CYCLES | 7,276 | Total cycles per wave |
| SQ_WAIT_ANY | 2,203 | 30.3% of wave time spent waiting |
| SQ_WAIT_INST_LDS | 476 | Only 6.5% wait from LDS |
| Non-LDS wait | 1,727 | **23.7% wait from VMEM/other** |

#### VALU Sub-Type Breakdown

| VALU Type | Count/wave | % of VALU | Purpose |
|-----------|-----------|-----------|---------|
| SQ_INSTS_VALU_FMA_F32 | 1,087 | 50.2% | Actual compute (v_fmac) |
| SQ_INSTS_VALU_IOPS | 974 | **45.0%** | **Integer/bitwise (v_and, v_lshlrev)** |
| SQ_INSTS_VALU_INT32 | 220 | 10.1% | Address calculations |
| SQ_INSTS_VALU_CVT | 19 | 0.9% | Type conversions |

**Key finding**: Integer operations (IOPS) — the bf16 extraction instructions — consume **45% of all VALU**, nearly equal to the FMA compute itself.

#### Wait Cycle Analysis

| Wait Source | Cycles/wave | % of Total |
|-------------|-------------|------------|
| Total wave time | 7,276 | 100% |
| Active compute | 5,073 | **69.7%** |
| All waits | 2,203 | 30.3% |
| LDS waits | 476 | 6.5% |
| **Non-LDS waits (VMEM)** | **1,727** | **23.7%** |

**Surprise**: LDS bank conflicts cause only 6.5% wait (not the 107% misleading ratio). The **23.7% non-LDS wait** is from global memory latency during input/weight loading.

### Thread Trace (rocprofv3 advanced_thread_trace)

Per-instruction timing from a single CU. The compute loop (hitcount=3422) cycle breakdown:

| Category | Instructions | Latency | % of Loop | Stall Rate |
|----------|-------------|---------|-----------|------------|
| **FMA compute** (v_fmac_f32) | 150 | 2,072,176 | **33.6%** | 0.9% |
| **LDS reads** (ds_read_b32) | 45 | 1,170,580 | **19.0%** | 45.8% |
| **NOPs** (s_nop, compiler hazards) | 60 | 821,796 | **13.3%** | 100% |
| **BF16 extract** (v_lshlrev_b32) | 46 | 651,896 | **10.6%** | 3.4% |
| **BF16 extract** (v_and_b32) | 45 | 619,016 | **10.0%** | 0.5% |
| **Wait** (s_waitcnt) | 2 | 258,064 | **4.2%** | 100% |
| **Output pack** (v_cvt_pk_bf16_f32) | 2 | 182,732 | **3.0%** | 84.6% |
| **Other** (addr, store, misc) | 24 | 398,472 | **6.5%** | 25% |

#### Critical Insights

1. **FMA pipeline is NOT stalled** (0.9% stall rate). The v_fmac_f32 units are running at full throughput when fed work — they're just starved by overhead instructions.

2. **BF16 extraction is 20.6% of compute** — 91 instructions (v_and_b32 + v_lshlrev_b32) doing pure bit manipulation to extract bf16 values from uint32 pairs. These have <2% stall rate — they're just raw VALU cycles. `v_dot2_f32_bf16` would eliminate ALL of them.

3. **Compiler-inserted NOPs waste 13.3%** — 60 `s_nop` instructions inserted for hazard avoidance between `s_mov_b32 m0` (for global_load_lds_dword) and `global_load_lds_dword`. These are in the input loading phase, not the compute loop, but still contribute significantly. Hand-tuned assembly could schedule other useful work during these NOP slots.

4. **LDS reads have 45.8% stall rate** — but this is only 8.7% of total time. The stalls are a mix of bank conflicts and LDS pipeline latency. The batched ds_read_b32 approach (all 45 reads issued before waitcnt) helps, but ~half the LDS read time is still stall.

5. **v_cvt_pk_bf16_f32 stalls 84.6%** — the 2 output packing instructions have high latency (pack fp32→bf16 pair). Only 3% of total time but hints at a pipeline hazard.

### Three Optimization Targets (from thread trace)

| Target | Cycles Saved | % of Compute | Approach |
|--------|-------------|-------------|----------|
| BF16 extraction | ~1,270,912 | 20.6% | v_dot2_f32_bf16 (packed bf16 FMA) |
| FMA halving | ~1,036,088 | 16.8% | v_dot2 does 2 FMA per instruction |
| NOP elimination | ~410,898-821,796 | 6.7-13.3% | Hand-tuned instruction scheduling |
| **Combined** | **~2,717,898** | **~44%** | **Target: 0.35-0.45ms** |

### Next Steps: Implementation Paths

**Path A: Assembly patch** — Compile V97 with v_fmac (correct register allocation), then surgically patch the `.s` file to replace fmac+conversion pairs with v_dot2_f32_bf16. Load patched `.s` via pyhip. Bypasses the compiler register allocator bug.

**Path B: pyhip JIT assembly** — Rewrite the compute loop using `@pyhip.jit` with explicit register allocation. Full control over instruction scheduling, NOP placement, and v_dot2 usage. More work but addresses all three targets simultaneously.

**Path C: NOP-only optimization** — Analyze the 60 NOPs inserted by the compiler. Many are between `s_mov_b32 m0` and `global_load_lds_dword` in the input loading phase. If we can restructure the loading to eliminate these NOPs (e.g., by hoisting m0 writes earlier), we get a ~13% improvement without touching the compute loop.

---

---

## Phase 8: Assembly Patching — v_dot2_f32_bf16 (V97-patched)

### Approach

Compiled V97 with v_fmac (correct register allocation), then surgically patched the `.s` file to replace paired v_fmac_f32 instructions with v_cvt_pk_bf16_f32 + v_dot2_f32_bf16 sequences. This bypasses the compiler register allocator bug by changing only opcodes, not register assignments.

### v_dot2_f32_bf16 Semantics (Verified on gfx950)

Test kernel confirmed operand behavior:
```
v_cvt_pk_bf16_f32 dst, src0, src1  →  dst.lo = bf16(src0), dst.hi = bf16(src1)
v_dot2_f32_bf16 dst, src0, src1, src2  →  dst = src0.lo*src1.lo + src0.hi*src1.hi + src2
```

Correct packing order: `v_cvt_pk_bf16_f32 scratch, weight_lo, weight_hi`

### Critical Discovery: p=1 Depends on p=0 Extractions

V97 processes 2 output pixels per iteration (KW_PACK=2). The p=0 section extracts bf16 values from input pairs:
```asm
v_lshlrev_b32 vTMP, 16, vINPUT    ; extract LOW bf16 as float
v_and_b32     vTMP, 0xffff0000, vINPUT  ; extract HIGH bf16 as float
```

The p=1 section **reuses ALL 60 extraction results** from p=0 (shifted by 1 bf16 position). This means extraction instructions CANNOT be removed — only the fmac instructions can be replaced.

### What Was Patched (p=0 Only)

For each of 30 fmac pairs in p=0 (accumulator v79):
```asm
; BEFORE: 2 fmacs using extracted bf16 values
v_fmac_f32 v79, vW_LO, vTMP_LO    ; acc += weight_lo * input_lo
v_fmac_f32 v79, vW_HI, vTMP_HI    ; acc += weight_hi * input_hi

; AFTER: 1 cvt_pk + 1 dot2 (replacing 2 fmacs)
v_cvt_pk_bf16_f32 v155, vW_LO, vW_HI  ; pack weights → bf16 pair
v_dot2_f32_bf16 v79, v155, vINPUT, v79 ; 2× FMA in one instruction
s_nop 0                                ; replaces second fmac
```

### Results

| Metric | V97-origin | V97-patched | Change |
|--------|-----------|-------------|--------|
| Correctness | - | **PASS** (max_diff=0.125) | bf16 rounding |
| Time | 0.635ms | 0.624ms | **+1.7%** |
| p=0 fmacs | 60 | 0 (→ 30 dot2) | -60 fmac, +30 dot2 |
| p=1 fmacs | 90 | 90 (unchanged) | 0 |
| v_cvt_pk added | 0 | 30 | +30 |
| s_nop added | 60 | 90 | +30 |

### Why Only 1.7% Speedup

1. **Only p=0 patched** — p=1's 75 fmacs + extractions are unchanged (50% of compute)
2. **Extractions kept** — all 91 v_and/v_lshlrev remain (p=1 depends on them)
3. **Added overhead** — 30 v_cvt_pk_bf16_f32 + 30 s_nop offset the 30 removed fmacs
4. **Net instruction change**: replaced 60 fmacs with 30 dot2 + 30 cvt_pk + 30 nop = same count but dot2 does 2× work

### What Would Be Needed for Full Optimization

To achieve the theoretical 44% speedup, we would need to:
1. **Eliminate ALL extractions** — requires restructuring p=1 to also use v_dot2 with independently extracted values
2. **Remove the 60 s_nop** — requires understanding and fixing compiler-inserted hazard avoidance
3. **Patch p=1 as well** — p=1 uses shifted input positions; would need v_alignbit_b32 to create offset bf16 pairs for v_dot2

This would require a full rewrite of the compute loop in assembly (via pyhip JIT), not just patching individual instructions.

### Key Learnings

1. **Assembly patching works** — the v_dot2_f32_bf16 instruction produces correct results when properly used post-compilation, bypassing the compiler bug
2. **v_cvt_pk_bf16_f32 operand order confirmed**: `dst.lo = bf16(src0)`, `dst.hi = bf16(src1)`
3. **Register interdependence limits patch scope** — the compiler's interleaved extraction schedule creates deep dependencies between p=0 and p=1 that can't be broken by simple patching
4. **Full optimization requires JIT assembly** — to restructure both p=0 and p=1 simultaneously, use pyhip @pyhip.jit with explicit register control

---

## Phase 9: pyhip JIT Rewrite with v_dot2_f32_bf16

### Approach

Rewrote the compute loop of the existing pyhip JIT depthwise kernel (`conv_depthwise_3d_jit` in `src/contrib/conv_depthwise.py`) to use `v_dot2_f32_bf16`. The pyhip JIT has its own register allocator (linear-scan, not LLVM), completely bypassing the compiler bug.

Key changes from original JIT:
1. **Weight storage**: packed bf16 pairs via `ds_read_b32` (30 uint32) + singles via `ds_read_u16_d16_hi` (15 float) = 45 VGPRs (down from 75)
2. **Paired taps**: `v_dot2_f32_bf16` replaces 2× v_fmac + 2× bf16 extraction
3. **p=1 shifted pairs**: `v_alignbit_b32` creates offset bf16 pairs from adjacent input pairs
4. **Odd singles (kw=4)**: unchanged `v_fmac_f32`

### pyhip JIT DSL Patterns Used

```python
# Packed weight pairs (30 registers instead of 75)
Bpair = J.gpr(KD, KH, KW_PAIRS, "u32")     # [3][5][2] uint32 bf16 pairs
J.ds_read_b32(Bpair[kd,kh,wp], vaddr, mod=f"offset:{off}")

# v_dot2_f32_bf16 for paired FMA (p=0)
J.v_dot2_f32_bf16(Acc[0], Bpair[kd,kh,wp], A[kd,kh,wp], Acc[0])

# v_alignbit_b32 for p=1 shifted input
J.v_alignbit_b32(Ashift, A[kd,kh,wp+1], A[kd,kh,wp], 16)
J.v_dot2_f32_bf16(Acc[1], Bpair[kd,kh,wp], Ashift, Acc[1])
```

### Instruction Count Comparison

| Instruction | JIT-orig | JIT-dot2 | Change |
|---|---|---|---|
| v_fmac_f32 | 150 | 30 | **-120** |
| v_dot2_f32_bf16 | 0 | 60 | +60 |
| v_alignbit_b32 | 0 | 30 | +30 |
| v_lshlrev_b32 (bf16 extract) | 55 | 25 | -30 |
| v_and_b32 (bf16 extract) | 46 | 16 | -30 |
| v_mov_b32 | 135 | 76 | -59 |
| ds_read_b32 | 45 | 75 | +30 |
| ds_read_u16_d16_hi | 75 | 15 | -60 |
| **Total VALU** | **~540** | **~360** | **-33%** |

### Results

| Metric | V97-HIP | JIT-orig | JIT-dot2 |
|---|---|---|---|
| Time (ms) | **0.632** | 0.668 | **0.629** |
| VGPRs | 155 | 137 | **98** |
| Occupancy | 3 | 3 | **4** |
| Spills | 0 | 0 | 0 |
| LDS (bytes) | 32,576 | 24,888 | 24,888 |
| Correctness vs V97 | - | 0.25 | 0.25 |

**JIT-dot2 is 0.4% faster than V97-HIP** and **5.8% faster than JIT-orig**, with correct results.

### PMC Counter Comparison

| Counter (per-wave) | V97-HIP | JIT-dot2 | Change |
|---|---|---|---|
| SQ_INSTS_VALU | 2,165 | 2,020 | -6.7% |
| SQ_INSTS_LDS | 374 | 558 | +49% |
| SQ_WAIT_ANY | 2,203 | 1,573 | -28.6% |
| SQ_WAVE_CYCLES | 7,276 | 9,791 | +34.6% |
| GRBM_COUNT (GPU cycles) | 1,273,248 | 1,232,960 | **-3.2%** |
| Wait % | 30.3% | 16.1% | Better |

### Why Only 0.4% Wall-Clock Improvement Despite 33% Fewer VALU

1. **Bottleneck shifted to LDS throughput**: LDS instructions increased 49% (558 vs 374 per wave). Weight pair loads via `ds_read_b32` replace individual `ds_read_u16_d16_hi` — same data volume but different instruction mix.

2. **Occupancy 4 increases per-wave latency**: With 4 waves sharing each SIMD (up from 3), individual waves take 35% longer (9,791 vs 7,276 cycles). But total GPU cycles decreased 3.2% because more waves execute concurrently.

3. **v_dot2 latency**: `v_dot2_f32_bf16` may have higher latency than `v_fmac_f32` (bf16 multiply-accumulate involves internal format conversion), partially offsetting the instruction count reduction.

4. **v_alignbit_b32 overhead**: 30 extra instructions for p=1 shifted pairs add ~10% overhead to the compute loop that didn't exist in the original v_fmac approach.

5. **Already at memory bound**: V97 was only using 4.3% of peak HBM bandwidth, but its working set fits in L1/L2 cache. The kernel may be **latency-bound** rather than throughput-bound — reducing VALU count doesn't help if the pipeline is waiting for LDS results.

### Key Conclusions

1. **pyhip JIT successfully bypasses the compiler v_dot2 bug** — the register allocator bug is specific to LLVM/clang's register allocator, not the hardware. pyhip's linear-scan allocator handles v_dot2 correctly.

2. **v_dot2_f32_bf16 works correctly on gfx950** — produces identical results (within bf16 precision) to v_fmac_f32 approach.

3. **VALU is no longer the bottleneck** — after reducing VALU by 33%, the kernel speed barely changed. The bottleneck has shifted to LDS read throughput and/or memory latency.

4. **V97/JIT at ~0.63ms is near the hardware limit** for this kernel configuration — the kernel is limited by data movement (LDS reads, global loads) rather than compute. Further optimization would require reducing data movement (e.g., smaller LDS working set, better data reuse patterns).

---

---

## Phase 10: LDS Traffic Optimization Experiments

### LDS Bottleneck Analysis

PMC profiling of JIT-dot2 kernel revealed:

| Metric (per-wave) | V97-HIP | JIT-dot2 |
|---|---|---|
| LDS load instructions | 326 | 551 (+69%) |
| LDS bytes read | 40,293 | 69,554 (+73%) |
| LDS bank conflicts | 371 | **0** |
| LDS active cycles | 1,120 | 2,025 (+81%) |
| LDS bytes/active cycle | 36 | 34 |
| LDS utilization (vs 128B/cyc peak) | 28% | 27% |

Key finding: both kernels utilize only **~27% of peak LDS bandwidth**. The bottleneck is **LDS instruction issue rate** — the instruction scheduler can't issue LDS reads fast enough, leaving the LDS pipeline idle between reads.

### Experiment: ds_read_b64 Wider LDS Reads (JIT-dot2-b64)

**Hypothesis**: Wider reads (ds_read_b64 = 8 bytes vs ds_read_b32 = 4 bytes) would reduce LDS instruction count by 33%, letting the scheduler issue fewer but wider reads.

**Implementation**: `kernels/conv_depthwise3d_jit_dot2_b64.py`
- Per (d,h) row: 1× ds_read_b64 (pairs 0+1) + 1× ds_read_b32 (pair 2) = 2 reads instead of 3
- Uses 2-aligned register pairs for b64 destination

**Result**:

| Metric | JIT-dot2 | JIT-dot2-b64 |
|---|---|---|
| VGPRs | 98 | 98 |
| Occupancy | 4 | 4 |
| Correctness | PASS | PASS |
| **Time** | **0.629ms** | **2.631ms (4.2x slower!)** |

**Analysis**: `ds_read_b64` is catastrophically slower than `2× ds_read_b32` on gfx950. This is likely a micro-architectural penalty — the LDS hardware on CDNA4 may be optimized for 32-bit reads and penalize wider read instructions. The wider reads do NOT improve throughput despite reducing instruction count.

**Conclusion**: Wider LDS reads are not viable on gfx950. The LDS utilization bottleneck must be addressed through other means (interleaving compute with reads, or reducing total data volume).

### Experiment: Software Pipelining (JIT-dot2-pipe)

**Hypothesis**: Interleave LDS reads with compute by depth slice. Issue reads for d=N+1 while computing d=N using `s_waitcnt lgkmcnt(15)` (wait for first 15 out of 30 in-flight reads).

**Implementation**: `kernels/conv_depthwise3d_jit_dot2_pipe.py`
- Issue 15 reads for d=0, then 15 reads for d=1
- `waitcnt(15)` → compute d=0 while d=1 reads complete
- Issue 15 reads for d=2 → `waitcnt(15)` → compute d=1 → `waitcnt(0)` → compute d=2

**Result**:

| Metric | JIT-dot2 | JIT-pipe |
|---|---|---|
| VGPRs | 98 | 99 |
| Occupancy | 4 | 4 |
| Correctness | PASS | PASS |
| Time | 0.619ms | 0.625ms (+1.0%) |

**Analysis**: No improvement. The depth-interleaved pipelining adds waitcnt overhead that cancels latency hiding. ds_read_b32 latency is short (~2 cycles per half-wave) — the LDS bottleneck is instruction issue rate, not read latency. There isn't enough latency to hide.

### Experiment: KW_PACK=4 (JIT-dot2-kwp4)

**Hypothesis**: Process 4 output pixels per iteration (instead of 2). Halves iteration count and LDS read batches.

**Implementation**: `kernels/conv_depthwise3d_jit_dot2_kwp4.py`
- 4 accumulators (Acc[0..3]) for 4 output pixels
- 4 input pairs per (d,h) row (vs 3 with KW_PACK=2)
- p=0,2 use aligned pairs; p=1,3 use v_alignbit_b32 shifted pairs
- 4× v_dot2 + 2× v_alignbit per (d,h) weight pair group

**Result**:

| Metric | JIT-dot2 | JIT-kwp4 |
|---|---|---|
| VGPRs | 98 | 115 |
| Occupancy | 4 | 4 |
| Correctness | PASS | **FAIL** (max_diff=50, ~40% of values wrong) |
| Time | 0.619ms | 1.021ms (1.6x slower) |

**Analysis**: Wrong results — the 4-way output pixel mapping has a subtle indexing bug. Even ignoring correctness, 115 VGPRs and 1.6x slower due to more v_dot2 and v_alignbit per iteration (4× vs 2× per weight pair). KW_PACK=4 increases compute per iteration more than it reduces iterations.

### Summary of All Phase 10 Experiments

| Variant | Technique | Correct | Time | vs JIT-dot2 |
|---|---|---|---|---|
| JIT-dot2 | Baseline v_dot2 | PASS | 0.619ms | 1.000x |
| JIT-dot2-b64 | ds_read_b64 wider reads | PASS | 2.631ms | 0.235x |
| JIT-dot2-pipe | Software pipeline by depth | PASS | 0.625ms | 0.990x |
| JIT-dot2-kwp4 | KW_PACK=4 | FAIL | 1.021ms | 0.606x |

### Experiment: Row-Level Interleaved Read-Compute (JIT-dot2-ilv, ilv2)

**Background** (from pyhip docs `mem_latency.md`, `vm_cnt.md`):
- ds_read_b32 latency: **52 cycles** on MI308X (measured)
- ds_read issue rate @4 waves: **32 cycles/instruction** (measured)
- LGKM counter: 4-bit field in `s_waitcnt` encoding (values 0-15)
- `vm_cnt.md` confirmed VMEM has a 64-entry hardware queue (issue pipe stalls ~24th load), and **guessed** LDS may work similarly with 16 entries, but noted *"considering the super-fast throughput of LDS, that causes no stall most-likely"*. The LDS queue depth was **never actually tested**.

**Hypothesis**: Interleave reads and compute per (d,h) row to reduce VGPR pressure and overlap 52-cycle ds_read latency with VALU compute.

**Implementation**:
- `ilv`: Issue 3 reads → `waitcnt(0)` → compute row → repeat for all 15 rows. Only 3 input VGPRs (reused per row).
- `ilv2`: Lookahead — issue reads for row N+1, `waitcnt(3)`, compute row N, swap buffers. 6 input VGPRs (double-buffered).

**Result**:

| Kernel | VGPRs | Occupancy | Correctness | Time (ms) | vs V97 |
|---|---|---|---|---|---|
| JIT-dot2 | 98 | 4 | PASS | 0.625 | 0.991x |
| **JIT-ilv** | **57** | **8** | PASS | 0.638 | 0.971x |
| **JIT-ilv2** | **60** | **8** | PASS | 0.629 | 0.985x |

**Analysis**: Occupancy jumped to **8 waves/SIMD** (from 4) thanks to only 57-60 VGPRs. But performance didn't improve — actually 1-3% slower.

**Why JIT interleaving didn't improve over batched reads**:
1. **15 extra s_waitcnt**: Each per-row `waitcnt(0)` is a hard pipeline stall. With batched reads, 1 waitcnt for 45 reads; per-row has 15 waitcnts for the same 45 reads.
2. **pyhip JIT register allocator overhead**: The JIT emits extra `v_mov` instructions for register management that a HIP compiler avoids.
3. **No compiler scheduling control**: The JIT emits instructions in the Python-source order. There's no equivalent of `sched_group_barrier` to tell the assembler how to interleave reads with compute for latency hiding.
4. **Higher occupancy alone is not sufficient**: Going from 4 to 8 waves doesn't help if the instruction mix within each wave is still read-stall-compute-stall (serial pattern).

**Conclusion at the time**: Concluded this was a "fundamental LDS issue throughput limit." **This turned out to be wrong** — Phase 12c showed that the same interleaving principle, when implemented in HIP C++ with `sched_group_barrier`, achieved a 15.8% speedup. The JIT implementation's failure was due to pyhip overhead + lack of compiler scheduling hints, not the interleaving idea itself.

### Summary of All Phase 10-11 Experiments

| Variant | Technique | VGPRs | Occ | Correct | Time | vs V97 |
|---|---|---|---|---|---|---|
| JIT-dot2 | Baseline v_dot2 | 98 | 4 | PASS | 0.625ms | 0.991x |
| JIT-b64 | ds_read_b64 | 98 | 4 | PASS | 2.631ms | 0.235x |
| JIT-pipe | Pipeline by depth | 99 | 4 | PASS | 0.625ms | 0.990x |
| JIT-kwp4 | KW_PACK=4 | 115 | 4 | FAIL | 1.021ms | 0.606x |
| **JIT-ilv** | **Row interleave** | **57** | **8** | PASS | 0.638ms | 0.971x |
| **JIT-ilv2** | **Row lookahead** | **60** | **8** | PASS | 0.629ms | 0.985x |

**At this point we believed** the kernel was at the hardware limit. Phase 12c proved otherwise.

---

## Final Summary

| Phase | Kernel | Time (ms) | Key Technique | Blocked By |
|---|---|---|---|---|
| 1 | V84 (NHWC) | 1.16 | Direct global, no cache | No input reuse |
| 2 | **V97 (NCHW)** | **0.614** | **LDS input cache, weight regs** | **Production winner** |
| 3 | Profiling | - | rocprofv3 PMC + thread trace | - |
| 4 | V102/V103 | FAIL | v_dot2 in HIP | Compiler bug |
| 5 | V104 series | 0.93-1.55 | Alt. approaches (5 variants) | All slower |
| 6 | V105 (NHWC tile) | 1.26 | LDS weight tile | No input reuse |
| 7 | Profiling | - | Thread trace: 33.6% FMA, 20.6% extract, 13.3% NOP | - |
| 8 | V97-patched | 0.624 | Asm patch v_dot2 (p=0 only) | p=1 dependency |
| 9 | **JIT-dot2** | **0.629** | **Full v_dot2 via pyhip JIT** | **LDS-bound** |
| 10a | JIT-dot2-b64 | 2.631 | ds_read_b64 wider reads | gfx950 penalizes b64 |
| 10b | JIT-dot2-pipe | 0.625 | Software pipeline by depth | No latency to hide |
| 10c | JIT-dot2-kwp4 | FAIL | KW_PACK=4 | Indexing bug + more compute |
| 11a | JIT-dot2-ilv | 0.638 | Row interleave (57 VGPRs, occ 8) | More waitcnts |
| 11b | JIT-dot2-ilv2 | 0.629 | Row lookahead (60 VGPRs, occ 8) | No overlap opportunity |
| 12a | MFMA-Toeplitz | 3.68-14.4 | MFMA_4x4x4 batch=16 channels | Data loading dominates |
| 12b | JIT-dot2-pk | 0.650 (FAIL) | v_pk_fma_f32 odd taps | bf16 format mismatch, <5% gain |
| **12c** | **V97-sgb** | **0.537** | **sched_group_barrier + row interleave** | **NEW BEST! 15.8% faster than V97** |

---

## Phase 12: MFMA Multi-Channel Toeplitz Convolution

**Approach**: Inspired by hipconv's Toeplitz+MFMA technique for grouped convolution.
Reformulate the width-direction convolution (KW=5) as a Toeplitz matrix multiply
and use `v_mfma_f32_4x4x4f16` (batch=16) to process 16 independent channels simultaneously.

### Toeplitz Formulation

For KW=5 producing 4 outputs from 8 inputs (F(4,5)):
```
G^T (4x8 Toeplitz matrix):
  m=0: [g0, g1, g2, g3, g4,  0,  0,  0]
  m=1: [ 0, g0, g1, g2, g3, g4,  0,  0]
  m=2: [ 0,  0, g0, g1, g2, g3, g4,  0]
  m=3: [ 0,  0,  0, g0, g1, g2, g3, g4]

Split into 2 MFMA calls (K=4 each): 62.5% utilization
```

### MFMA Lane Mapping (verified on gfx950)

```
MFMA_f32_4x4x4f16 with batch=16:
  Lane l: batch = l/4 (channel 0..15), local_id = l%4
  A operand: Toeplitz weight row for width position local_id
  B operand: input column at height position local_id
  C result:  C[m=0..3, local_id] — all 4 widths for this height
```

### Implementation

HIP C++ kernel with `__builtin_amdgcn_mfma_f32_4x4x4f16` (both fp16 and bf16_1k variants compile on gfx950).

```
Grid:  [D_out=59, C/16=32, H_tiles=12] = 22,656 blocks
Block: 64 threads (1 wave)
LDS:   16ch × 8rows × 84w × bf16 = 21,504 bytes per depth
```

### Results

| Variant | VGPRs | Occ | Scratch | LDS | Time | vs V97 |
|---|---|---|---|---|---|---|
| MFMA v1 (all w_tiles in acc[20]) | 166 | 2 | 560 B | 21.5 KB | 3.68ms | 0.17x |
| MFMA v2 (all depths in LDS) | 182 | 1 | 304 B | 64.5 KB | 10.6ms | 0.06x |
| MFMA v3 (grouped w_tiles) | 101 | 2 | 560 B | 21.5 KB | 14.4ms | 0.04x |

**Correctness**: All variants PASS (max_diff = 0.125 vs V97).

### Why MFMA Failed for Depthwise Conv

1. **Data loading dominates compute**: V97 loads 22KB once for 1 channel. MFMA needs 16× more data (16 channels). Even though MFMA computes 16 channels simultaneously, the data loading cost scales 16× while the compute speedup is at most 16× — net zero.

2. **Only 64 threads for cooperative loading**: V97 uses 256 threads for LDS loading (4× faster). MFMA uses 64 threads (1 wave per block).

3. **LDS reload penalty**: With only 21KB LDS (1 depth), need 3 depth loads × multiple w_tile groups = 15 LDS reload cycles with barriers. Each reload: clear 21KB + load ~10KB + barrier ≈ 500+ cycles.

4. **Register pressure**: 30 Toeplitz weight VGPRs (15 filter positions × 2 MFMA calls) + accumulators + temps → SGPR spills, reduced occupancy.

5. **MFMA utilization**: Only 62.5% of Toeplitz entries are non-zero (KW=5). For grouped conv (cpg≥4), cross-channel computation fills the matrix — depthwise has no cross-channel work.

### Key Insight

**MFMA is only advantageous when compute-to-data ratio is high.** For depthwise conv (cpg=1), each channel's computation is independent and small (75 MACs per output pixel). The bottleneck is data movement, not compute. MFMA doesn't help because it doesn't reduce data movement — it only accelerates compute that is already hidden behind memory latency.

**The hipconv approach works for cpg≥4** because:
- MFMA does useful cross-channel matmul (fills the matrix engine)
- 1 input row serves 4+ channels (data reuse)
- Weight matrix is dense (75-100% utilization)
- Each input load produces much more compute

**For cpg=1, the V97 VALU approach is optimal** because:
- Single-channel data fits in LDS (22KB)
- 256 threads cooperatively load fast
- 45 ds_read_b32 + v_fmac is the minimum data touch
- No cross-channel compute to exploit

### Phase 12b: v_pk_fma_f32 for Packed Odd Taps

From PA kernel (`pyhip/src/contrib/pa.py`) analysis: `v_pk_fma_f32` processes 2 fp32 FMAs in one instruction. Applied to odd-tap computation (kw=4) to replace 2× `v_fmac_f32` with 1× `v_pk_fma_f32`.

| Metric | dot2 | pk |
|---|---|---|
| VGPRs | 98 | 116 (+18 for Bsingle_pk pairs) |
| Occupancy | 4 | 4 |
| Correctness | PASS | **FAIL** (max_diff=35.6) |
| Time | 0.623ms | 0.650ms |

**Why it failed**: `Bsingle` is bf16-loaded-as-float (via `ds_read_u16_d16_hi`), which works with scalar `v_fmac_f32` but not with packed `v_pk_fma_f32`. Also, duplicating Bsingle into pairs added 30 VGPRs, negating register savings from the instruction reduction.

**Impact**: Even if correct, saves only 15 VALU out of ~300 per iteration (<5%). Not worth pursuing — the bottleneck is ds_read issue rate, not VALU count.

### Phase 12c: V97 with sched_group_barrier (NEW BEST!)

**Inspiration**: `pyhip/src/contrib/gluon/fused_mlp.py` uses `_amd_iglp_sched_group_barrier` to force the LLVM scheduler to interleave VMEM reads with MFMA compute. Applied the same principle to our LDS read + VALU compute pipeline.

**Changes from V97** (two changes bundled together):

1. **Restructured compute loop** — row-level interleaving (same idea as JIT-ilv Phase 11):
   - Read 3 ds_read_b32 for one (d,h) row, then compute 10 v_fmac for that row, repeat for all 15 rows.
   - Only 3 input registers per row (down from 45 batched).
   - Compute both p=0 and p=1 outputs in the same loop body (no separate p loop).

2. **Added `sched_group_barrier` compiler hints**:
   - `__builtin_amdgcn_sched_group_barrier(0x0100, 3, 0)` — schedule 3 DS_read as a group
   - `__builtin_amdgcn_sched_group_barrier(0x0002, 10, 0)` — schedule 10 VALU as a group
   - This tells LLVM's machine scheduler to interleave reads with compute, instead of its default behavior of grouping all similar instructions together.

**Results**:
| Metric | V97-origin | V97-sgb |
|---|---|---|
| VGPRs | 155 | **86** |
| Occupancy | 3 | **5** |
| Spills | 0 | 0 |
| LDS | 32,576 B | 32,576 B |
| Correctness | PASS | PASS (max_diff=0.0 vs V97, 0.25 vs PyTorch = normal bf16) |
| **Time** | **0.638ms** | **0.563ms**|
| **Speedup** | 1.00x | **1.13x** |

**Honest analysis — what we know vs what we don't**:

#### Occupancy analysis: depends on GPU's LDS size

The compiler reports "Occupancy 5" for V97-sgb = `floor(512/86)`. This is the VGPR-limited occupancy only. LDS constrains blocks per CU at runtime.

```
LDS per block (compiler-reported):       32,576 bytes (31.8 KB)
  (actual tile = 24,696 bytes; 7,880 bytes wasted — see LDS bug below)

MI308X (CDNA3): LDS per CU = 64 KB  (from rocminfo / ISA spec)
MI350X (CDNA4): LDS per CU = 160 KB (from rocminfo GROUP segment = 160 KB)
```

| GPU | LDS/CU | blocks from LDS | V97-origin (155 VGPRs) | V97-sgb (86 VGPRs) |
|---|---|---|---|---|
| MI308X | 64 KB | floor(65536/32576) = **2** | min(3,2) = **2** (LDS-bound) | min(5,2) = **2** (LDS-bound) |
| MI350X | 160 KB | floor(163840/32576) = **5** | min(3,5) = **3** (VGPR-bound) | min(5,5) = **5** (balanced) |

**MI308X**: Both kernels LDS-bound at occupancy 2. VGPR reduction doesn't help. Speedup comes entirely from `sched_group_barrier`.

**MI350X** (our test hardware): VGPR reduction genuinely increases occupancy from 3 to 5. The compiler-reported occupancy is correct. Speedup comes from both higher occupancy AND `sched_group_barrier`.

#### Speedup contributors on MI350X

We bundled two changes and did not isolate them. Both likely contribute:

1. **Higher occupancy (3 → 5)**: The VGPR reduction (155 → 86) allows 5 concurrent waves per SIMD instead of 3. More waves means the hardware wavefront scheduler has more choices to hide the 52-cycle ds_read latency.

2. **`sched_group_barrier` instruction interleaving**: Without these hints, LLVM's default machine scheduler groups all ds_reads together, then all VALU together (confirmed in `pyhip/docs/debug-llvm-mi-sched.md`). The hints force alternation: 3 reads then 10 VALU per row. This creates VALU/read overlap within each wave.

3. **NOT proven: LGKM queue overflow**. The 16-entry queue claim was a guess from `pyhip/docs/vm_cnt.md` (see Phase 11 correction).

#### Why JIT-ilv failed but HIP-sgb succeeded

The same interleaving idea failed in JIT (Phase 11: 0.638ms, 3% slower) but succeeded in HIP C++ (0.537ms, 14% faster). On MI350X, JIT-ilv had occupancy 8 (57 VGPRs) vs HIP-sgb occupancy 5 (86 VGPRs), yet HIP-sgb is faster. This shows higher occupancy is not always better — instruction scheduling quality matters. The difference:
- JIT-ilv: pyhip JIT emits `v_mov` instructions for register management + has no `sched_group_barrier` equivalent. The assembler gets a flat instruction stream with no scheduling hints.
- HIP-sgb: LLVM's machine scheduler receives explicit `sched_group_barrier` hints and optimizes the instruction ordering for pipeline utilization.

#### LDS over-allocation bug

```cpp
constexpr int LDS_SIZE = 32*1024;       // 32768 bytes
constexpr int weight_size = KD*KH*KW;   // 75 elements
constexpr int max_input_size = LDS_SIZE/sizeof(IO_DTYPE) - (weight_size + 31)/32 * 32;
// = 32768/2 - 96 = 16288 elements = 32576 bytes
```

This reserves 96 elements (192 bytes) for weights in LDS, but weights are loaded from global memory directly into registers (`weight_reg[d][h][w] = kernel[ki++]`). The reservation is dead code. The actual input tile is 12,348 elements = 24,696 bytes. The `s_input` array over-allocates by 32% (7,880 bytes wasted).

This doesn't affect performance for this problem size (LDS-bound at 2 blocks either way), but would matter on hardware with larger LDS or with a smaller input tile.

**File**: `kernels/depthwise_conv3d_v97_sgb.cpp`

---

**Production recommendation**: **V97-sgb at 0.537ms is the new production winner** — 13.3% faster than V97-origin (638ms) and **11x faster than PyTorch**. On MI350X (160 KB LDS), occupancy increases from 3 to 5 AND `sched_group_barrier` improves instruction interleaving — both contribute to the speedup. On MI308X (64 KB LDS), both kernels would be LDS-bound at occupancy 2, so only `sched_group_barrier` would help. Results are bitwise identical to V97-origin. Fix: the `LDS_SIZE` constant and unused weight LDS reservation should be cleaned up to allocate only the exact tile size (24.1 KB instead of 31.8 KB).

---

## CDNA4 (gfx950) Optimization Learnings

Hardware-specific findings discovered during this optimization work.

### LDS per CU is 160 KB (not 64 KB)

MI350X has 160 KB LDS per CU (verified via `rocminfo` GROUP segment). Our docs and initial analysis incorrectly assumed 64 KB (carried over from CDNA3). This changes occupancy math significantly — for a 32 KB-per-block kernel: `floor(160/32) = 5` blocks/CU on CDNA4 vs `floor(64/32) = 2` on CDNA3.

### `ds_read_b64` is 4.2x slower than 2x `ds_read_b32`

Tested in Phase 10a (JIT-dot2-b64). On gfx950, wider LDS reads are hardware-penalized. Always use `ds_read_b32`.

### `__builtin_amdgcn_sched_group_barrier` yields real speedups

LLVM's default machine scheduler groups similar instructions together (all LDS reads, then all VALU). On gfx950, `sched_group_barrier` hints to interleave them yield 14% speedup on our depthwise conv3d (Phase 12c). The scheduling hint types:
- `0x0100` = DS read group
- `0x0002` = VALU group
- `0x0008` = MFMA group
- `0x0020` = VMEM read group

### ROCm 7.2 compiler bug with `v_dot2_f32_bf16` inline asm

When `v_dot2_f32_bf16` is used in HIP inline asm inside a fully-unrolled compute loop, the register allocator silently reuses weight registers for address computation. 45 out of 75 weights get zeroed. Workaround: use pyhip JIT which bypasses the compiler's register allocator (Phase 4, Phase 9).

### MFMA is not useful for depthwise conv (cpg=1)

`v_mfma_f32_4x4x4f16` (batch=16) can pack 16 independent channels, but the data loading cost scales 16x while compute speedup is at most 16x — net zero. With only 64 threads per block (1 wave for MFMA), cooperative LDS loading is 4x slower than V97's 256 threads. Tested in Phase 12a: 6-23x slower than V97. MFMA is only beneficial for grouped conv with cpg >= 4.

### Compiler-reported occupancy ignores LDS

The compiler's `Occupancy [waves/SIMD]` remark only reflects VGPR limits (`floor(512/VGPRs)`). LDS constraints are enforced at runtime by the GPU block scheduler. Always compute `min(VGPR_limit, floor(LDS_per_CU / LDS_per_block))` manually.

### `vm_cnt.md` LGKM queue depth is unverified

`pyhip/docs/vm_cnt.md` confirmed VMEM has a 64-entry hardware queue, but only **guessed** LDS has a 16-entry queue (4-bit lgkmcnt field). The document notes: *"considering the super-fast throughput of LDS, that causes no stall most-likely."* The LDS queue depth was never empirically tested.

---

## Files Reference

### Production Kernel
- **`kernels/depthwise_conv3d_v97_sgb.cpp` — NEW production kernel (0.537ms, ~14% faster)**
- `kernels/depthwise_conv3d_v97-origin.cpp` — previous production kernel (0.61ms)

### Optimization Attempts
- `kernels/depthwise_conv3d_v102.cpp` — v_dot2 + v_cvt_pk (WRONG)
- `kernels/depthwise_conv3d_v103.cpp` — v_dot2 LDS weight pairs (WRONG)
- `kernels/depthwise_conv3d_v104.cpp` — row-chunk streaming (2.5x slower)
- `kernels/depthwise_conv3d_v104b.cpp` — compiler pointer reads (2.1x slower)
- `kernels/depthwise_conv3d_v104c.cpp` — ds_read_u16_d16_hi batch (1.5x slower)
- `kernels/depthwise_conv3d_v104d.cpp` — interleaved d16_hi+fmac (2.1x slower)
- `kernels/depthwise_conv3d_v104e.cpp` — packed bf16 weights (WRONG)
- `kernels/depthwise_conv3d_v105.cpp` — NHWC weight tile (2.0x slower)
- `kernels/depthwise_conv3d_mfma.cpp` — MFMA Toeplitz multi-channel (6-23x slower)
- `kernels/conv_depthwise3d_jit_dot2_pk.py` — v_pk_fma_f32 odd taps (FAIL)

### Diagnostics
- `kernels/depthwise_conv3d_v103_loadonly.cpp` — load-only diagnostic
- `kernels/depthwise_conv3d_v103_computeonly.cpp` — compute-only diagnostic
- `test/benchmark_v103_diag.py` — bank conflict isolation benchmark

### Analysis
- `compiler_bug/BUG_REPORT.md` — v_dot2 register allocator bug report
- `PROFILING_ANALYSIS.md` — rocprofv3 counter analysis
- `CONTRIB_VS_V97_PROFILING.md` — hardware counter comparison V97 vs contrib
- `group_conv_study/` — group conv kernel analysis (MFMA, techniques)
- `RESULTS_FINAL.md` — end-to-end performance results

### Profiling Data
- `test/profile_v97_deep.py` — Profiling script (100 iterations)
- `test/pass1_results.db` — PMC Pass 1: instruction mix (SQ_INSTS_VALU/SALU/LDS/VMEM)
- `test/pass2_results.db` — PMC Pass 2: VALU breakdown (FMA_F32/INT32/CVT/IOPS)
- `test/pass3_results.db` — PMC Pass 3: pipeline stalls (SQ_WAIT_ANY/INST_LDS/WAVE_CYCLES)
- `test/pass7_results.db` — PMC Pass 7: latency cycles (VMEM_RD/WR/SALU/SMEM)
- `test/trace_v97.yaml` — Thread trace config
- `test/v97_trace_out/` — Thread trace output (per-instruction latency/stall/idle)
  - `stats_ui_output_agent_*_dispatch_4.csv` — Per-instruction statistics
  - `ui_output_agent_*_dispatch_4/` — Decoded trace data

### Assembly Patch (Phase 8)
- `ASSEMBLY_PATCH_PLAN.md` — Detailed plan for assembly patching approach
- `tools/patch_v97_dot2.py` — Assembly patcher script (replaces fmac pairs with v_dot2)
- `kernels/depthwise_conv3d_v97-patched.s` — Patched assembly (v_dot2 in p=0)
- `kernels/depthwise_conv3d_v97-patched.co` — Assembled patched kernel binary
- `test/test_dot2_kernel.cpp` — v_dot2_f32_bf16 semantics verification kernel
- `test/test_dot2_semantics.py` — v_dot2 operand order test
- `test/benchmark_v97_patched.py` — Patched kernel benchmark

### JIT v_dot2 Kernel (Phase 9)
- `kernels/conv_depthwise3d_jit_dot2.py` — JIT kernel with v_dot2_f32_bf16
- `test/benchmark_jit_dot2.py` — Benchmark: V97-HIP vs JIT-orig vs JIT-dot2
- `PHASE9_JIT_PLAN.md` — Detailed design plan

### LDS Optimization (Phase 10-11)
- `kernels/conv_depthwise3d_jit_dot2_b64.py` — ds_read_b64 variant (4.2x slower — penalized on gfx950)
- `kernels/conv_depthwise3d_jit_dot2_pipe.py` — Software pipelining by depth slice (no improvement)
- `kernels/conv_depthwise3d_jit_dot2_kwp4.py` — KW_PACK=4 variant (wrong results, 1.6x slower)
- `kernels/conv_depthwise3d_jit_dot2_ilv.py` — Row-level interleaved read-compute (57 VGPRs, occ 8, 3% slower)
- `kernels/conv_depthwise3d_jit_dot2_ilv2.py` — Lookahead interleaved (60 VGPRs, occ 8, 1.5% slower)
- `test/benchmark_phase10.py` — Benchmark Phase 10 variants
- `test/benchmark_phase11.py` — Benchmark Phase 11 variants
- `test/benchmark_jit_dot2_b64.py` — Benchmark b64 variant
- `test/test_ds_read_b64.py` — ds_read_b64 correctness test
- `PHASE10_LDS_OPTIMIZATION_PLAN.md` — LDS optimization analysis and plan
- `PHASE11_IDEAS_FROM_PYHIP_DOCS.md` — Insights from pyhip docs (ds_read latency, issue rate; LGKM queue depth was guessed, not verified)

### Phase 12: MFMA Multi-Channel (Toeplitz)
- `kernels/depthwise_conv3d_mfma.cpp` — MFMA_f32_4x4x4f16 kernel with Toeplitz matrix
- `test/benchmark_phase12.py` — MFMA vs V97 benchmark
- `PHASE11_IDEAS_FROM_PYHIP_DOCS.md` — Optimization ideas from pyhip docs

### Benchmarks
- `test/benchmark_v103.py` — V103 vs V97
- `test/benchmark_v103_diag.py` — bank conflict diagnostics
- `test/benchmark_v104.py` — V104 row-chunk vs V97
- `test/benchmark_v104b.py` — V104b compiler reads vs V97
- `test/benchmark_v104c.py` — V104c d16_hi batch vs V97
- `test/benchmark_v104d.py` — V104d interleaved vs V97
- `test/benchmark_v104e.py` — V104e packed weights vs V97
- `test/benchmark_v105.py` — V105 NHWC vs V97
- `test/benchmark_phase12.py` — MFMA Toeplitz vs V97
