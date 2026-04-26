# AMD CDNA4 Architecture (MI350X Series)

Documentation for AMD Instinct MI350X series GPUs based on the CDNA4 architecture.

## Supported Hardware

- **MI350X**: Datacenter accelerator (gfx950)

## Key Documents

| Document | File | Description |
|----------|------|-------------|
| ISA Reference | [cdna4-isa.pdf](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) | Complete instruction set architecture |
| Architecture Whitepaper | [cdna4-whitepaper.pdf](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf) | High-level architecture overview |

## Hardware Specs (MI350X, verified via `rocminfo`)

```
Architecture:           CDNA4 (gfx950)
GPU Compute Dies (GCD): 8
Compute Units per GCD:  256
SIMDs per CU:           4
Wavefront Size:         64 threads
Max Waves per CU:       32
Max Waves per SIMD:     8
Max Clock Freq:         2200 MHz
VGPRs per SIMD:         512
Workgroup Max Size:     1024 threads

LDS per CU:             160 KB
Cacheline Size:         128 bytes

HBM per GCD:            ~252 GB (HBM3E)
Memory Bandwidth:       ~8 TB/s
```

Source: `rocminfo` on MI350X, GROUP segment = 160 KB.

## Comparison with CDNA3

| Feature | CDNA3 (MI300X) | CDNA4 (MI350X) |
|---------|----------------|----------------|
| GPU Architecture | gfx942 | gfx950 |
| LDS per CU | 64 KB | 160 KB |
| SIMDs per CU | 4 | 4 |
| VGPRs per SIMD | 512 | 512 |
| Max Waves per CU | 32 | 32 |
| HBM | HBM3, 192 GB | HBM3E, ~252 GB per GCD |
| Memory Bandwidth | 5.3 TB/s | ~8.0 TB/s |
| Max Clock Freq | 2100 MHz | 2200 MHz |
| Compute Units | 304 (8 XCDs x 38) | 256 per GCD, 8 GCDs |
| Cacheline Size | 64 bytes | 128 bytes |

## Instructions (verified on gfx950)

Available (not exhaustive — see ISA PDF for full list):
- `v_dot2_f32_bf16` — packed 2x bf16 FMA
- `v_pk_fma_f32` — packed 2x fp32 FMA
- `v_pk_mul_f32` / `v_pk_add_f32` — packed fp32 arithmetic
- `v_mfma_f32_4x4x4f16` — 4x4x4 MFMA, batch=16 (fp16 input, fp32 output)
- `v_mfma_f32_4x4x4bf16_1k` — 4x4x4 MFMA, batch=16 (bf16 input)
- `v_mfma_f32_16x16x32_bf16` — 16x16x32 MFMA (bf16)
- `__builtin_amdgcn_sched_group_barrier(type, count, mask)` — instruction scheduling hint

## See Also

- [CDNA3 Documentation](../cdna3/README.md) - Previous generation reference
- [Quick Reference Cards](../quick_reference.md)
