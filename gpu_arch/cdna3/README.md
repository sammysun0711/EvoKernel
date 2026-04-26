# AMD CDNA3 Architecture (MI300 Series)

Documentation for AMD Instinct MI300 series GPUs based on the CDNA3 architecture.

## Supported Hardware

- **MI300X**: 8x XCD (Accelerator Compute Die), 304 CUs, 192GB HBM3
- **MI300A**: APU variant with integrated CPU + GPU
- **MI308X**: 4x XCD (Accelerator Compute Die), 80 CUs, 192GB HBM3

## Key Documents

| Document | File | Description |
|----------|------|-------------|
| ISA Reference | [cdna3-isa.pdf](cdna3-isa.pdf) | Complete instruction set architecture |
| Architecture Whitepaper | [cdna3-whitepaper.pdf](cdna3-whitepaper.pdf) | High-level architecture overview |

## Quick Specifications (MI308X)

| Parameter | Value |
|-----------|-------|
| XCDs | 4 |
| Compute Units | 80 (4 XCDs x 20 CUs/XCD) |
| Stream Processors | 5,120 (80 CUs x 64 SPs/CU) |
| Clock Speed | 1420 MHz (base) / 2100 MHz (boost) |
| Wavefront Size | 64 threads |
| LDS per CU | 64 KB |
| VGPRs per CU | 512 (32-bit) |
| SGPRs per CU | 800 |
| L2 Cache | 4 MB per XCD (16 MB total, private to each XCD) |
| LLC Cache | 256 MB (shared across all XCDs) |
| Memory | 192 GB HBM3 |
| Memory Bandwidth | 5.3 TB/s |
| FP32 Peak | ~43 TFLOPS |
| BF16 Peak | ~172 TFLOPS |
| FP16 Peak | ~172 TFLOPS |
| INT8 Peak | ~344 TOPS |

## Ridge Points (Roofline Analysis)

The ridge point is where the roofline transitions from memory-bound to compute-bound.

### MI308X (4 XCDs, 80 CUs)

| Dtype | Peak (TFLOPS) | Ridge Point (FLOPs/byte) |
|-------|---------------|--------------------------|
| FP32 | ~43 | ~8.1 |
| BF16 | ~172 | ~32.5 |
| FP16 | ~172 | ~32.5 |
| INT8 | ~344 | ~65.0 |

### MI300X (8 XCDs, 304 CUs) - For Comparison

| Dtype | Peak (TFLOPS) | Ridge Point (FLOPs/byte) |
|-------|---------------|--------------------------|
| FP32 | 163 | 30.8 |
| BF16 | 653 | 123.2 |
| FP16 | 653 | 123.2 |
| INT8 | 1306 | 246.4 |

**Interpretation**: If your kernel's arithmetic intensity (FLOPs/byte) is below the ridge point, it's memory-bound and should focus on bandwidth optimization.

**Note**: MI308X has the same memory bandwidth as MI300X but fewer CUs, so workloads become memory-bound at a lower arithmetic intensity.

## Key Architecture Features

### Compute Units (CUs)

**MI300X**: 304 CUs organized in 8 XCDs (38 CUs per XCD)
**MI308X**: 80 CUs organized in 4 XCDs (20 CUs per XCD)

- Each CU has 4 SIMD units
- Each SIMD processes 16 work-items per cycle (64-wide wavefront over 4 cycles)
- 4 concurrent wavefronts per SIMD = 16 waves per CU maximum

### Memory Hierarchy

Memory operations are ordered from fastest to slowest:

```
Thread
  └── VGPRs (Vector General Purpose Registers)
      └── LDS (64 KB per workgroup) - Very fast
          └── L1 Cache (16 KB per CU) - Fast
              └── L2 Cache (4 MB per XCD, private) - Moderate
                  └── LLC Cache (256 MB shared across XCDs)
                      └── HBM3 (192 GB, 5.3 TB/s bandwidth)
```

**Cache Architecture Notes**:
- L2 is private to each XCD (16 MB total for MI308X, 32 MB for MI300X)
- LLC (Last Level Cache) at 256 MB is shared across all XCDs
- Cross-XCD communication goes through LLC

**Optimization Tip**: Exact latencies vary by workload. Focus on minimizing global memory accesses and maximizing register and LDS usage.

### Memory Coalescing

- 64-byte cache line
- Adjacent threads in a wavefront should access adjacent memory for coalescing
- Perfect coalescing: threads 0-63 access bytes 0-127 (2 bytes each, BF16)
- Scattered access can reduce bandwidth by 10-30x

### LDS (Local Data Share)

- 64 KB per workgroup (compute unit)
- 32 banks, each 4 bytes wide
- Bank conflicts cause serialization
- Much faster than global memory access
- Bandwidth: ~10 TB/s (theoretical)

## ISA Quick Reference

### Common Instructions

| Instruction | Description | Throughput |
|-------------|-------------|------------|
| `v_fma_f32` | FMA (fused multiply-add) | 1/cycle |
| `v_add_f32` | Floating-point add | 1/cycle |
| `v_mul_f32` | Floating-point multiply | 1/cycle |
| `global_load_dword` | 32-bit global load | 1/cycle |
| `ds_read_b32` | 32-bit LDS read | 1/cycle |
| `s_waitcnt` | Synchronization | N/A |

**Note**: Consult the ISA documentation for detailed instruction characteristics. Memory instruction performance depends heavily on access patterns and cache behavior.

### Vectorized Loads

- `global_load_dwordx4`: Load 4x32-bit (128 bits) in one instruction
- Use vectorized loads when accessing consecutive memory
- Reduces instruction count and improves efficiency

## Optimization Guidelines

### For Memory-Bound Kernels

1. **Layout Transformation**: Use NHWC over NCHW for channel-last access patterns
2. **Coalescing**: Ensure adjacent threads access adjacent memory
3. **LDS Usage**: Cache frequently accessed data in LDS
4. **Cooperative Loading**: Use workgroup threads to load data collaboratively

### For Compute-Bound Kernels

1. **Occupancy**: Target 4+ waves per SIMD for latency hiding
2. **Register Pressure**: Keep VGPR usage under 128 to allow 4+ waves
3. **Loop Unrolling**: Use `#pragma unroll` for small loops
4. **Vectorization**: Use vector types (float4, bfloat162) where possible

### General Best Practices

- Avoid divergent branches within wavefronts
- Use `__syncthreads()` sparingly (expensive barrier)
- Prefer FMA over separate multiply + add
- Pre-compute constant expressions

## See Also

- [Quick Reference Cards](../quick_reference.md)
- [GPU Architecture Reference Guide](../../guides/gpu_architecture_reference.md)
- [Depthwise Conv3D Optimization](../../../kernels/depthwise_conv3d/README.md) - Real CDNA3 optimization example
