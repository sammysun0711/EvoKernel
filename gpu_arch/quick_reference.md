# GPU Quick Reference Cards

One-page specifications for AMD MI-series GPUs. Use these for quick roofline calculations and optimization decisions.

## AMD Instinct MI308X (CDNA3)

```
Architecture: CDNA3 (gfx942)
┌─────────────────────────────────────────────────────────────┐
│  COMPUTE                                                     │
├─────────────────────────────────────────────────────────────┤
│  XCDs (Accelerator Compute Dies): 4                          │
│  Compute Units:        80 (4 XCDs x 20 CUs/XCD)              │
│  Stream Processors:    5,120 (80 CUs x 64 SPs/CU)            │
│  Wavefront Size:       64 threads                            │
│  Max Waves/CU:         16                                    │
│  Clock (Base/Boost):   1420 / 2100 MHz                       │
├─────────────────────────────────────────────────────────────┤
│  MEMORY                                                      │
├─────────────────────────────────────────────────────────────┤
│  LDS per CU:           64 KB                                 │
│  L1 Cache per CU:      16 KB                                 │
│  L2 Cache:             4 MB per XCD (16 MB total)            │
│  LLC Cache:            256 MB (shared across XCDs)           │
│  HBM3 Capacity:        192 GB                                │
│  Memory Bandwidth:     5,300 GB/s (5.3 TB/s)                 │
├─────────────────────────────────────────────────────────────┤
│  PEAK PERFORMANCE                                            │
├─────────────────────────────────────────────────────────────┤
│  FP32:                 ~43 TFLOPS                            │
│  BF16:                 ~172 TFLOPS                           │
│  FP16:                 ~172 TFLOPS                           │
│  INT8:                 ~344 TOPS                             │
├─────────────────────────────────────────────────────────────┤
│  RIDGE POINTS (FLOPs/byte)                                   │
├─────────────────────────────────────────────────────────────┤
│  FP32 Ridge:           ~8.1  FLOPs/byte                      │
│  BF16 Ridge:           ~32.5 FLOPs/byte                      │
│  FP16 Ridge:           ~32.5 FLOPs/byte                      │
│  INT8 Ridge:           ~65.0 FLOPs/byte                      │
└─────────────────────────────────────────────────────────────┘

ISA Reference: docs/gpu_arch/cdna3/cdna3-isa.pdf
Whitepaper:    docs/gpu_arch/cdna3/cdna3-whitepaper.pdf
```

**Quick Classification**:
- AI < 32.5 FLOPs/byte (BF16) => Memory-bound, optimize bandwidth
- AI > 32.5 FLOPs/byte (BF16) => Compute-bound, optimize occupancy

**Note**: MI308X has 4 XCDs vs MI300X's 8 XCDs. Same memory subsystem means
MI308X is more memory-bound relative to its compute capacity.

---

## AMD Instinct MI300X (CDNA3)

```
Architecture: CDNA3 (gfx942)
┌─────────────────────────────────────────────────────────────┐
│  COMPUTE                                                     │
├─────────────────────────────────────────────────────────────┤
│  XCDs (Accelerator Compute Dies): 8                          │
│  Compute Units:        304 (8 XCDs x 38 CUs/XCD)             │
│  Stream Processors:    19,456 (304 CUs x 64 SPs/CU)          │
│  Wavefront Size:       64 threads                            │
│  Max Waves/CU:         16                                    │
│  Clock (Base/Boost):   2100 MHz (boost)                      │
├─────────────────────────────────────────────────────────────┤
│  MEMORY                                                      │
├─────────────────────────────────────────────────────────────┤
│  LDS per CU:           64 KB                                 │
│  L1 Cache per CU:      16 KB                                 │
│  L2 Cache:             4 MB per XCD (32 MB total)            │
│  LLC Cache:            256 MB (shared across XCDs)           │
│  HBM3 Capacity:        192 GB                                │
│  Memory Bandwidth:     5,300 GB/s (5.3 TB/s)                 │
├─────────────────────────────────────────────────────────────┤
│  PEAK PERFORMANCE                                            │
├─────────────────────────────────────────────────────────────┤
│  FP32:                 163 TFLOPS                            │
│  BF16:                 653 TFLOPS                            │
│  FP16:                 653 TFLOPS                            │
│  INT8:                 1,306 TOPS                            │
├─────────────────────────────────────────────────────────────┤
│  RIDGE POINTS (FLOPs/byte)                                   │
├─────────────────────────────────────────────────────────────┤
│  FP32 Ridge:           30.8  FLOPs/byte                      │
│  BF16 Ridge:           123.2 FLOPs/byte                      │
└─────────────────────────────────────────────────────────────┘

ISA Reference: docs/gpu_arch/cdna3/cdna3-isa.pdf
Whitepaper:    docs/gpu_arch/cdna3/cdna3-whitepaper.pdf
```

---

## MI300 Series Variants

| Feature | MI308X | MI300X |
|---------|--------|--------|
| XCDs | 4 | 8 |
| Compute Units | 80 (20/XCD) | 304 (38/XCD) |
| Stream Processors | 5,120 | 19,456 |
| L2 Cache (per XCD) | 4 MB | 4 MB |
| Total L2 Cache | 16 MB | 32 MB |
| LLC Cache | 256 MB | 256 MB |
| HBM3 Memory | 192 GB | 192 GB |
| Memory Bandwidth | 5.3 TB/s | 5.3 TB/s |
| FP32 Peak | ~43 TFLOPS | 163 TFLOPS |
| BF16 Peak | ~172 TFLOPS | 653 TFLOPS |
| Ridge Point (BF16) | ~32.5 FLOPs/byte | ~123 FLOPs/byte |

**Note**: MI308X has fewer XCDs but same memory subsystem, making it more
memory-bound relative to MI300X.

---

## AMD Instinct MI350X (CDNA4)

```
Architecture: CDNA4 (gfx950)
Source: rocminfo on MI350X hardware
┌─────────────────────────────────────────────────────────────┐
│  COMPUTE                                                     │
├─────────────────────────────────────────────────────────────┤
│  GPU Compute Dies (GCD):  8                                  │
│  Compute Units per GCD:   256                                │
│  SIMDs per CU:            4                                  │
│  Wavefront Size:          64 threads                         │
│  Max Waves per CU:        32                                 │
│  Max Waves per SIMD:      8                                  │
│  VGPRs per SIMD:          512                                │
│  Clock (Max):             2200 MHz                           │
├─────────────────────────────────────────────────────────────┤
│  MEMORY                                                      │
├─────────────────────────────────────────────────────────────┤
│  LDS per CU:              160 KB  *** 2.5x larger than CDNA3 │
│  Cacheline Size:          128 bytes                          │
│  HBM3E per GCD:           ~252 GB                            │
│  Memory Bandwidth:        ~8.0 TB/s                          │
└─────────────────────────────────────────────────────────────┘
```

**Key difference from CDNA3**: LDS is 160 KB per CU (vs 64 KB). This means LDS
is much less likely to be the occupancy bottleneck — VGPRs typically bind first.

---

## Ridge Point Formula

```
Ridge Point = Peak Compute (FLOPS) / Memory Bandwidth (Bytes/s)

Example (MI300X BF16):
  Ridge = 653 TFLOPS / 5.3 TB/s
        = 653 x 10^12 / 5.3 x 10^12
        = 123.2 FLOPs/byte

Example (MI308X BF16):
  Ridge = 172 TFLOPS / 5.3 TB/s
        = 172 x 10^12 / 5.3 x 10^12
        = 32.5 FLOPs/byte
```

## Arithmetic Intensity Formula

```
Arithmetic Intensity (AI) = Total FLOPs / Total Memory Bytes

Example (Depthwise Conv3D):
  FLOPs  = 16.31 GFLOPs
  Memory = 440 MB
  AI     = 16.31 x 10^9 / 440 x 10^6
         = 37 FLOPs/byte

Classification: 37 < 123 => Memory-bound
```

## Quick Classification Guide

```
If AI < Ridge * 0.8:  MEMORY-BOUND
   Focus: Memory coalescing, layout optimization, LDS caching

If AI > Ridge * 1.2:  COMPUTE-BOUND
   Focus: Occupancy, ILP, vectorization, loop unrolling

Otherwise:            BALANCED
   Profile to determine actual bottleneck
```

## See Also

- [GPU Architecture Reference Guide](../guides/gpu_architecture_reference.md)
- [CDNA3 Details](cdna3/README.md)
- [CDNA4 Details](cdna4/README.md)
