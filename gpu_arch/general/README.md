# General GPU Architecture Concepts

This directory contains cross-generation documentation and general GPU optimization concepts that apply to multiple architectures.

## Core Concepts

### Roofline Model

The roofline model helps determine whether a kernel is memory-bound or compute-bound:

```
         Compute Ceiling (peak TFLOPS)
        _________________________________
       /
      /
     /  <- Roofline
    /
   /
  /________________________
  ^                        ^
  |                        |
  Memory-Bound            Compute-Bound
  (low AI)                (high AI)

AI = Arithmetic Intensity = FLOPs / Bytes
```

**Ridge Point** = Peak TFLOPS / Memory Bandwidth (GB/s)

### Memory Hierarchy (General)

Most modern GPUs have similar hierarchies:

1. **Registers** - Fastest, per-thread
2. **Shared/Local Memory (LDS/SMEM)** - Fast, per-workgroup
3. **L1 Cache** - Per compute unit/SM
4. **L2 Cache** - Unified across GPU
5. **Global Memory (HBM/GDDR)** - Slowest, largest

### Thread Organization (AMD Terminology)

| AMD Term | Description |
|----------|-------------|
| Work-item | Single execution unit |
| Wavefront | Group of 64 threads executing together |
| Workgroup | Threads sharing local memory (LDS) |
| Grid | All threads for a kernel launch |

### Memory Access Patterns

**Coalesced Access** (Fast):
```
Thread 0: memory[0]
Thread 1: memory[1]
Thread 2: memory[2]
...
```

**Strided Access** (Slower):
```
Thread 0: memory[0]
Thread 1: memory[512]
Thread 2: memory[1024]
...
```

**Random Access** (Slowest):
```
Thread 0: memory[random]
Thread 1: memory[random]
...
```

## Optimization Checklist

### Memory-Bound Kernels

- [ ] Memory layout matches access pattern (NCHW vs NHWC)
- [ ] Adjacent threads access adjacent memory
- [ ] Data reused from local/shared memory when possible
- [ ] Vectorized loads where applicable
- [ ] Minimal redundant memory access

### Compute-Bound Kernels

- [ ] Sufficient occupancy for latency hiding
- [ ] Register usage within limits
- [ ] No excessive register spills
- [ ] Loop unrolling applied
- [ ] FMA instructions used where possible

### General

- [ ] No divergent branches within wavefront/warp
- [ ] Synchronization minimized
- [ ] Constant memory for read-only data
- [ ] Pre-computed indices where possible

## AMD Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/)
- [rocprof Profiler](https://rocm.docs.amd.com/projects/rocprofiler/)

## See Also

- [CDNA3 Documentation](../cdna3/README.md)
- [CDNA4 Documentation](../cdna4/README.md)
- [GPU Architecture Reference Guide](../../guides/gpu_architecture_reference.md)
