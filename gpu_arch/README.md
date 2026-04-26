# GPU Architecture Documentation

This directory contains GPU architecture documentation and references for kernel optimization. Understanding the target hardware is essential for effective kernel development.

## Directory Structure

```
gpu_arch/
├── README.md              # This file
├── quick_reference.md     # One-page specs for all supported GPUs
├── cdna3/                 # AMD Instinct MI300 series (CDNA3)
│   ├── README.md
├── cdna4/                 # AMD Instinct MI400 series (CDNA4)
│   ├── README.md
└── general/               # Cross-generation documentation
    └── README.md
```

## Quick Links

### CDNA3 (MI300 Series)

| Document | Description | Use When |
|----------|-------------|----------|
| [CDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) | Complete instruction reference | Instruction-level optimization, understanding execution model |
| [CDNA3 Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf) | Architecture overview | Understanding memory hierarchy, compute units |

### CDNA4 (MI355X Series)

| Document | Description | Use When |
|----------|-------------|----------|
| [CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) | Complete instruction reference | Instruction-level optimization for MI400 |
| [CDNA4 Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf) | Architecture overview | Understanding new features, memory improvements |

## How to Use This Documentation

### For Roofline Analysis

1. Check [quick_reference.md](quick_reference.md) for peak compute and bandwidth numbers
2. Calculate your kernel's arithmetic intensity
3. Use the ridge point to determine if memory-bound or compute-bound

### For Memory Optimization

1. Read the whitepaper sections on memory hierarchy
2. Understand L1/L2 cache sizes and behavior
3. Review LDS (Local Data Share) specifications

### For Instruction-Level Optimization

1. Consult the ISA document for:
   - Instruction latencies
   - Instruction throughput
   - Special instructions (e.g., vectorized loads)
   - Register usage guidelines

## Key Architectural Differences

| Feature | CDNA3 (MI300X) | CDNA4 (MI355X) |
|---------|----------------|----------------|
| Architecture | gfx942 | gfx950 |
| Wavefront Size | 64 | 64 |
| LDS per CU | 64 KB | 160 KB |
| L2 Cache | 256 MB | TBD |
| HBM Type | HBM3 | HBM3E |
