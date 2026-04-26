#!/usr/bin/env python3
"""
EvoKernel Benchmark: 5-step depthwise conv3d optimization journey.

Compares PyTorch F.conv3d against 5 HIP C++ kernels, each representing
a distinct algorithmic approach.

Usage:
    python benchmark.py              # benchmark all steps
    python benchmark.py --steps 3 5  # benchmark specific steps
    python benchmark.py --iters 200  # more iterations for stable timing
"""
import sys, os, math, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from pyhip import module

# ─── Problem definition (from task.md) ───
B, C, D, H, W = 1, 512, 61, 45, 80
KD, KH, KW = 3, 5, 5
PAD = (0, 2, 2)
D_out, H_out, W_out = D - KD + 1, H, W  # 59, 45, 80

KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


def make_inputs(device, dt=torch.bfloat16):
    torch.manual_seed(42)
    inp = torch.randn(B, C, D, H, W, device=device, dtype=dt)
    wt = torch.randn(C, 1, KD, KH, KW, device=device, dtype=dt)
    bias = torch.randn(C, device=device, dtype=dt)
    return inp, wt, bias


def pytorch_ref(inp, wt, bias):
    return F.conv3d(inp, wt, bias=bias, padding=PAD, groups=C)


def benchmark_fn(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


# ─── Step runners ───

def run_step1(inp, wt, bias, out):
    """Naive: 1 thread = 1 output, global memory, grid-stride loop."""
    src = os.path.join(KERNEL_DIR, "step1_naive.cpp")
    m = module(src, "-O3")
    total = B * C * D_out * H_out * W_out
    block = 256
    grid = [math.ceil(total / block)]
    m.conv_depthwise3d_cuda_kernel_reference_bf16(
        grid, [block],
        inp.data_ptr(), out.data_ptr(), wt.data_ptr(), bias.data_ptr(),
        B, C, C, D, H, W, D_out, H_out, W_out,
        KD, KH, KW, 1, 1, 1, PAD[0], PAD[1], PAD[2], 1, 1, 1)
    torch.cuda.synchronize()


def run_step2(inp_nhwc, wt_nhwc, bias, out_nhwc):
    """NHWC: coalesced channel access, 8 channels/thread, no data reuse."""
    src = os.path.join(KERNEL_DIR, "step2_nhwc.cpp")
    m = module(src, "-O3")
    total_spatial = D_out * H_out * W_out
    grid = [math.ceil(total_spatial / 8)]  # 8 spatial per block
    m.conv_depthwise3d_v84_nhwc_vec_bf16(
        grid, [512],
        inp_nhwc.data_ptr(), out_nhwc.data_ptr(),
        wt_nhwc.data_ptr(), bias.data_ptr(), total_spatial)
    torch.cuda.synchronize()


def run_step3(inp, wt, bias, out):
    """NCHW + LDS cache: cooperative loading, weight registers, batched ds_read."""
    src = os.path.join(KERNEL_DIR, "step3_nchw_lds.cpp")
    flags = ("-O3 -DKD=3 -DKH=5 -DKW=5 -DPaddingD=0 "
             "-DPaddingH=2 -DPaddingW=2 -DBLOCK_H=45 -DBLOCK_W=80 "
             "-DIO_DTYPE=__hip_bfloat16")
    m = module(src, flags)
    m.conv_depthwise3d_hip(
        [B, C, D_out], [256],
        inp.data_ptr(), out.data_ptr(), wt.data_ptr(), bias.data_ptr(),
        C, D, H, W, C, D_out, H_out, W_out)
    torch.cuda.synchronize()


def run_step4(inp, wt, bias, out):
    """MFMA Toeplitz: 16-channel batching via matrix engine (failure case)."""
    src = os.path.join(KERNEL_DIR, "step4_mfma.cpp")
    flags = ("-O3 -DKD=3 -DKH=5 -DKW=5 "
             "-DPaddingH=2 -DPaddingW=2 -DBLOCK_H=45 -DBLOCK_W=80 -std=c++17")
    m = module(src, flags)
    H_tiles = math.ceil(H_out / 4)
    m.depthwise_conv3d_mfma(
        [D_out, C // 16, H_tiles], [64],
        inp.data_ptr(), out.data_ptr(), wt.data_ptr(), bias.data_ptr(),
        C, D, H, W, D_out, H_out, W_out)
    torch.cuda.synchronize()


def run_step5(inp, wt, bias, out):
    """sched_group_barrier + row-interleave: the breakthrough."""
    src = os.path.join(KERNEL_DIR, "step5_sgb.cpp")
    flags = ("-O3 -DKD=3 -DKH=5 -DKW=5 -DPaddingD=0 "
             "-DPaddingH=2 -DPaddingW=2 -DBLOCK_H=45 -DBLOCK_W=80 "
             "-DIO_DTYPE=__hip_bfloat16")
    m = module(src, flags)
    m.conv_depthwise3d_hip(
        [B, C, D_out], [256],
        inp.data_ptr(), out.data_ptr(), wt.data_ptr(), bias.data_ptr(),
        C, D, H, W, C, D_out, H_out, W_out)
    torch.cuda.synchronize()


# ─── NHWC layout helpers ───

def to_nhwc_3d(nchw):
    """NCHW [B,C,D,H,W] -> NHWC [B,D,H,W,C] -> flatten batch -> [D,H,W,C]"""
    return nchw.permute(0, 2, 3, 4, 1).contiguous().view(D, H, W, C)

def to_nhwc_3d_out(nchw_shape):
    return torch.empty(D_out, H_out, W_out, C, device="cuda", dtype=torch.bfloat16)

def nhwc_to_nchw(nhwc, shape):
    """[D,H,W,C] -> [1,C,D,H,W]"""
    return nhwc.view(1, shape[0], shape[1], shape[2], C).permute(0, 4, 1, 2, 3).contiguous()

def wt_to_nhwc(wt_nchw):
    """weight [C,1,KD,KH,KW] -> [KD*KH*KW, C]"""
    return wt_nchw.squeeze(1).reshape(C, KD*KH*KW).T.contiguous()


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="EvoKernel: 5-step depthwise conv3d benchmark")
    parser.add_argument("--iters", type=int, default=100, help="benchmark iterations")
    parser.add_argument("--steps", type=int, nargs="*", default=None,
                        help="which steps to run (default: all)")
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"Input:   [{B}, {C}, {D}, {H}, {W}] bf16")
    print(f"Weight:  [{C}, 1, {KD}, {KH}, {KW}] bf16")
    print(f"Output:  [{B}, {C}, {D_out}, {H_out}, {W_out}] bf16")
    print(f"Padding: {PAD}  Groups: {C}  GFLOPs: {2*B*C*D_out*H_out*W_out*KD*KH*KW/1e9:.2f}")

    inp, wt, bias = make_inputs(device)
    ref = pytorch_ref(inp, wt, bias)

    # Prepare NHWC tensors for step 2
    inp_nhwc = to_nhwc_3d(inp)
    wt_nhwc = wt_to_nhwc(wt)
    out_nhwc = to_nhwc_3d_out(None)

    all_steps = {
        1: ("Step1: Naive",          lambda o: run_step1(inp, wt, bias, o), False),
        2: ("Step2: NHWC",           lambda o: run_step2(inp_nhwc, wt_nhwc, bias, o), True),
        3: ("Step3: NCHW+LDS",       lambda o: run_step3(inp, wt, bias, o), False),
        4: ("Step4: MFMA",           lambda o: run_step4(inp, wt, bias, o), False),
        5: ("Step5: SGB",            lambda o: run_step5(inp, wt, bias, o), False),
    }

    steps_to_run = args.steps if args.steps else list(all_steps.keys())

    # ─── Warmup + Correctness ───
    print(f"\n{'='*72}")
    print("CORRECTNESS vs PyTorch F.conv3d")
    print(f"{'='*72}")

    active = {}
    for step in steps_to_run:
        name, runner, is_nhwc = all_steps[step]
        out = out_nhwc.clone() if is_nhwc else torch.empty_like(ref)
        try:
            for _ in range(3):
                runner(out)
            runner(out)
            if is_nhwc:
                out_nchw = nhwc_to_nchw(out, (D_out, H_out, W_out))
            else:
                out_nchw = out
            max_diff = (ref.float() - out_nchw.float()).abs().max().item()
            status = "PASS" if max_diff < 2.0 else "FAIL"
            print(f"  {name:20s}: max_diff={max_diff:.6e}  {status}")
            active[step] = (name, lambda o=out, r=runner: r(o), is_nhwc)
        except Exception as e:
            print(f"  {name:20s}: SKIP ({e})")

    # ─── Benchmark ───
    N = args.iters
    print(f"\n{'='*72}")
    print(f"BENCHMARK ({N} iterations)")
    print(f"{'='*72}")

    # PyTorch baseline
    pytorch_ms = benchmark_fn(lambda: pytorch_ref(inp, wt, bias), iters=N)

    results = {"PyTorch": pytorch_ms}
    for step in steps_to_run:
        if step not in active:
            continue
        name, runner, is_nhwc = active[step]
        out = out_nhwc.clone() if is_nhwc else torch.empty_like(ref)
        results[name] = benchmark_fn(lambda: runner(out), iters=N)

    pt_ms = results["PyTorch"]
    print(f"\n{'Kernel':<22} {'Time (ms)':>10} {'vs PyTorch':>11} {'Speedup':>8}")
    print("-" * 55)
    prev_ms = pt_ms
    for name, ms in results.items():
        vs_pt = pt_ms / ms
        print(f"{name:<22} {ms:>10.4f} {vs_pt:>10.1f}x")


if __name__ == "__main__":
    main()
