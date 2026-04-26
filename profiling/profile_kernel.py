#!/usr/bin/env python3
"""
Generic PMC profiling script for any EvoKernel step.

Usage (wrap with rocprofv3):

  # Pass 1: Instruction mix
  rocprofv3 --pmc SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_INSTS_VMEM SQ_WAVES GRBM_COUNT \
    -- python profiling/profile_kernel.py --step 3

  # Pass 2: LDS bank conflicts
  rocprofv3 --pmc SQ_LDS_BANK_CONFLICT SQ_LDS_ADDR_CONFLICT SQ_INSTS_LDS_LOAD SQ_WAVES GRBM_COUNT \
    -- python profiling/profile_kernel.py --step 5

  # Thread trace (instruction-level):
  rocprofv3 --plugin att profiling/thread_trace.yaml \
    -- python profiling/profile_kernel.py --step 3
"""
import sys, os, math, argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from pyhip import module

B, C, D, H, W = 1, 512, 61, 45, 80
KD, KH, KW = 3, 5, 5
D_out, H_out, W_out = D - KD + 1, H, W
KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True, choices=[1,2,3,4,5])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda")
    dt = torch.bfloat16
    torch.manual_seed(42)
    inp = torch.randn(B, C, D, H, W, device=device, dtype=dt)
    wt = torch.randn(C, 1, KD, KH, KW, device=device, dtype=dt)
    bias = torch.randn(C, device=device, dtype=dt)
    out = torch.empty(B, C, D_out, H_out, W_out, device=device, dtype=dt)

    if args.step == 1:
        src = os.path.join(KERNEL_DIR, "step1_naive.cpp")
        m = module(src, "-O3")
        total = B * C * D_out * H_out * W_out
        def run():
            m.conv_depthwise3d_cuda_kernel_reference_bf16(
                [math.ceil(total/256)], [256],
                inp.data_ptr(), out.data_ptr(), wt.data_ptr(), bias.data_ptr(),
                B,C,C,D,H,W,D_out,H_out,W_out,KD,KH,KW,1,1,1,0,2,2,1,1,1)
    elif args.step == 3:
        src = os.path.join(KERNEL_DIR, "step3_nchw_lds.cpp")
        flags = "-O3 -DKD=3 -DKH=5 -DKW=5 -DPaddingD=0 -DPaddingH=2 -DPaddingW=2 -DBLOCK_H=45 -DBLOCK_W=80 -DIO_DTYPE=__hip_bfloat16"
        m = module(src, flags)
        def run():
            m.conv_depthwise3d_hip([B,C,D_out],[256],
                inp.data_ptr(),out.data_ptr(),wt.data_ptr(),bias.data_ptr(),
                C,D,H,W,C,D_out,H_out,W_out)
    elif args.step == 5:
        src = os.path.join(KERNEL_DIR, "step5_sgb.cpp")
        flags = "-O3 -DKD=3 -DKH=5 -DKW=5 -DPaddingD=0 -DPaddingH=2 -DPaddingW=2 -DBLOCK_H=45 -DBLOCK_W=80 -DIO_DTYPE=__hip_bfloat16"
        m = module(src, flags)
        def run():
            m.conv_depthwise3d_hip([B,C,D_out],[256],
                inp.data_ptr(),out.data_ptr(),wt.data_ptr(),bias.data_ptr(),
                C,D,H,W,C,D_out,H_out,W_out)
    else:
        print(f"Step {args.step} profiling not implemented (steps 1, 3, 5 supported)")
        return

    for _ in range(args.warmup):
        run(); torch.cuda.synchronize()

    for _ in range(args.iters):
        run(); torch.cuda.synchronize()

    print(f"Done — step {args.step}, {args.iters} profiled iterations")

if __name__ == "__main__":
    main()
