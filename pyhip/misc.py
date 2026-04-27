"""Numeric helpers (aligned with full pyhip `misc.calc_diff` for benchmarks)."""

import torch

# https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/testing/numeric.py#L5
def calc_diff(x: torch.Tensor, y: torch.Tensor, diff_thr=None):
    def get_diff(x, y):
        x, y = x.double(), y.double()
        denominator = (x * x + y * y).sum()
        if denominator == 0:
            return 0.0
        sim = 2 * (x * y).sum() / denominator
        diff = (1 - sim).item()
        return diff

    diff = get_diff(x, y)
    if diff != diff or (diff_thr is not None and diff > diff_thr):
        if diff_thr is None or diff_thr < 0:
            return diff
        print(x)
        print(y)
        print(x.shape)
        print(y.shape)
        if len(x.shape) == 2:
            print_count = 0
            M, N = x.shape
            for m in range(0, M, 16):
                dm = get_diff(x[m : m + 16, :], y[m : m + 16, :])
                if dm != dm:
                    print(x[m : m + 16, :])
                    print(y[m : m + 16, :])
                    assert 0
                elif dm >= diff_thr:
                    print_count += 1
                    assert print_count < 16, f"Too many errors in calc_diff with {diff_thr=:.3f}"
                    print(f"[{m:6}]: ", end="")
                    for n in range(0, N, 16):
                        d = get_diff(
                            x[m : m + 16, n : n + 16], y[m : m + 16, n : n + 16]
                        )
                        if d < diff_thr:
                            print(f"_.__ ", end="")
                        else:
                            print(f"{d:.2f} ", end="")
                    print()
            print()
        assert 0, f"{diff=} > {diff_thr=} !!!"
    assert diff == diff, "diff is nan!"
    return diff
