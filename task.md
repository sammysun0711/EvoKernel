import torch
import torch.nn.functional as F

B, C_in, C_out, D, H, W = 1, 512, 512, 61, 45, 80
kernel_size = (3, 5, 5)
padding = (0, 2, 2)
groups = 512
stride = (1, 1, 1)
dilation = (1, 1, 1)
device = torch.device("cuda")
input_dtype = torch.bfloat16 
input_tensor = torch.randn(B, C_in, D, H, W).to(dtype=input_dtype).to(device)
weight_tensor = torch.randn(C_out, C_in // groups, *kernel_size).to(dtype=input_dtype).to(device)
bias_tensor = torch.randn(C_out).to(dtype=input_dtype).to(device)

# 计算输出尺寸
D_out = (D + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
H_out = (H + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
W_out = (W + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) // stride[2] + 1
print(f"输出尺寸: [{B}, {C_out}, {D_out}, {H_out}, {W_out}]")

# 计算量 (GFLOPs)
gflops = (2.0 * B * C_out * D_out * H_out * W_out * (C_in // groups) * kernel_size[0] * kernel_size[1] * kernel_size[2]) / 1e9
print(f"理论计算量: {gflops:.4f} GFLOPs")


F.conv3d(input_tensor, weight_tensor, 
                        bias=bias_tensor,
                        stride=stride, 
                        padding=padding, 
                        dilation=dilation,
                        groups=groups)
