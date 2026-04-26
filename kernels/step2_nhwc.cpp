/*
 * Step 2: NHWC Layout Depthwise 3D Convolution (V84)
 *
 * Key idea: NHWC memory layout makes adjacent threads access adjacent channels,
 * giving coalesced global memory reads. Each thread handles 8 channels at one
 * spatial position. 512 threads per block = 8 spatial positions x 64 threads.
 *
 * Result: 1.16ms — 5x faster than naive (Step 1), but still reads every input
 * value from global memory (no data reuse via LDS).
 *
 * Input:  [D, H, W, C]  bf16 (NHWC layout, single batch)
 * Output: [D_out, H_out, W_out, C]  bf16
 * Weight: [KT*KH*KW, C]  bf16 (filter taps x channels)
 * Bias:   [C]  bf16
 *
 * Compile-time constants: KT, KH, KW, PAD_T/H/W, IN_T/H/W, OUT_T/H/W, CHANNELS
 */
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#ifndef KT
#define KT 3
#define KH 5
#define KW 5
#define PAD_T 0
#define PAD_H 2
#define PAD_W 2
#define IN_W 80
#define OUT_W 80
#define IN_H 45
#define OUT_H 45
#define IN_T 61
#define OUT_T 59
#define CHANNELS 512
#endif

#define V84_BLOCK_SIZE 512
#define V84_CHANNELS_PER_THREAD 8
#define V84_THREADS_PER_SPATIAL 64
#define V84_SPATIAL_PER_BLOCK 8

__global__ __launch_bounds__(V84_BLOCK_SIZE, 4)
void conv_depthwise3d_v84_nhwc_vec_bf16(
    const __hip_bfloat16* __restrict__ input,    // [D_in, H, W, C]
    __hip_bfloat16* __restrict__ output,         // [D_out, H_out, W_out, C]
    const __hip_bfloat16* __restrict__ kernel_weights,  // [KT*KH*KW, C]
    const __hip_bfloat16* __restrict__ bias,     // [C]
    const int total_spatial_outputs)             // D_out * H_out * W_out
{
    const int tid = threadIdx.x;
    const int spatial_group = tid / V84_THREADS_PER_SPATIAL;
    const int local_tid = tid % V84_THREADS_PER_SPATIAL;
    const int channel_base = local_tid * V84_CHANNELS_PER_THREAD;

    const int spatial_idx = blockIdx.x * V84_SPATIAL_PER_BLOCK + spatial_group;
    if (spatial_idx >= total_spatial_outputs) return;

    const int out_w = spatial_idx % OUT_W;
    const int tmp = spatial_idx / OUT_W;
    const int out_h = tmp % OUT_H;
    const int out_t = tmp / OUT_H;

    const int in_t = out_t;
    const int in_h_base = out_h - PAD_H;
    const int in_w_base = out_w - PAD_W;

    const int kh_start = (in_h_base < 0) ? -in_h_base : 0;
    const int kh_end = (in_h_base + KH > IN_H) ? IN_H - in_h_base : KH;
    const int kw_start = (in_w_base < 0) ? -in_w_base : 0;
    const int kw_end = (in_w_base + KW > IN_W) ? IN_W - in_w_base : KW;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    for (int kt = 0; kt < KT; ++kt) {
        const int t = in_t + kt;
        const int t_offset = t * (IN_H * IN_W * CHANNELS);

        for (int kh = kh_start; kh < kh_end; ++kh) {
            const int h = in_h_base + kh;
            const int h_offset = t_offset + h * (IN_W * CHANNELS);
            const int k_row_base = kt * KH * KW + kh * KW;

            for (int kw = kw_start; kw < kw_end; ++kw) {
                const int w = in_w_base + kw;
                const int input_base = h_offset + w * CHANNELS + channel_base;
                const int k_idx = k_row_base + kw;
                const int weight_base = k_idx * CHANNELS + channel_base;

                const float i0 = (float)input[input_base + 0];
                const float i1 = (float)input[input_base + 1];
                const float i2 = (float)input[input_base + 2];
                const float i3 = (float)input[input_base + 3];
                const float i4 = (float)input[input_base + 4];
                const float i5 = (float)input[input_base + 5];
                const float i6 = (float)input[input_base + 6];
                const float i7 = (float)input[input_base + 7];

                const float w0 = (float)kernel_weights[weight_base + 0];
                const float w1 = (float)kernel_weights[weight_base + 1];
                const float w2 = (float)kernel_weights[weight_base + 2];
                const float w3 = (float)kernel_weights[weight_base + 3];
                const float w4 = (float)kernel_weights[weight_base + 4];
                const float w5 = (float)kernel_weights[weight_base + 5];
                const float w6 = (float)kernel_weights[weight_base + 6];
                const float w7 = (float)kernel_weights[weight_base + 7];

                acc0 += i0 * w0; acc1 += i1 * w1;
                acc2 += i2 * w2; acc3 += i3 * w3;
                acc4 += i4 * w4; acc5 += i5 * w5;
                acc6 += i6 * w6; acc7 += i7 * w7;
            }
        }
    }

    if (bias) {
        acc0 += (float)bias[channel_base + 0]; acc1 += (float)bias[channel_base + 1];
        acc2 += (float)bias[channel_base + 2]; acc3 += (float)bias[channel_base + 3];
        acc4 += (float)bias[channel_base + 4]; acc5 += (float)bias[channel_base + 5];
        acc6 += (float)bias[channel_base + 6]; acc7 += (float)bias[channel_base + 7];
    }

    const int output_base = out_t * (OUT_H * OUT_W * CHANNELS) +
                            out_h * (OUT_W * CHANNELS) + out_w * CHANNELS + channel_base;

    output[output_base + 0] = (__hip_bfloat16)acc0; output[output_base + 1] = (__hip_bfloat16)acc1;
    output[output_base + 2] = (__hip_bfloat16)acc2; output[output_base + 3] = (__hip_bfloat16)acc3;
    output[output_base + 4] = (__hip_bfloat16)acc4; output[output_base + 5] = (__hip_bfloat16)acc5;
    output[output_base + 6] = (__hip_bfloat16)acc6; output[output_base + 7] = (__hip_bfloat16)acc7;
}
