/*
 * Phase 12: MFMA-based Depthwise Conv3D
 *
 * Key innovation: Use MFMA_f32_4x4x4f16 (batch=16) to process 16 channels
 * simultaneously via Toeplitz matrix formulation of the width convolution.
 *
 * MFMA_4x4x4 with batch=16:
 *   C[4,4,16] += A[4,4,16] * B[4,4,16]
 *   - M=4: output width positions (Toeplitz row)
 *   - N=4: output height positions (via local_id)
 *   - K=4: input width positions (filter tap group)
 *   - Batch=16: independent channels
 *
 * Lane mapping (verified on gfx950):
 *   lane l: batch=l/4 (channel), local_id=l%4
 *   A[local_id, k=0..3]: Toeplitz weight row
 *   B[k=0..3, local_id]: input column at height=local_id
 *   C[m=0..3, local_id]: result column (all 4 widths for this height)
 *
 * Compile:
 *   hipcc -O3 --offload-arch=gfx950 depthwise_conv3d_mfma.cpp ...
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

using fp16x4_t = __attribute__((ext_vector_type(4))) _Float16;
using fp32x4_t = __attribute__((ext_vector_type(4))) float;

// Compile-time constants (set via -D flags)
// KD, KH, KW, PaddingD, PaddingH, PaddingW, BLOCK_H, BLOCK_W

#ifndef KD
#define KD 3
#endif
#ifndef KH
#define KH 5
#endif
#ifndef KW
#define KW 5
#endif
#ifndef PaddingH
#define PaddingH 2
#endif
#ifndef PaddingW
#define PaddingW 2
#endif
#ifndef BLOCK_H
#define BLOCK_H 45
#endif
#ifndef BLOCK_W
#define BLOCK_W 80
#endif

// Channels processed per block
constexpr int CHANNELS_PER_BLOCK = 16;

// Output tile: 4 widths x 4 heights
constexpr int TILE_W = 4;
constexpr int TILE_H = 4;

// Number of width tiles
constexpr int NUM_W_TILES = (BLOCK_W + TILE_W - 1) / TILE_W;  // 20

// Height tiles per block
constexpr int NUM_H_TILES = (BLOCK_H + TILE_H - 1) / TILE_H;  // 12

// Input tile size per depth slice in LDS
// Need heights: h_tile*4 - PaddingH .. h_tile*4 + 3 + (KH-1-PaddingH)
// = h_tile*4 - 2 .. h_tile*4 + 6 = 8 or 9 rows
// For the full height strip: need TILE_H + KH - 1 = 8 consecutive input rows
constexpr int INPUT_ROWS_PER_STRIP = TILE_H + KH - 1;  // 8
constexpr int PADDED_W = PaddingW + BLOCK_W + PaddingW;  // 84
constexpr int LDS_ROW_STRIDE = PADDED_W;  // in bf16 units

// LDS layout: [channel][height_row][width] in bf16 (one depth at a time)
// Total: 16 * 8 * 84 * 2 = 21,504 bytes
constexpr int LDS_SIZE_BF16 = CHANNELS_PER_BLOCK * INPUT_ROWS_PER_STRIP * PADDED_W;

// Process w_tiles in groups to balance register pressure vs LDS reloads
constexpr int WT_GROUP = 4;  // 4 w_tiles per group → 16 VGPRs for accumulators
constexpr int NUM_WT_GROUPS = (NUM_W_TILES + WT_GROUP - 1) / WT_GROUP;  // 5

// Weight count per channel
constexpr int WEIGHT_SIZE = KD * KH * KW;  // 75


__global__ void __launch_bounds__(64, 2)
depthwise_conv3d_mfma(
    const __hip_bfloat16* __restrict__ input,   // [B, C, D, H, W]
    __hip_bfloat16* __restrict__ output,        // [B, C, D_out, H_out, W_out]
    const __hip_bfloat16* __restrict__ weight,  // [C, 1, KD, KH, KW]
    const __hip_bfloat16* __restrict__ bias,    // [C]
    int C_total,  // total channels
    int D_in,     // input depth
    int H_in,     // input height = BLOCK_H
    int W_in,     // input width  = BLOCK_W
    int D_out,    // output depth
    int H_out,    // output height
    int W_out     // output width
)
{
    const int d_out = blockIdx.x;
    const int ch_group = blockIdx.y * CHANNELS_PER_BLOCK;
    const int h_tile_idx = blockIdx.z;  // which 4-height tile

    const int tid = threadIdx.x;  // 0..63
    const int batch_id = tid / 4;   // channel within group (0..15)
    const int local_id = tid % 4;   // spatial position within tile

    const int my_channel = ch_group + batch_id;
    if (my_channel >= C_total) return;

    const int h_tile_base = h_tile_idx * TILE_H;  // output height start

    // ===== Phase 1: Load Toeplitz weights into registers =====
    // For each (kd, kh), build 2 Toeplitz rows (for 2 MFMA K-segments)
    // Toeplitz row for local_id and K=0..3 (call 1) and K=4..7 (call 2)

    // Load all 75 weight values for this channel
    float w_all[KD][KH][KW];
    const __hip_bfloat16* my_weight = weight + my_channel * WEIGHT_SIZE;
    #pragma unroll
    for (int i = 0; i < WEIGHT_SIZE; i++) {
        ((float*)w_all)[i] = (float)my_weight[i];
    }

    // Build Toeplitz A operands: A_upper[kd][kh] and A_lower[kd][kh]
    // Each is fp16x4 (4 values for K=4 segment)
    // Toeplitz row for output position m=local_id:
    //   Full row: [g(m), g(m-1), ..., g(0), 0, ..., 0, g(KW-1), ..., g(m+1)]
    //   Split into upper (K=0..3) and lower (K=4..7)
    //
    // For KW=5, F(4,5): input positions 0..7, output positions 0..3
    // G^T[m, t] = g[t - m] if 0 <= t-m < KW, else 0
    // K segment 1: t=0..3 -> g[0-m]..g[3-m]
    // K segment 2: t=4..7 -> g[4-m]..g[7-m]

    fp16x4_t A_upper[KD][KH];
    fp16x4_t A_lower[KD][KH];

    #pragma unroll
    for (int kd = 0; kd < KD; kd++) {
        #pragma unroll
        for (int kh = 0; kh < KH; kh++) {
            // Upper segment: t=0..3, tap = t - local_id
            for (int t = 0; t < 4; t++) {
                int tap = t - local_id;
                if (tap >= 0 && tap < KW)
                    A_upper[kd][kh][t] = (_Float16)w_all[kd][kh][tap];
                else
                    A_upper[kd][kh][t] = (_Float16)0.0f;
            }
            // Lower segment: t=4..7, tap = t - local_id
            for (int t = 0; t < 4; t++) {
                int tap = (t + 4) - local_id;
                if (tap >= 0 && tap < KW)
                    A_lower[kd][kh][t] = (_Float16)w_all[kd][kh][tap];
                else
                    A_lower[kd][kh][t] = (_Float16)0.0f;
            }
        }
    }

    // Load bias
    float bias_val = 0.0f;
    if (bias != nullptr)
        bias_val = (float)bias[my_channel];

    // ===== Phase 2: LDS for input caching (one depth at a time) =====
    __shared__ __hip_bfloat16 s_input[LDS_SIZE_BF16];

    // ===== Phase 3: Compute =====
    // Process w_tiles in groups of WT_GROUP. For each group:
    //   - kd outer loop loads LDS per depth
    //   - kh inner loop does MFMA for all tiles in the group
    //   - Accumulators persist across kd for the group's tiles

    int h_out_pos = h_tile_base + local_id;
    __hip_bfloat16* out_base = nullptr;
    if (h_out_pos < H_out) {
        out_base = output +
            ((size_t)my_channel * D_out + d_out) * H_out * W_out +
            h_out_pos * W_out;
    }

    for (int wg = 0; wg < NUM_WT_GROUPS; wg++) {
        int wt_start = wg * WT_GROUP;
        int wt_end = wt_start + WT_GROUP;
        if (wt_end > NUM_W_TILES) wt_end = NUM_W_TILES;

        // Initialize accumulators for this group
        fp32x4_t acc[WT_GROUP];
        #pragma unroll
        for (int i = 0; i < WT_GROUP; i++) {
            acc[i] = {bias_val, bias_val, bias_val, bias_val};
        }

        for (int kd = 0; kd < KD; kd++) {
            const int d_in = d_out + kd;

            // Load input for this depth into LDS
            for (int i = tid; i < LDS_SIZE_BF16; i += 64) {
                s_input[i] = __float2bfloat16(0.0f);
            }
            __syncthreads();

            for (int ch_local = 0; ch_local < CHANNELS_PER_BLOCK; ch_local++) {
                int global_ch = ch_group + ch_local;
                if (global_ch >= C_total) continue;

                for (int row = 0; row < INPUT_ROWS_PER_STRIP; row++) {
                    int h_in = h_tile_base - PaddingH + row;
                    if (h_in < 0 || h_in >= H_in) continue;

                    const __hip_bfloat16* src_row = input +
                        ((size_t)global_ch * D_in + d_in) * H_in * W_in + h_in * W_in;
                    __hip_bfloat16* dst_row = s_input +
                        ch_local * INPUT_ROWS_PER_STRIP * PADDED_W +
                        row * PADDED_W + PaddingW;

                    for (int w = tid; w < W_in; w += 64) {
                        dst_row[w] = src_row[w];
                    }
                }
            }
            __syncthreads();

            // Compute MFMA for all kh and tiles in this group
            #pragma unroll
            for (int kh = 0; kh < KH; kh++) {
                int h_row = local_id + kh;
                __hip_bfloat16* lds_row = s_input +
                    batch_id * INPUT_ROWS_PER_STRIP * PADDED_W +
                    h_row * PADDED_W;

                #pragma unroll
                for (int ti = 0; ti < WT_GROUP; ti++) {
                    int wt = wt_start + ti;
                    if (wt >= NUM_W_TILES) break;
                    int w_base = wt * TILE_W;

                    __hip_bfloat16* lds_base = lds_row + w_base;

                    fp16x4_t B_upper, B_lower;
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        B_upper[k] = (_Float16)(float)lds_base[k];
                        B_lower[k] = (_Float16)(float)lds_base[k + 4];
                    }

                    acc[ti] = __builtin_amdgcn_mfma_f32_4x4x4f16(
                        A_upper[kd][kh], B_upper, acc[ti], 0, 0, 0);
                    acc[ti] = __builtin_amdgcn_mfma_f32_4x4x4f16(
                        A_lower[kd][kh], B_lower, acc[ti], 0, 0, 0);
                }
            }
            __syncthreads();  // Before next kd LDS reload
        }

        // Store results for this group
        if (out_base) {
            #pragma unroll
            for (int ti = 0; ti < WT_GROUP; ti++) {
                int wt = wt_start + ti;
                if (wt >= NUM_W_TILES) break;
                int w_base = wt * TILE_W;
                #pragma unroll
                for (int m = 0; m < TILE_W; m++) {
                    if (w_base + m < W_out) {
                        out_base[w_base + m] = __float2bfloat16(acc[ti][m]);
                    }
                }
            }
        }
    }
}
