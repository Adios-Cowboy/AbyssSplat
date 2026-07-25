#include "backward.cuh"
#include "helpers.cuh"
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

inline __device__ void warpSum3(float3& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
    val.z = cg::reduce(tile, val.z, cg::plus<float>());
}

inline __device__ void warpSum2(float2& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
}

inline __device__ void warpSum(float& val, cg::thread_block_tile<32>& tile){
    val = cg::reduce(tile, val, cg::plus<float>());
}
__global__ void nd_rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussians_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float* __restrict__ medium_rgb,
    const float* __restrict__ medium_bs,
    const float* __restrict__ medium_attn,
    const float* __restrict__ medium_omega,
    const float* __restrict__ depths,
    const float* __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const int* __restrict__ first_index,
    const float* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float* __restrict__ v_rgb,
    float* __restrict__ v_opacity,
    float* __restrict__ v_medium_rgb,
    float* __restrict__ v_medium_bs,
    float* __restrict__ v_medium_attn,
    float* __restrict__ v_medium_omega
) {
    auto block = cg::this_thread_block();
    const int tr = block.thread_rank();
    int32_t tile_id = blockIdx.y * tile_bounds.x + blockIdx.x;
    unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    float px = (float)j;
    float py = (float)i;
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);
    // which gaussians get gradients for this pixel
    const int2 range = tile_bins[tile_id];
    // df/d_out for this pixel
    const float *v_out = &(v_output[channels * pix_id]);
    const float v_out_alpha = v_output_alpha[pix_id];
    // this is the T AFTER the last gaussian in this pixel
    float T_final = final_Ts[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    
    extern __shared__ half workspace[];

    half *S = (half*)(&workspace[channels * tr]);
    #pragma unroll
    for(int c=0; c<channels; ++c){
        S[c] = __float2half(0.f);
    }
    const int bin_final = inside ? final_index[pix_id] : 0;
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    for (int idx = warp_bin_final - 1; idx >= range.x; --idx) {
        int valid = inside && idx < bin_final;
        const int32_t g = gaussians_ids_sorted[idx];
        const float3 conic = conics[g];
        const float2 center = xys[g];
        const float2 delta = {center.x - px, center.y - py};
        const float sigma =
            0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
            conic.y * delta.x * delta.y;
        valid &= (sigma >= 0.f);
        const float opac = opacities[g];
        const float vis = __expf(-sigma);
        const float alpha = min(0.99f, opac * vis);
        valid &= (alpha >= 1.f / 255.f);
        if(!warp.any(valid)){
            continue;
        }
        float v_alpha = 0.f;
        float3 v_conic_local = {0.f, 0.f, 0.f};
        float2 v_xy_local = {0.f, 0.f};
        float v_opacity_local = 0.f;
        if(valid){
            // compute the current T for this gaussian
            const float ra = 1.f / (1.f - alpha);
            T *= ra;
            // update v_rgb for this gaussian
            const float fac = alpha * T;
            for (int c = 0; c < channels; ++c) {
                // gradient wrt rgb
                atomicAdd(&(v_rgb[channels * g + c]), fac * v_out[c]);
                // contribution from this pixel
                v_alpha += (rgbs[channels * g + c] * T - __half2float(S[c]) * ra) * v_out[c];
                // contribution from background pixel
                v_alpha += -T_final * ra * background[c] * v_out[c];
                // update the running sum
                S[c] = __hadd(S[c], __float2half(rgbs[channels * g + c] * fac));
            }
            v_alpha += T_final * ra * v_out_alpha;
            const float v_sigma = -opac * vis * v_alpha;
            v_conic_local = {0.5f * v_sigma * delta.x * delta.x, 
                             v_sigma * delta.x * delta.y,
                             0.5f * v_sigma * delta.y * delta.y};
            v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y), 
                          v_sigma * (conic.y * delta.x + conic.z * delta.y)};
            v_opacity_local = vis * v_alpha;
        }
        warpSum3(v_conic_local, warp);
        warpSum2(v_xy_local, warp);
        warpSum(v_opacity_local, warp);
        if (warp.thread_rank() == 0) {
            float* v_conic_ptr = (float*)(v_conic);
            float* v_xy_ptr = (float*)(v_xy);
            atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
            atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
            atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
            atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
            atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);
            atomicAdd(v_opacity + g, v_opacity_local);
        }
    }
}

__global__ void rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    float2* __restrict__ xys_grad_abs,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float3* __restrict__ medium_rgb,
    const float3* __restrict__ medium_bs,
    const float3* __restrict__ medium_attn,
    const float3* __restrict__ medium_omega,
    const float* __restrict__ depths,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const int* __restrict__ first_index,
    const float3* __restrict__ v_output,
    const float3* __restrict__ v_out_medium,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity,
    float3* __restrict__ v_medium_rgb,
    float3* __restrict__ v_medium_bs,
    float3* __restrict__ v_medium_attn,
    float3* __restrict__ v_medium_omega
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    const bool inside = (i < img_size.y && j < img_size.x);

    float T_final = final_Ts[pix_id];
    float T = T_final;
    float3 buffer = {0.f, 0.f, 0.f};
    float3 buffer_medium = {0.f, 0.f, 0.f};
    const int bin_final = inside ? final_index[pix_id] : 0;

    const int2 range = tile_bins[tile_id];
    const int block_size = block.size();
    const int num_batches = (range.y - range.x + block_size - 1) / block_size;

    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[MAX_BLOCK_SIZE];
    __shared__ float3 conic_batch[MAX_BLOCK_SIZE];
    __shared__ float3 rgbs_batch[MAX_BLOCK_SIZE];
    __shared__ float depth_batch[MAX_BLOCK_SIZE];

    const float3 v_out     = v_output[pix_id];
    const float3 v_out_med = v_out_medium[pix_id];
    const float  v_out_alpha = v_output_alpha[pix_id];

    const float3 medium_rgb_pix   = medium_rgb[pix_id];
    const float3 medium_bs_pix    = medium_bs[pix_id];
    const float3 medium_attn_pix  = medium_attn[pix_id];
    const float3 medium_omega_pix = medium_omega[pix_id];

    float min_medium_attn_pix = std::min(medium_attn_pix.x,
                                std::min(medium_attn_pix.y, medium_attn_pix.z));
    min_medium_attn_pix = std::min(0.f, min_medium_attn_pix);

    float  latter_depth  = 0.f;
    float3 latter_exp_bs = {0.f, 0.f, 0.f};
    bool has_latter = false;

    // 光线级局部寄存器，提升至循环外部以保证全视线积分连续性
    float3 v_medium_rgb_ray   = {0.f, 0.f, 0.f};
    float3 v_medium_bs_ray    = {0.f, 0.f, 0.f};
    float3 v_medium_attn_ray  = {0.f, 0.f, 0.f};
    float3 v_medium_omega_ray = {0.f, 0.f, 0.f};

    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());

    for (int b = 0; b < num_batches; ++b) {
        block.sync();

        const int batch_end  = range.y - 1 - block_size * b;
        int       batch_size = min(block_size, batch_end + 1 - range.x);
        const int idx        = batch_end - tr;
        if (idx >= range.x) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr]          = g_id;
            const float2 xy       = xys[g_id];
            const float  opac     = opacities[g_id];
            xy_opacity_batch[tr]  = {xy.x, xy.y, opac};
            conic_batch[tr]       = conics[g_id];
            rgbs_batch[tr]        = rgbs[g_id];
            depth_batch[tr]       = depths[g_id];
        }
        block.sync();

        for (int t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            int valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            float2 delta;
            float3 conic;
            float  vis;
            float  depth;
            if (valid) {
                depth        = depth_batch[t];
                conic        = conic_batch[t];
                float3 xy_opac = xy_opacity_batch[t];
                opac         = xy_opac.z;
                delta        = {xy_opac.x - px, xy_opac.y - py};
                float sigma  = 0.5f * (conic.x * delta.x * delta.x +
                                       conic.z * delta.y * delta.y) +
                               conic.y * delta.x * delta.y;
                vis   = __expf(-sigma);
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha * __expf(-min_medium_attn_pix * depth) < 1.f / 255.f) {
                    valid = 0;
                }
            }
            if (!warp.any(valid)) {
                continue;
            }

            float3 v_rgb_local             = {0.f, 0.f, 0.f};
            float3 v_conic_local           = {0.f, 0.f, 0.f};
            float2 v_xy_local              = {0.f, 0.f};
            float2 v_xy_abs_local          = {0.f, 0.f};
            float  v_z_abs_local           = 0.f;
            float  v_opacity_local         = 0.f;

            if (valid) {
                float3 exp_attn;
                exp_attn.x = __expf(-medium_attn_pix.x * depth);
                exp_attn.y = __expf(-medium_attn_pix.y * depth);
                exp_attn.z = __expf(-medium_attn_pix.z * depth);

                float3 exp_bs;
                exp_bs.x = __expf(-medium_bs_pix.x * depth);
                exp_bs.y = __expf(-medium_bs_pix.y * depth);
                exp_bs.z = __expf(-medium_bs_pix.z * depth);

                float3 passive_seg = {
                    medium_rgb_pix.x * (exp_bs.x - latter_exp_bs.x),
                    medium_rgb_pix.y * (exp_bs.y - latter_exp_bs.y),
                    medium_rgb_pix.z * (exp_bs.z - latter_exp_bs.z)
                };
//xiugai
                float3 active_seg = {0.f, 0.f, 0.f};
                if (has_latter) {
                    float z_mid_seg      = 0.5f * (depth + latter_depth);
                    float delta_z_seg    = latter_depth - depth;
                    float geom_atten_seg = 1.0f / (z_mid_seg * z_mid_seg + 0.1);
                    float3 exp_active_seg;
                    exp_active_seg.x = __expf(-2.0f * medium_attn_pix.x * z_mid_seg);
                    exp_active_seg.y = __expf(-2.0f * medium_attn_pix.y * z_mid_seg);
                    exp_active_seg.z = __expf(-2.0f * medium_attn_pix.z * z_mid_seg);

                    active_seg.x = medium_omega_pix.x * geom_atten_seg * exp_active_seg.x * delta_z_seg;
                    active_seg.y = medium_omega_pix.y * geom_atten_seg * exp_active_seg.y * delta_z_seg;
                    active_seg.z = medium_omega_pix.z * geom_atten_seg * exp_active_seg.z * delta_z_seg;

                    // 剥离 warpSum，光线域内标量累加
                    v_medium_omega_ray.x += v_out_med.x * T * geom_atten_seg * exp_active_seg.x * delta_z_seg;
                    v_medium_omega_ray.y += v_out_med.y * T * geom_atten_seg * exp_active_seg.y * delta_z_seg;
                    v_medium_omega_ray.z += v_out_med.z * T * geom_atten_seg * exp_active_seg.z * delta_z_seg;

                    v_medium_attn_ray.x += v_out_med.x * T * medium_omega_pix.x * geom_atten_seg * (-2.0f * z_mid_seg) * exp_active_seg.x * delta_z_seg;
                    v_medium_attn_ray.y += v_out_med.y * T * medium_omega_pix.y * geom_atten_seg * (-2.0f * z_mid_seg) * exp_active_seg.y * delta_z_seg;
                    v_medium_attn_ray.z += v_out_med.z * T * medium_omega_pix.z * geom_atten_seg * (-2.0f * z_mid_seg) * exp_active_seg.z * delta_z_seg;
                }

                v_medium_rgb_ray.x += v_out_med.x * T * (exp_bs.x - latter_exp_bs.x);
                v_medium_rgb_ray.y += v_out_med.y * T * (exp_bs.y - latter_exp_bs.y);
                v_medium_rgb_ray.z += v_out_med.z * T * (exp_bs.z - latter_exp_bs.z);

                float3 v_exp_bs_local = {
                    v_out_med.x * T * medium_rgb_pix.x,
                    v_out_med.y * T * medium_rgb_pix.y,
                    v_out_med.z * T * medium_rgb_pix.z
                };
                v_medium_bs_ray.x += v_exp_bs_local.x * (-depth * exp_bs.x + latter_depth * latter_exp_bs.x);
                v_medium_bs_ray.y += v_exp_bs_local.y * (-depth * exp_bs.y + latter_depth * latter_exp_bs.y);
                v_medium_bs_ray.z += v_exp_bs_local.z * (-depth * exp_bs.z + latter_depth * latter_exp_bs.z);

                if (!has_latter) {
                    buffer_medium.x = T * passive_seg.x;
                    buffer_medium.y = T * passive_seg.y;
                    buffer_medium.z = T * passive_seg.z;
                } else {
                    buffer_medium.x += T * (passive_seg.x + active_seg.x);
                    buffer_medium.y += T * (passive_seg.y + active_seg.y);
                    buffer_medium.z += T * (passive_seg.z + active_seg.z);
                }

                float ra = 1.f / (1.f - alpha);
                T *= ra;

                v_z_abs_local += fabsf(v_exp_bs_local.x * medium_bs_pix.x * exp_bs.x * (ra + 1.f));
                v_z_abs_local += fabsf(v_exp_bs_local.y * medium_bs_pix.y * exp_bs.y * (ra + 1.f));
                v_z_abs_local += fabsf(v_exp_bs_local.z * medium_bs_pix.z * exp_bs.z * (ra + 1.f));

                const float3 rgb = rgbs_batch[t];
                const float  fac = alpha * T;
                float v_alpha    = 0.f;
                float3 exp_attn_fac   = {fac * exp_attn.x, fac * exp_attn.y, fac * exp_attn.z};
                v_rgb_local = {v_out.x * exp_attn_fac.x,
                               v_out.y * exp_attn_fac.y,
                               v_out.z * exp_attn_fac.z};
                float3 v_exp_attn_local = {v_rgb_local.x * rgb.x,
                                           v_rgb_local.y * rgb.y,
                                           v_rgb_local.z * rgb.z};

                v_medium_attn_ray.x += -v_exp_attn_local.x * depth;
                v_medium_attn_ray.y += -v_exp_attn_local.y * depth;
                v_medium_attn_ray.z += -v_exp_attn_local.z * depth;

                v_z_abs_local += fabsf(v_exp_attn_local.x * medium_attn_pix.x);
                v_z_abs_local += fabsf(v_exp_attn_local.y * medium_attn_pix.y);
                v_z_abs_local += fabsf(v_exp_attn_local.z * medium_attn_pix.z);

                v_alpha += (rgb.x * T * exp_attn.x - buffer.x * ra) * v_out.x - buffer_medium.x * ra * v_out_med.x;
                v_alpha += (rgb.y * T * exp_attn.y - buffer.y * ra) * v_out.y - buffer_medium.y * ra * v_out_med.y;
                v_alpha += (rgb.z * T * exp_attn.z - buffer.z * ra) * v_out.z - buffer_medium.z * ra * v_out_med.z;
                v_alpha += T_final * ra * v_out_alpha;

                buffer.x += rgb.x * exp_attn_fac.x;
                buffer.y += rgb.y * exp_attn_fac.y;
                buffer.z += rgb.z * exp_attn_fac.z;

                latter_depth  = depth;
                latter_exp_bs = exp_bs;
                has_latter    = true;

                const float v_sigma = -opac * vis * v_alpha;
                v_conic_local = {0.5f * v_sigma * delta.x * delta.x,
                                 v_sigma * delta.x * delta.y,
                                 0.5f * v_sigma * delta.y * delta.y};
                v_xy_local    = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
                                 v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                v_xy_abs_local = {fabsf(v_xy_local.x), fabsf(v_xy_local.y)};
                v_opacity_local = vis * v_alpha;
            }

            // 高斯基元空间参数仍维持 warpSum
            warpSum3(v_rgb_local, warp);
            warpSum3(v_conic_local, warp);
            warpSum2(v_xy_local, warp);
            warpSum2(v_xy_abs_local, warp);
            warpSum(v_z_abs_local, warp);
            warpSum(v_opacity_local, warp);

            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];

                float* v_rgb_ptr = (float*)(v_rgb);
                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);

                float* v_conic_ptr = (float*)(v_conic);
                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);

                float* v_xy_ptr = (float*)(v_xy);
                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);

                float* v_xy_abs_ptr = (float*)(xys_grad_abs);
                atomicAdd(v_xy_abs_ptr + 2*g + 0, v_xy_abs_local.x);
                atomicAdd(v_xy_abs_ptr + 2*g + 1, v_xy_abs_local.y);

                atomicAdd(v_opacity + g, v_opacity_local);
            }
        }
    }

    block.sync();

    // 尾部区域介质特征计算
    if (inside) {
        float3 exp_bs = {1.f, 1.f, 1.f};
        T = 1.f;

        v_medium_rgb_ray.x += v_out_med.x * T * (exp_bs.x - latter_exp_bs.x);
        v_medium_rgb_ray.y += v_out_med.y * T * (exp_bs.y - latter_exp_bs.y);
        v_medium_rgb_ray.z += v_out_med.z * T * (exp_bs.z - latter_exp_bs.z);

        float3 v_exp_bs_local = {
            v_out_med.x * T * medium_rgb_pix.x,
            v_out_med.y * T * medium_rgb_pix.y,
            v_out_med.z * T * medium_rgb_pix.z
        };
        v_medium_bs_ray.x += v_exp_bs_local.x * latter_depth * latter_exp_bs.x;
        v_medium_bs_ray.y += v_exp_bs_local.y * latter_depth * latter_exp_bs.y;
        v_medium_bs_ray.z += v_exp_bs_local.z * latter_depth * latter_exp_bs.z;

//xiugai

        if (has_latter) {
            float z_mid_tail      = 0.5f * latter_depth;
            float delta_z_tail    = latter_depth;
            float geom_atten_tail = 1.0f / (z_mid_tail * z_mid_tail + 0.1);
            float3 exp_active_tail;
            exp_active_tail.x = __expf(-2.0f * medium_attn_pix.x * z_mid_tail);
            exp_active_tail.y = __expf(-2.0f * medium_attn_pix.y * z_mid_tail);
            exp_active_tail.z = __expf(-2.0f * medium_attn_pix.z * z_mid_tail);

            v_medium_omega_ray.x += v_out_med.x * geom_atten_tail * exp_active_tail.x * delta_z_tail;
            v_medium_omega_ray.y += v_out_med.y * geom_atten_tail * exp_active_tail.y * delta_z_tail;
            v_medium_omega_ray.z += v_out_med.z * geom_atten_tail * exp_active_tail.z * delta_z_tail;

            v_medium_attn_ray.x += v_out_med.x * medium_omega_pix.x * geom_atten_tail * (-2.0f * z_mid_tail) * exp_active_tail.x * delta_z_tail;
            v_medium_attn_ray.y += v_out_med.y * medium_omega_pix.y * geom_atten_tail * (-2.0f * z_mid_tail) * exp_active_tail.y * delta_z_tail;
            v_medium_attn_ray.z += v_out_med.z * medium_omega_pix.z * geom_atten_tail * (-2.0f * z_mid_tail) * exp_active_tail.z * delta_z_tail;
        }

        // 光线积分结束后，单次解耦写回全局显存
        float* v_medium_rgb_ptr = (float*)(v_medium_rgb);
        atomicAdd(v_medium_rgb_ptr + 3*pix_id + 0, v_medium_rgb_ray.x);
        atomicAdd(v_medium_rgb_ptr + 3*pix_id + 1, v_medium_rgb_ray.y);
        atomicAdd(v_medium_rgb_ptr + 3*pix_id + 2, v_medium_rgb_ray.z);

        float* v_medium_bs_ptr = (float*)(v_medium_bs);
        atomicAdd(v_medium_bs_ptr + 3*pix_id + 0, v_medium_bs_ray.x);
        atomicAdd(v_medium_bs_ptr + 3*pix_id + 1, v_medium_bs_ray.y);
        atomicAdd(v_medium_bs_ptr + 3*pix_id + 2, v_medium_bs_ray.z);

        float* v_medium_attn_ptr = (float*)(v_medium_attn);
        atomicAdd(v_medium_attn_ptr + 3*pix_id + 0, v_medium_attn_ray.x);
        atomicAdd(v_medium_attn_ptr + 3*pix_id + 1, v_medium_attn_ray.y);
        atomicAdd(v_medium_attn_ptr + 3*pix_id + 2, v_medium_attn_ray.z);

        float* v_medium_omega_ptr = (float*)(v_medium_omega);
        atomicAdd(v_medium_omega_ptr + 3*pix_id + 0, v_medium_omega_ray.x);
        atomicAdd(v_medium_omega_ptr + 3*pix_id + 1, v_medium_omega_ray.y);
        atomicAdd(v_medium_omega_ptr + 3*pix_id + 2, v_medium_omega_ray.z);
    }
}

// ----version2-----
//__global__ void rasterize_backward_kernel(
//    const dim3 tile_bounds,
//    const dim3 img_size,
//    const int32_t* __restrict__ gaussian_ids_sorted,
//    const int2* __restrict__ tile_bins,
//    const float2* __restrict__ xys,
//    float2* __restrict__ xys_grad_abs,
//    const float3* __restrict__ conics,
//    const float3* __restrict__ rgbs,
//    const float* __restrict__ opacities,
//    const float3* __restrict__ medium_rgb,
//    const float3* __restrict__ medium_bs,
//    const float3* __restrict__ medium_attn,
//    const float3* __restrict__ medium_omega,
//    const float* __restrict__ depths,
//    const float3& __restrict__ background,
//    const float* __restrict__ final_Ts,
//    const int* __restrict__ final_index,
//    const int* __restrict__ first_index,
//    const float3* __restrict__ v_output,
//    const float3* __restrict__ v_out_medium,
//    const float* __restrict__ v_output_alpha,
//    float2* __restrict__ v_xy,
//    float3* __restrict__ v_conic,
//    float3* __restrict__ v_rgb,
//    float* __restrict__ v_opacity,
//    float3* __restrict__ v_medium_rgb,
//    float3* __restrict__ v_medium_bs,
//    float3* __restrict__ v_medium_attn,
//    float3* __restrict__ v_medium_omega
//) {
//    auto block = cg::this_thread_block();
//    int32_t tile_id =
//        block.group_index().y * tile_bounds.x + block.group_index().x;
//    unsigned i =
//        block.group_index().y * block.group_dim().y + block.thread_index().y;
//    unsigned j =
//        block.group_index().x * block.group_dim().x + block.thread_index().x;
//
//    const float px = (float)j;
//    const float py = (float)i;
//    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);
//
//    const bool inside = (i < img_size.y && j < img_size.x);
//
//    float T_final = final_Ts[pix_id];
//    float T = T_final;
//    float3 buffer = {0.f, 0.f, 0.f};
//    float3 buffer_medium = {0.f, 0.f, 0.f};
//    const int bin_final = inside ? final_index[pix_id] : 0;
//
//    const int2 range = tile_bins[tile_id];
//    const int block_size = block.size();
//    const int num_batches = (range.y - range.x + block_size - 1) / block_size;
//
//    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
//    __shared__ float3 xy_opacity_batch[MAX_BLOCK_SIZE];
//    __shared__ float3 conic_batch[MAX_BLOCK_SIZE];
//    __shared__ float3 rgbs_batch[MAX_BLOCK_SIZE];
//    __shared__ float depth_batch[MAX_BLOCK_SIZE];
//
//    const float3 v_out     = v_output[pix_id];
//    const float3 v_out_med = v_out_medium[pix_id];
//    const float  v_out_alpha = v_output_alpha[pix_id];
//
//    const float3 medium_rgb_pix   = medium_rgb[pix_id];
//    const float3 medium_bs_pix    = medium_bs[pix_id];
//    const float3 medium_attn_pix  = medium_attn[pix_id];
//    const float3 medium_omega_pix = medium_omega[pix_id];
//
//    float min_medium_attn_pix = std::min(medium_attn_pix.x,
//                                std::min(medium_attn_pix.y, medium_attn_pix.z));
//    min_medium_attn_pix = std::min(0.f, min_medium_attn_pix);
//
//    float  latter_depth  = 0.f;
//    float3 latter_exp_bs = {0.f, 0.f, 0.f};
//    bool has_latter = false;
//
//    const int tr = block.thread_rank();
//    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
//    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
//
//    for (int b = 0; b < num_batches; ++b) {
//        block.sync();
//
//        const int batch_end  = range.y - 1 - block_size * b;
//        int       batch_size = min(block_size, batch_end + 1 - range.x);
//        const int idx        = batch_end - tr;
//        if (idx >= range.x) {
//            int32_t g_id = gaussian_ids_sorted[idx];
//            id_batch[tr]          = g_id;
//            const float2 xy       = xys[g_id];
//            const float  opac     = opacities[g_id];
//            xy_opacity_batch[tr]  = {xy.x, xy.y, opac};
//            conic_batch[tr]       = conics[g_id];
//            rgbs_batch[tr]        = rgbs[g_id];
//            depth_batch[tr]       = depths[g_id];
//        }
//        block.sync();
//
//        for (int t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
//            int valid = inside;
//            if (batch_end - t > bin_final) {
//                valid = 0;
//            }
//            float alpha;
//            float opac;
//            float2 delta;
//            float3 conic;
//            float  vis;
//            float  depth;
//            if (valid) {
//                depth        = depth_batch[t];
//                conic        = conic_batch[t];
//                float3 xy_opac = xy_opacity_batch[t];
//                opac         = xy_opac.z;
//                delta        = {xy_opac.x - px, xy_opac.y - py};
//                float sigma  = 0.5f * (conic.x * delta.x * delta.x +
//                                       conic.z * delta.y * delta.y) +
//                               conic.y * delta.x * delta.y;
//                vis   = __expf(-sigma);
//                alpha = min(0.99f, opac * vis);
//                if (sigma < 0.f || alpha * __expf(-min_medium_attn_pix * depth) < 1.f / 255.f) {
//                    valid = 0;
//                }
//            }
//            if (!warp.any(valid)) {
//                continue;
//            }
//
//            float3 v_rgb_local             = {0.f, 0.f, 0.f};
//            float3 v_conic_local           = {0.f, 0.f, 0.f};
//            float2 v_xy_local              = {0.f, 0.f};
//            float2 v_xy_abs_local          = {0.f, 0.f};
//            float  v_z_abs_local           = 0.f;
//            float  v_opacity_local         = 0.f;
//            float3 v_medium_rgb_pix_local  = {0.f, 0.f, 0.f};
//            float3 v_medium_bs_pix_local   = {0.f, 0.f, 0.f};
//            float3 v_medium_attn_pix_local = {0.f, 0.f, 0.f};
//            float3 v_medium_omega_pix_local= {0.f, 0.f, 0.f};
//
//            if (valid) {
//                float3 exp_attn;
//                exp_attn.x = __expf(-medium_attn_pix.x * depth);
//                exp_attn.y = __expf(-medium_attn_pix.y * depth);
//                exp_attn.z = __expf(-medium_attn_pix.z * depth);
//
//                float3 exp_bs;
//                exp_bs.x = __expf(-medium_bs_pix.x * depth);
//                exp_bs.y = __expf(-medium_bs_pix.y * depth);
//                exp_bs.z = __expf(-medium_bs_pix.z * depth);
//
//                float3 passive_seg = {
//                    medium_rgb_pix.x * (exp_bs.x - latter_exp_bs.x),
//                    medium_rgb_pix.y * (exp_bs.y - latter_exp_bs.y),
//                    medium_rgb_pix.z * (exp_bs.z - latter_exp_bs.z)
//                };
//
//                float3 active_seg = {0.f, 0.f, 0.f};
//                if (has_latter) {
//                    float z_mid_seg      = 0.5f * (depth + latter_depth);
//                    float delta_z_seg    = latter_depth - depth;
//                    float geom_atten_seg = 1.0f / (z_mid_seg * z_mid_seg + 1e-6f);
//                    float3 exp_active_seg;
//                    exp_active_seg.x = __expf(-2.0f * medium_attn_pix.x * z_mid_seg);
//                    exp_active_seg.y = __expf(-2.0f * medium_attn_pix.y * z_mid_seg);
//                    exp_active_seg.z = __expf(-2.0f * medium_attn_pix.z * z_mid_seg);
//
//                    active_seg.x = medium_omega_pix.x * geom_atten_seg * exp_active_seg.x * delta_z_seg;
//                    active_seg.y = medium_omega_pix.y * geom_atten_seg * exp_active_seg.y * delta_z_seg;
//                    active_seg.z = medium_omega_pix.z * geom_atten_seg * exp_active_seg.z * delta_z_seg;
//
//                    v_medium_omega_pix_local.x += v_out_med.x * T * geom_atten_seg * exp_active_seg.x * delta_z_seg;
//                    v_medium_omega_pix_local.y += v_out_med.y * T * geom_atten_seg * exp_active_seg.y * delta_z_seg;
//                    v_medium_omega_pix_local.z += v_out_med.z * T * geom_atten_seg * exp_active_seg.z * delta_z_seg;
//
//                    v_medium_attn_pix_local.x += v_out_med.x * T * medium_omega_pix.x * geom_atten_seg * (-2.0f * z_mid_seg) * exp_active_seg.x * delta_z_seg;
//                    v_medium_attn_pix_local.y += v_out_med.y * T * medium_omega_pix.y * geom_atten_seg * (-2.0f * z_mid_seg) * exp_active_seg.y * delta_z_seg;
//                    v_medium_attn_pix_local.z += v_out_med.z * T * medium_omega_pix.z * geom_atten_seg * (-2.0f * z_mid_seg) * exp_active_seg.z * delta_z_seg;
//                }
//
//                v_medium_rgb_pix_local.x += v_out_med.x * T * (exp_bs.x - latter_exp_bs.x);
//                v_medium_rgb_pix_local.y += v_out_med.y * T * (exp_bs.y - latter_exp_bs.y);
//                v_medium_rgb_pix_local.z += v_out_med.z * T * (exp_bs.z - latter_exp_bs.z);
//
//                float3 v_exp_bs_local = {
//                    v_out_med.x * T * medium_rgb_pix.x,
//                    v_out_med.y * T * medium_rgb_pix.y,
//                    v_out_med.z * T * medium_rgb_pix.z
//                };
//                v_medium_bs_pix_local.x += v_exp_bs_local.x * (-depth * exp_bs.x + latter_depth * latter_exp_bs.x);
//                v_medium_bs_pix_local.y += v_exp_bs_local.y * (-depth * exp_bs.y + latter_depth * latter_exp_bs.y);
//                v_medium_bs_pix_local.z += v_exp_bs_local.z * (-depth * exp_bs.z + latter_depth * latter_exp_bs.z);
//
//                if (!has_latter) {
//                    buffer_medium.x = T * passive_seg.x;
//                    buffer_medium.y = T * passive_seg.y;
//                    buffer_medium.z = T * passive_seg.z;
//                } else {
//                    buffer_medium.x += T * (passive_seg.x + active_seg.x);
//                    buffer_medium.y += T * (passive_seg.y + active_seg.y);
//                    buffer_medium.z += T * (passive_seg.z + active_seg.z);
//                }
//
//                float ra = 1.f / (1.f - alpha);
//                T *= ra;
//
//                v_z_abs_local += fabsf(v_exp_bs_local.x * medium_bs_pix.x * exp_bs.x * (ra + 1.f));
//                v_z_abs_local += fabsf(v_exp_bs_local.y * medium_bs_pix.y * exp_bs.y * (ra + 1.f));
//                v_z_abs_local += fabsf(v_exp_bs_local.z * medium_bs_pix.z * exp_bs.z * (ra + 1.f));
//
//                const float3 rgb = rgbs_batch[t];
//                const float  fac = alpha * T;
//                float v_alpha    = 0.f;
//                float3 exp_attn_fac   = {fac * exp_attn.x, fac * exp_attn.y, fac * exp_attn.z};
//                v_rgb_local = {v_out.x * exp_attn_fac.x,
//                               v_out.y * exp_attn_fac.y,
//                               v_out.z * exp_attn_fac.z};
//                float3 v_exp_attn_local = {v_rgb_local.x * rgb.x,
//                                           v_rgb_local.y * rgb.y,
//                                           v_rgb_local.z * rgb.z};
//
//                v_medium_attn_pix_local.x += -v_exp_attn_local.x * depth;
//                v_medium_attn_pix_local.y += -v_exp_attn_local.y * depth;
//                v_medium_attn_pix_local.z += -v_exp_attn_local.z * depth;
//
//                v_z_abs_local += fabsf(v_exp_attn_local.x * medium_attn_pix.x);
//                v_z_abs_local += fabsf(v_exp_attn_local.y * medium_attn_pix.y);
//                v_z_abs_local += fabsf(v_exp_attn_local.z * medium_attn_pix.z);
//
//                v_alpha += (rgb.x * T * exp_attn.x - buffer.x * ra) * v_out.x - buffer_medium.x * ra * v_out_med.x;
//                v_alpha += (rgb.y * T * exp_attn.y - buffer.y * ra) * v_out.y - buffer_medium.y * ra * v_out_med.y;
//                v_alpha += (rgb.z * T * exp_attn.z - buffer.z * ra) * v_out.z - buffer_medium.z * ra * v_out_med.z;
//                v_alpha += T_final * ra * v_out_alpha;
//
//                buffer.x += rgb.x * exp_attn_fac.x;
//                buffer.y += rgb.y * exp_attn_fac.y;
//                buffer.z += rgb.z * exp_attn_fac.z;
//
//                latter_depth  = depth;
//                latter_exp_bs = exp_bs;
//                has_latter    = true;
//
//                const float v_sigma = -opac * vis * v_alpha;
//                v_conic_local = {0.5f * v_sigma * delta.x * delta.x,
//                                 v_sigma * delta.x * delta.y,
//                                 0.5f * v_sigma * delta.y * delta.y};
//                v_xy_local    = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
//                                 v_sigma * (conic.y * delta.x + conic.z * delta.y)};
//                v_xy_abs_local = {fabsf(v_xy_local.x), fabsf(v_xy_local.y)};
//                v_opacity_local = vis * v_alpha;
//            }
//
//            warpSum3(v_rgb_local, warp);
//            warpSum3(v_conic_local, warp);
//            warpSum2(v_xy_local, warp);
//            warpSum2(v_xy_abs_local, warp);
//            warpSum(v_z_abs_local, warp);
//            warpSum(v_opacity_local, warp);
//            warpSum3(v_medium_rgb_pix_local, warp);
//            warpSum3(v_medium_bs_pix_local, warp);
//            warpSum3(v_medium_attn_pix_local, warp);
//            warpSum3(v_medium_omega_pix_local, warp);
//
//            if (warp.thread_rank() == 0) {
//                int32_t g = id_batch[t];
//
//                float* v_rgb_ptr = (float*)(v_rgb);
//                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
//                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
//                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);
//
//                float* v_conic_ptr = (float*)(v_conic);
//                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
//                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
//                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
//
//                float* v_xy_ptr = (float*)(v_xy);
//                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
//                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);
//
//                float* v_xy_abs_ptr = (float*)(xys_grad_abs);
//                atomicAdd(v_xy_abs_ptr + 2*g + 0, v_xy_abs_local.x);
//                atomicAdd(v_xy_abs_ptr + 2*g + 1, v_xy_abs_local.y);
//
//                atomicAdd(v_opacity + g, v_opacity_local);
//
//                float* v_medium_attn_ptr = (float*)(v_medium_attn);
//                atomicAdd(v_medium_attn_ptr + 3*pix_id + 0, v_medium_attn_pix_local.x);
//                atomicAdd(v_medium_attn_ptr + 3*pix_id + 1, v_medium_attn_pix_local.y);
//                atomicAdd(v_medium_attn_ptr + 3*pix_id + 2, v_medium_attn_pix_local.z);
//
//                float* v_medium_rgb_ptr = (float*)(v_medium_rgb);
//                atomicAdd(v_medium_rgb_ptr + 3*pix_id + 0, v_medium_rgb_pix_local.x);
//                atomicAdd(v_medium_rgb_ptr + 3*pix_id + 1, v_medium_rgb_pix_local.y);
//                atomicAdd(v_medium_rgb_ptr + 3*pix_id + 2, v_medium_rgb_pix_local.z);
//
//                float* v_medium_bs_ptr = (float*)(v_medium_bs);
//                atomicAdd(v_medium_bs_ptr + 3*pix_id + 0, v_medium_bs_pix_local.x);
//                atomicAdd(v_medium_bs_ptr + 3*pix_id + 1, v_medium_bs_pix_local.y);
//                atomicAdd(v_medium_bs_ptr + 3*pix_id + 2, v_medium_bs_pix_local.z);
//
//                float* v_medium_omega_ptr = (float*)(v_medium_omega);
//                atomicAdd(v_medium_omega_ptr + 3*pix_id + 0, v_medium_omega_pix_local.x);
//                atomicAdd(v_medium_omega_ptr + 3*pix_id + 1, v_medium_omega_pix_local.y);
//                atomicAdd(v_medium_omega_ptr + 3*pix_id + 2, v_medium_omega_pix_local.z);
//            }
//        }
//    }
//
//    block.sync();
//
//    if (inside) {
//        float3 v_medium_rgb_pix_local  = {0.f, 0.f, 0.f};
//        float3 v_medium_bs_pix_local   = {0.f, 0.f, 0.f};
//        float3 v_medium_attn_pix_local = {0.f, 0.f, 0.f};
//        float3 v_medium_omega_pix_local= {0.f, 0.f, 0.f};
//
//        float3 exp_bs = {1.f, 1.f, 1.f};
//        T = 1.f;
//
//        v_medium_rgb_pix_local.x += v_out_med.x * T * (exp_bs.x - latter_exp_bs.x);
//        v_medium_rgb_pix_local.y += v_out_med.y * T * (exp_bs.y - latter_exp_bs.y);
//        v_medium_rgb_pix_local.z += v_out_med.z * T * (exp_bs.z - latter_exp_bs.z);
//
//        float3 v_exp_bs_local = {
//            v_out_med.x * T * medium_rgb_pix.x,
//            v_out_med.y * T * medium_rgb_pix.y,
//            v_out_med.z * T * medium_rgb_pix.z
//        };
//        v_medium_bs_pix_local.x += v_exp_bs_local.x * latter_depth * latter_exp_bs.x;
//        v_medium_bs_pix_local.y += v_exp_bs_local.y * latter_depth * latter_exp_bs.y;
//        v_medium_bs_pix_local.z += v_exp_bs_local.z * latter_depth * latter_exp_bs.z;
//
//        if (has_latter) {
//            float z_mid_tail      = 0.5f * latter_depth;
//            float delta_z_tail    = latter_depth;
//            float geom_atten_tail = 1.0f / (z_mid_tail * z_mid_tail + 1e-6f);
//            float3 exp_active_tail;
//            exp_active_tail.x = __expf(-2.0f * medium_attn_pix.x * z_mid_tail);
//            exp_active_tail.y = __expf(-2.0f * medium_attn_pix.y * z_mid_tail);
//            exp_active_tail.z = __expf(-2.0f * medium_attn_pix.z * z_mid_tail);
//
//            v_medium_omega_pix_local.x += v_out_med.x * geom_atten_tail * exp_active_tail.x * delta_z_tail;
//            v_medium_omega_pix_local.y += v_out_med.y * geom_atten_tail * exp_active_tail.y * delta_z_tail;
//            v_medium_omega_pix_local.z += v_out_med.z * geom_atten_tail * exp_active_tail.z * delta_z_tail;
//
//            v_medium_attn_pix_local.x += v_out_med.x * medium_omega_pix.x * geom_atten_tail * (-2.0f * z_mid_tail) * exp_active_tail.x * delta_z_tail;
//            v_medium_attn_pix_local.y += v_out_med.y * medium_omega_pix.y * geom_atten_tail * (-2.0f * z_mid_tail) * exp_active_tail.y * delta_z_tail;
//            v_medium_attn_pix_local.z += v_out_med.z * medium_omega_pix.z * geom_atten_tail * (-2.0f * z_mid_tail) * exp_active_tail.z * delta_z_tail;
//        }
//
//        float* v_medium_rgb_ptr = (float*)(v_medium_rgb);
//        atomicAdd(v_medium_rgb_ptr + 3*pix_id + 0, v_medium_rgb_pix_local.x);
//        atomicAdd(v_medium_rgb_ptr + 3*pix_id + 1, v_medium_rgb_pix_local.y);
//        atomicAdd(v_medium_rgb_ptr + 3*pix_id + 2, v_medium_rgb_pix_local.z);
//
//        float* v_medium_bs_ptr = (float*)(v_medium_bs);
//        atomicAdd(v_medium_bs_ptr + 3*pix_id + 0, v_medium_bs_pix_local.x);
//        atomicAdd(v_medium_bs_ptr + 3*pix_id + 1, v_medium_bs_pix_local.y);
//        atomicAdd(v_medium_bs_ptr + 3*pix_id + 2, v_medium_bs_pix_local.z);
//
//        if (has_latter) {
//            float* v_medium_attn_ptr = (float*)(v_medium_attn);
//            atomicAdd(v_medium_attn_ptr + 3*pix_id + 0, v_medium_attn_pix_local.x);
//            atomicAdd(v_medium_attn_ptr + 3*pix_id + 1, v_medium_attn_pix_local.y);
//            atomicAdd(v_medium_attn_ptr + 3*pix_id + 2, v_medium_attn_pix_local.z);
//
//            float* v_medium_omega_ptr = (float*)(v_medium_omega);
//            atomicAdd(v_medium_omega_ptr + 3*pix_id + 0, v_medium_omega_pix_local.x);
//            atomicAdd(v_medium_omega_ptr + 3*pix_id + 1, v_medium_omega_pix_local.y);
//            atomicAdd(v_medium_omega_ptr + 3*pix_id + 2, v_medium_omega_pix_local.z);
//        }
//    }
//}





//----------------version0------------------
//__global__ void rasterize_backward_kernel(
//    const dim3 tile_bounds,
//    const dim3 img_size,
//    const int32_t* __restrict__ gaussian_ids_sorted,
//    const int2* __restrict__ tile_bins,
//    const float2* __restrict__ xys,
//    float2* __restrict__ xys_grad_abs,
//    const float3* __restrict__ conics,
//    const float3* __restrict__ rgbs,
//    const float* __restrict__ opacities,
//    const float3* __restrict__ medium_rgb,
//    const float3* __restrict__ medium_bs,
//    const float3* __restrict__ medium_attn,
//    const float3* __restrict__ medium_omega,
//    const float* __restrict__ depths,
//    const float3& __restrict__ background,
//    const float* __restrict__ final_Ts,
//    const int* __restrict__ final_index,
//    const int* __restrict__ first_index,
//    const float3* __restrict__ v_output,
//    const float3* __restrict__ v_out_medium,
//    const float* __restrict__ v_output_alpha,
//    float2* __restrict__ v_xy,
//    float3* __restrict__ v_conic,
//    float3* __restrict__ v_rgb,
//    float* __restrict__ v_opacity,
//    float3* __restrict__ v_medium_rgb,
//    float3* __restrict__ v_medium_bs,
//    float3* __restrict__ v_medium_attn,
//    float3* __restrict__ v_medium_omega
//) {
//    auto block = cg::this_thread_block();
//    int32_t tile_id =
//        block.group_index().y * tile_bounds.x + block.group_index().x;
//    unsigned i =
//        block.group_index().y * block.group_dim().y + block.thread_index().y;
//    unsigned j =
//        block.group_index().x * block.group_dim().x + block.thread_index().x;
//
//    const float px = (float)j;
//    const float py = (float)i;
//    // clamp this value to the last pixel
//    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);
//
//    // keep not rasterizing threads around for reading data
//    const bool inside = (i < img_size.y && j < img_size.x);
//
//    // this is the T AFTER the last gaussian in this pixel
//    float T_final = final_Ts[pix_id];
//    float T = T_final;
//    // the contribution from gaussians behind the current one
//    float3 buffer = {0.f, 0.f, 0.f};
//    // the contribution from medium behind the current one
//    float3 buffer_medium = {0.f, 0.f, 0.f};
//    // index of last gaussian to contribute to this pixel
//    const int bin_final = inside ? final_index[pix_id] : 0;
//    // index of first gaussian to contribute to this pixel
//    // const int bin_first = inside ? first_index[pix_id] : 0;
//
//    // have all threads in tile process the same gaussians in batches
//    // first collect gaussians between range.x and range.y in batches
//    // which gaussians to look through in this tile
//    const int2 range = tile_bins[tile_id];
//    const int block_size = block.size();
//    const int num_batches = (range.y - range.x + block_size - 1) / block_size;
//
//    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
//    __shared__ float3 xy_opacity_batch[MAX_BLOCK_SIZE];
//    __shared__ float3 conic_batch[MAX_BLOCK_SIZE];
//    __shared__ float3 rgbs_batch[MAX_BLOCK_SIZE];
//    __shared__ float depth_batch[MAX_BLOCK_SIZE];
//
//    // df/d_out for this pixel
//    const float3 v_out = v_output[pix_id];
//    const float3 v_out_med = v_out_medium[pix_id];
//    const float v_out_alpha = v_output_alpha[pix_id];
//
//    const float3 medium_rgb_pix = medium_rgb[pix_id];
//    const float3 medium_bs_pix = medium_bs[pix_id];
//    const float3 medium_attn_pix = medium_attn[pix_id];
//    const float3 medium_omega_pix = medium_omega[pix_id];
//    // get the smallest one of medium_attn_pix xyz
//    float min_medium_attn_pix = std::min(medium_attn_pix.x, std::min(medium_attn_pix.y, medium_attn_pix.z));
//    min_medium_attn_pix = std::min(0.f, min_medium_attn_pix);
//
//    // latter depth
//    float latter_depth = 10.f;
//    float3 latter_exp_bs = {0.f, 0.f, 0.f};
//
//    // collect and process batches of gaussians
//    // each thread loads one gaussian at a time before rasterizing
//    const int tr = block.thread_rank();
//    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
//    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
//    for (int b = 0; b < num_batches; ++b) {
//        // resync all threads before writing next batch of shared mem
//        block.sync();
//
//        // each thread fetch 1 gaussian from back to front
//        // 0 index will be furthest back in batch
//        // index of gaussian to load
//        // batch end is the index of the last gaussian in the batch
//        const int batch_end = range.y - 1 - block_size * b;
//        int batch_size = min(block_size, batch_end + 1 - range.x);
//        const int idx = batch_end - tr;
//        if (idx >= range.x) {
//            int32_t g_id = gaussian_ids_sorted[idx];
//            id_batch[tr] = g_id;
//            const float2 xy = xys[g_id];
//            const float opac = opacities[g_id];
//            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
//            conic_batch[tr] = conics[g_id];
//            rgbs_batch[tr] = rgbs[g_id];
//            depth_batch[tr] = depths[g_id];
//        }
//        // wait for other threads to collect the gaussians in batch
//        block.sync();
//        // process gaussians in the current batch for this pixel
//        // 0 index is the furthest back gaussian in the batch
//        for (int t = max(0,batch_end - warp_bin_final); t < batch_size; ++t) {
//            int valid = inside;
//            if (batch_end - t > bin_final) {
//                valid = 0;
//            }
//            float alpha;
//            float opac;
//            float2 delta;
//            float3 conic;
//            float vis;
//            float depth;
//            if(valid){
//                depth = depth_batch[t];
//                conic = conic_batch[t];
//                float3 xy_opac = xy_opacity_batch[t];
//                opac = xy_opac.z;
//                delta = {xy_opac.x - px, xy_opac.y - py};
//                float sigma = 0.5f * (conic.x * delta.x * delta.x +
//                                            conic.z * delta.y * delta.y) +
//                                    conic.y * delta.x * delta.y;
//                vis = __expf(-sigma);
//                alpha = min(0.99f, opac * vis);
//                if (sigma < 0.f || alpha * __expf(-min_medium_attn_pix * depth) < 1.f / 255.f) {
//                    valid = 0;
//                }
//            }
//            // if all threads are inactive in this warp, skip this loop
//            if(!warp.any(valid)){
//                continue;
//            }
//            float3 v_rgb_local = {0.f, 0.f, 0.f};
//            float3 v_conic_local = {0.f, 0.f, 0.f};
//            float2 v_xy_local = {0.f, 0.f};
//            float2 v_xy_abs_local = {0.f, 0.f};
//            float v_z_abs_local = 0.f;
//            float v_opacity_local = 0.f;
//            float3 v_medium_rgb_pix_local = {0.f, 0.f, 0.f};
//            float3 v_medium_bs_pix_local = {0.f, 0.f, 0.f};
//            float3 v_medium_attn_pix_local = {0.f, 0.f, 0.f};
//            float3 v_medium_omega_pix_local = {0.f, 0.f, 0.f};
//            //initialize everything to 0, only set if the lane is valid
//            if(valid){
//                float3 exp_attn;
//                exp_attn.x = __expf(-medium_attn_pix.x * depth);
//                exp_attn.y = __expf(-medium_attn_pix.y * depth);
//                exp_attn.z = __expf(-medium_attn_pix.z * depth);
//                float3 exp_bs;
//                exp_bs.x = __expf(-medium_bs_pix.x * depth);
//                exp_bs.y = __expf(-medium_bs_pix.y * depth);
//                exp_bs.z = __expf(-medium_bs_pix.z * depth);
//
//                float depth_sq = depth * depth;
//                float3 A_depth;
//                A_depth.x = (medium_omega_pix.x / (depth_sq + medium_omega_pix.x + 1e-7f)) * exp_attn.x;
//                A_depth.y = (medium_omega_pix.y / (depth_sq + medium_omega_pix.y + 1e-7f)) * exp_attn.y;
//                A_depth.z = (medium_omega_pix.z / (depth_sq + medium_omega_pix.z + 1e-7f)) * exp_attn.z;
//                float3 B_inf_new_depth;
//                B_inf_new_depth.x = (1.0f - medium_rgb_pix.x) * A_depth.x + medium_rgb_pix.x;
//                B_inf_new_depth.y = (1.0f - medium_rgb_pix.y) * A_depth.y + medium_rgb_pix.y;
//                B_inf_new_depth.z = (1.0f - medium_rgb_pix.z) * A_depth.z + medium_rgb_pix.z;
//
//                // update the running sum of medium for current depth to update current alpha
//                // T not updated yet
//                buffer_medium.x += T * B_inf_new_depth.x * exp_bs.x;
//                buffer_medium.y += T * B_inf_new_depth.y * exp_bs.y;
//                buffer_medium.z += T * B_inf_new_depth.z * exp_bs.z;
//
//                // 确定积分区间的评估点 z_eval。若是最后一个高斯到背景的无限远区间，评估点为 depth；否则为 latter_depth。
//                bool is_bg_interval = (latter_exp_bs.x == 0.f && latter_exp_bs.y == 0.f && latter_exp_bs.z == 0.f);
//                float z_eval = is_bg_interval ? depth : latter_depth;
//                float z_eval_sq = z_eval * z_eval;
//                // 计算评估点处的衰减 exp(-beta_D * z_eval)
//                float3 exp_attn_eval = {
//                    __expf(-medium_attn_pix.x * z_eval),
//                    __expf(-medium_attn_pix.y * z_eval),
//                    __expf(-medium_attn_pix.z * z_eval)
//                };
//                // 计算评估点处的 A(z_eval)
//                float denom_x = z_eval_sq + medium_omega_pix.x + 1e-7f;
//                float denom_y = z_eval_sq + medium_omega_pix.y + 1e-7f;
//                float denom_z = z_eval_sq + medium_omega_pix.z + 1e-7f;
//                float3 A_eval = {
//                    (medium_omega_pix.x / denom_x) * exp_attn_eval.x,
//                    (medium_omega_pix.y / denom_y) * exp_attn_eval.y,
//                    (medium_omega_pix.z / denom_z) * exp_attn_eval.z
//                };
//                // 整个体积介质项在该区间的梯度来源 (即 dL / dB_inf_new)
//                float3 V_Bnew = {
//                    v_out_med.x * T * (exp_bs.x - latter_exp_bs.x),
//                    v_out_med.y * T * (exp_bs.y - latter_exp_bs.y),
//                    v_out_med.z * T * (exp_bs.z - latter_exp_bs.z)
//                };
//                // 1. 累积到 B_inf (medium_rgb) 的梯度: dL/dB_inf_new * (1 - A)
//                v_medium_rgb_pix_local.x += V_Bnew.x * (1.0f - A_eval.x);
//                v_medium_rgb_pix_local.y += V_Bnew.y * (1.0f - A_eval.y);
//                v_medium_rgb_pix_local.z += V_Bnew.z * (1.0f - A_eval.z);
//                // 2. A(z) 的中间梯度
//                float3 V_A = {
//                    V_Bnew.x * (1.0f - medium_rgb_pix.x),
//                    V_Bnew.y * (1.0f - medium_rgb_pix.y),
//                    V_Bnew.z * (1.0f - medium_rgb_pix.z)
//                };
//                // 3. 累积到 lambda (medium_omega) 的梯度: dL/dA * [ z^2 / (z^2 + lambda)^2 ] * exp(...)
//                v_medium_omega_pix_local.x += V_A.x * (z_eval_sq / (denom_x * denom_x)) * exp_attn_eval.x;
//                v_medium_omega_pix_local.y += V_A.y * (z_eval_sq / (denom_y * denom_y)) * exp_attn_eval.y;
//                v_medium_omega_pix_local.z += V_A.z * (z_eval_sq / (denom_z * denom_z)) * exp_attn_eval.z;
//                // 4. 累积到吸收系数 beta_D (medium_attn) 的梯度: dL/dA * (-z_eval) * A
//                v_medium_attn_pix_local.x += V_A.x * (-z_eval) * A_eval.x;
//                v_medium_attn_pix_local.y += V_A.y * (-z_eval) * A_eval.y;
//                v_medium_attn_pix_local.z += V_A.z * (-z_eval) * A_eval.z;
//                // 5. 更新散射系数 beta_A (medium_bs) 的梯度，替换掉原先的 medium_rgb_pix
//                float3 B_inf_new_eval = {
//                    (1.0f - medium_rgb_pix.x) * A_eval.x + medium_rgb_pix.x,
//                    (1.0f - medium_rgb_pix.y) * A_eval.y + medium_rgb_pix.y,
//                    (1.0f - medium_rgb_pix.z) * A_eval.z + medium_rgb_pix.z
//                };
//                float3 v_exp_bs_local = {
//                    v_out_med.x * T * B_inf_new_eval.x,
//                    v_out_med.y * T * B_inf_new_eval.y,
//                    v_out_med.z * T * B_inf_new_eval.z
//                };
//                v_medium_bs_pix_local.x += v_exp_bs_local.x * (-depth * exp_bs.x + latter_depth * latter_exp_bs.x);
//                v_medium_bs_pix_local.y += v_exp_bs_local.y * (-depth * exp_bs.y + latter_depth * latter_exp_bs.y);
//                v_medium_bs_pix_local.z += v_exp_bs_local.z * (-depth * exp_bs.z + latter_depth * latter_exp_bs.z);
//                // ======== [修改 4 结束] ========
//
//                // compute the current T for this gaussian
//                float ra = 1.f / (1.f - alpha);
//                T *= ra;
//
//                v_z_abs_local += fabsf(v_exp_bs_local.x * medium_bs_pix.x * exp_bs.x * (ra + 1.f));
//                v_z_abs_local += fabsf(v_exp_bs_local.y * medium_bs_pix.y * exp_bs.y * (ra + 1.f));
//                v_z_abs_local += fabsf(v_exp_bs_local.z * medium_bs_pix.z * exp_bs.z * (ra + 1.f));
//
//                // update v_rgb for this gaussian
//                const float3 rgb = rgbs_batch[t];
//                const float fac = alpha * T;
//                float v_alpha = 0.f;
//                float3 exp_attn_fac = {fac * exp_attn.x, fac * exp_attn.y, fac * exp_attn.z};
//                v_rgb_local = {v_out.x * exp_attn_fac.x, v_out.y * exp_attn_fac.y, v_out.z * exp_attn_fac.z};
//                float3 v_exp_attn_local = {v_rgb_local.x * rgb.x, v_rgb_local.y * rgb.y, v_rgb_local.z * rgb.z};
//
//                // update v_medium_attn
//                v_medium_attn_pix_local.x += -v_exp_attn_local.x * depth;
//                v_medium_attn_pix_local.y += -v_exp_attn_local.y * depth;
//                v_medium_attn_pix_local.z += -v_exp_attn_local.z * depth;
//
//                v_z_abs_local += fabsf(v_exp_attn_local.x * medium_attn_pix.x);
//                v_z_abs_local += fabsf(v_exp_attn_local.y * medium_attn_pix.y);
//                v_z_abs_local += fabsf(v_exp_attn_local.z * medium_attn_pix.z);
//
//                // contribution from this pixel
//                v_alpha += (rgb.x * T * exp_attn.x - buffer.x * ra) * v_out.x - buffer_medium.x * ra * v_out_med.x;
//                v_alpha += (rgb.y * T * exp_attn.y - buffer.y * ra) * v_out.y - buffer_medium.y * ra * v_out_med.y;
//                v_alpha += (rgb.z * T * exp_attn.z - buffer.z * ra) * v_out.z - buffer_medium.z * ra * v_out_med.z;
//                v_alpha += T_final * ra * v_out_alpha;
//
//                // update the running sum
//                buffer.x += rgb.x * exp_attn_fac.x;
//                buffer.y += rgb.y * exp_attn_fac.y;
//                buffer.z += rgb.z * exp_attn_fac.z;
//
//                // update the running sum of medium for depth to update former alpha
//                buffer_medium.x += -T * B_inf_new_depth.x * exp_bs.x;
//                buffer_medium.y += -T * B_inf_new_depth.y * exp_bs.y;
//                buffer_medium.z += -T * B_inf_new_depth.z * exp_bs.z;
//
//                latter_depth = depth;
//                latter_exp_bs.x = exp_bs.x;
//                latter_exp_bs.y = exp_bs.y;
//                latter_exp_bs.z = exp_bs.z;
//
//                const float v_sigma = -opac * vis * v_alpha;
//                v_conic_local = {0.5f * v_sigma * delta.x * delta.x,
//                                 v_sigma * delta.x * delta.y,
//                                 0.5f * v_sigma * delta.y * delta.y};
//                v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
//                                    v_sigma * (conic.y * delta.x + conic.z * delta.y)};
//                v_xy_abs_local = {fabsf(v_xy_local.x), fabsf(v_xy_local.y)};
//                v_opacity_local = vis * v_alpha;
//            }
//            warpSum3(v_rgb_local, warp);
//            warpSum3(v_conic_local, warp);
//            warpSum2(v_xy_local, warp);
//            warpSum2(v_xy_abs_local, warp);
//            warpSum(v_z_abs_local, warp);
//            warpSum(v_opacity_local, warp);
//            warpSum3(v_medium_rgb_pix_local, warp);
//            warpSum3(v_medium_bs_pix_local, warp);
//            warpSum3(v_medium_attn_pix_local, warp);
//            warpSum3(v_medium_omega_pix_local, warp);
//            if (warp.thread_rank() == 0) {
//                int32_t g = id_batch[t];
//                float* v_rgb_ptr = (float*)(v_rgb);
//                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
//                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
//                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);
//
//                float* v_conic_ptr = (float*)(v_conic);
//                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
//                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
//                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
//
//                float* v_xy_ptr = (float*)(v_xy);
//                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
//                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);
//
//                float* v_xy_abs_ptr = (float*)(xys_grad_abs);
//                atomicAdd(v_xy_abs_ptr + 2*g + 0, v_xy_abs_local.x);
//                atomicAdd(v_xy_abs_ptr + 2*g + 1, v_xy_abs_local.y);
//
//                atomicAdd(v_opacity + g, v_opacity_local);
//
//                float* v_medium_attn_ptr = (float*)(v_medium_attn);
//                atomicAdd(v_medium_attn_ptr + 3*pix_id + 0, v_medium_attn_pix_local.x);
//                atomicAdd(v_medium_attn_ptr + 3*pix_id + 1, v_medium_attn_pix_local.y);
//                atomicAdd(v_medium_attn_ptr + 3*pix_id + 2, v_medium_attn_pix_local.z);
//
//                float* v_medium_rgb_ptr = (float*)(v_medium_rgb);
//                atomicAdd(v_medium_rgb_ptr + 3*pix_id + 0, v_medium_rgb_pix_local.x);
//                atomicAdd(v_medium_rgb_ptr + 3*pix_id + 1, v_medium_rgb_pix_local.y);
//                atomicAdd(v_medium_rgb_ptr + 3*pix_id + 2, v_medium_rgb_pix_local.z);
//
//                float* v_medium_bs_ptr = (float*)(v_medium_bs);
//                atomicAdd(v_medium_bs_ptr + 3*pix_id + 0, v_medium_bs_pix_local.x);
//                atomicAdd(v_medium_bs_ptr + 3*pix_id + 1, v_medium_bs_pix_local.y);
//                atomicAdd(v_medium_bs_ptr + 3*pix_id + 2, v_medium_bs_pix_local.z);
//
//                // ======== [修改 7 开始] ========
//                float* v_medium_omega_ptr = (float*)(v_medium_omega);
//                atomicAdd(v_medium_omega_ptr + 3*pix_id + 0, v_medium_omega_pix_local.x);
//                atomicAdd(v_medium_omega_ptr + 3*pix_id + 1, v_medium_omega_pix_local.y);
//                atomicAdd(v_medium_omega_ptr + 3*pix_id + 2, v_medium_omega_pix_local.z);
//                // ======== [修改 7 结束] ========
//            }
//        }
//    }
//
//    block.sync();
//
//    // the medium from the camera to the first gaussian
//    if (inside) {
//        float3 v_medium_rgb_pix_local = {0.f, 0.f, 0.f};
//        float3 v_medium_bs_pix_local = {0.f, 0.f, 0.f};
//
//        // ======== [修改 8 开始] ========
//        float3 v_medium_omega_pix_local = {0.f, 0.f, 0.f};
//        float3 v_medium_attn_pix_local = {0.f, 0.f, 0.f};
//        // ======== [修改 8 结束] ========
//
//        float3 exp_bs = {1.f, 1.f, 1.f};
//        T = 1.f;
//
//        // ======== [修改 9 开始] ========
//        // 区间 [0, latter_depth] 的求导
//        float z_eval = latter_depth;
//        float z_eval_sq = z_eval * z_eval;
//        float3 exp_attn_eval = {
//            __expf(-medium_attn_pix.x * z_eval),
//            __expf(-medium_attn_pix.y * z_eval),
//            __expf(-medium_attn_pix.z * z_eval)
//        };
//        float denom_x = z_eval_sq + medium_omega_pix.x + 1e-7f;
//        float denom_y = z_eval_sq + medium_omega_pix.y + 1e-7f;
//        float denom_z = z_eval_sq + medium_omega_pix.z + 1e-7f;
//        float3 A_eval = {
//            (medium_omega_pix.x / denom_x) * exp_attn_eval.x,
//            (medium_omega_pix.y / denom_y) * exp_attn_eval.y,
//            (medium_omega_pix.z / denom_z) * exp_attn_eval.z
//        };
//        float3 V_Bnew = {
//            v_out_med.x * T * (exp_bs.x - latter_exp_bs.x),
//            v_out_med.y * T * (exp_bs.y - latter_exp_bs.y),
//            v_out_med.z * T * (exp_bs.z - latter_exp_bs.z)
//        };
//        // B_inf
//        v_medium_rgb_pix_local.x += V_Bnew.x * (1.0f - A_eval.x);
//        v_medium_rgb_pix_local.y += V_Bnew.y * (1.0f - A_eval.y);
//        v_medium_rgb_pix_local.z += V_Bnew.z * (1.0f - A_eval.z);
//        float3 V_A = {
//            V_Bnew.x * (1.0f - medium_rgb_pix.x),
//            V_Bnew.y * (1.0f - medium_rgb_pix.y),
//            V_Bnew.z * (1.0f - medium_rgb_pix.z)
//        };
//        v_medium_omega_pix_local.x += V_A.x * (z_eval_sq / (denom_x * denom_x)) * exp_attn_eval.x;
//        v_medium_omega_pix_local.y += V_A.y * (z_eval_sq / (denom_y * denom_y)) * exp_attn_eval.y;
//        v_medium_omega_pix_local.z += V_A.z * (z_eval_sq / (denom_z * denom_z)) * exp_attn_eval.z;
//        // beta_D
//        v_medium_attn_pix_local.x += V_A.x * (-z_eval) * A_eval.x;
//        v_medium_attn_pix_local.y += V_A.y * (-z_eval) * A_eval.y;
//        v_medium_attn_pix_local.z += V_A.z * (-z_eval) * A_eval.z;
//        float3 B_inf_new_eval = {
//            (1.0f - medium_rgb_pix.x) * A_eval.x + medium_rgb_pix.x,
//            (1.0f - medium_rgb_pix.y) * A_eval.y + medium_rgb_pix.y,
//            (1.0f - medium_rgb_pix.z) * A_eval.z + medium_rgb_pix.z
//        };
//        float3 v_exp_bs_local = {
//            v_out_med.x * T * B_inf_new_eval.x,
//            v_out_med.y * T * B_inf_new_eval.y,
//            v_out_med.z * T * B_inf_new_eval.z
//        };
//        v_medium_bs_pix_local.x += v_exp_bs_local.x * latter_depth * latter_exp_bs.x;
//        v_medium_bs_pix_local.y += v_exp_bs_local.y * latter_depth * latter_exp_bs.y;
//        v_medium_bs_pix_local.z += v_exp_bs_local.z * latter_depth * latter_exp_bs.z;
//        // ======== [修改 9 结束] ========
//
//        float* v_medium_omega_ptr = (float*)(v_medium_omega);
//        atomicAdd(v_medium_omega_ptr + 3*pix_id + 0, v_medium_omega_pix_local.x);
//        atomicAdd(v_medium_omega_ptr + 3*pix_id + 1, v_medium_omega_pix_local.y);
//        atomicAdd(v_medium_omega_ptr + 3*pix_id + 2, v_medium_omega_pix_local.z);
//
//        float* v_medium_bs_ptr = (float*)(v_medium_bs);
//        atomicAdd(v_medium_bs_ptr + 3*pix_id + 0, v_medium_bs_pix_local.x);
//        atomicAdd(v_medium_bs_ptr + 3*pix_id + 1, v_medium_bs_pix_local.y);
//        atomicAdd(v_medium_bs_ptr + 3*pix_id + 2, v_medium_bs_pix_local.z);
//
//        // 将原代码遗漏的 v_medium_attn (beta_D) 的原子加在这里补上，因为这个收尾区间现在对它产生了梯度！
//        float* v_medium_attn_ptr = (float*)(v_medium_attn);
//        atomicAdd(v_medium_attn_ptr + 3*pix_id + 0, v_medium_attn_pix_local.x);
//        atomicAdd(v_medium_attn_ptr + 3*pix_id + 1, v_medium_attn_pix_local.y);
//        atomicAdd(v_medium_attn_ptr + 3*pix_id + 2, v_medium_attn_pix_local.z);
//        // ======== [修改 10 结束] ========
//    }
//}

////----------------version1------------------
//__global__ void rasterize_backward_kernel(
//    const dim3 tile_bounds,
//    const dim3 img_size,
//    const int32_t* __restrict__ gaussian_ids_sorted,
//    const int2* __restrict__ tile_bins,
//    const float2* __restrict__ xys,
//    float2* __restrict__ xys_grad_abs,
//    const float3* __restrict__ conics,
//    const float3* __restrict__ rgbs,
//    const float* __restrict__ opacities,
//    const float3* __restrict__ medium_rgb,
//    const float3* __restrict__ medium_bs,
//    const float3* __restrict__ medium_attn,
//    const float3* __restrict__ medium_omega,
//    const float* __restrict__ depths,
//    const float3& __restrict__ background,
//    const float* __restrict__ final_Ts,
//    const int* __restrict__ final_index,
//    const int* __restrict__ first_index,
//    const float3* __restrict__ v_output,
//    const float3* __restrict__ v_out_medium,
//    const float* __restrict__ v_output_alpha,
//    float2* __restrict__ v_xy,
//    float3* __restrict__ v_conic,
//    float3* __restrict__ v_rgb,
//    float* __restrict__ v_opacity,
//    float3* __restrict__ v_medium_rgb,
//    float3* __restrict__ v_medium_bs,
//    float3* __restrict__ v_medium_attn,
//    float3* __restrict__ v_medium_omega
//) {
//    auto block = cg::this_thread_block();
//    int32_t tile_id =
//        block.group_index().y * tile_bounds.x + block.group_index().x;
//    unsigned i =
//        block.group_index().y * block.group_dim().y + block.thread_index().y;
//    unsigned j =
//        block.group_index().x * block.group_dim().x + block.thread_index().x;
//
//    const float px = (float)j;
//    const float py = (float)i;
//    // clamp this value to the last pixel
//    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);
//
//    // keep not rasterizing threads around for reading data
//    const bool inside = (i < img_size.y && j < img_size.x);
//
//    // this is the T AFTER the last gaussian in this pixel
//    float T_final = final_Ts[pix_id];
//    float T = T_final;
//    // the contribution from gaussians behind the current one
//    float3 buffer = {0.f, 0.f, 0.f};
//    // the contribution from medium behind the current one
//    float3 buffer_medium = {0.f, 0.f, 0.f};
//    // index of last gaussian to contribute to this pixel
//    const int bin_final = inside ? final_index[pix_id] : 0;
//    // index of first gaussian to contribute to this pixel
//    // const int bin_first = inside ? first_index[pix_id] : 0;
//
//    // have all threads in tile process the same gaussians in batches
//    // first collect gaussians between range.x and range.y in batches
//    // which gaussians to look through in this tile
//    const int2 range = tile_bins[tile_id];
//    const int block_size = block.size();
//    const int num_batches = (range.y - range.x + block_size - 1) / block_size;
//
//    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
//    __shared__ float3 xy_opacity_batch[MAX_BLOCK_SIZE];
//    __shared__ float3 conic_batch[MAX_BLOCK_SIZE];
//    __shared__ float3 rgbs_batch[MAX_BLOCK_SIZE];
//    __shared__ float depth_batch[MAX_BLOCK_SIZE];
//
//    // df/d_out for this pixel
//    const float3 v_out = v_output[pix_id];
//    const float3 v_out_med = v_out_medium[pix_id];
//    const float v_out_alpha = v_output_alpha[pix_id];
//
//    const float3 medium_rgb_pix = medium_rgb[pix_id];
//    const float3 medium_bs_pix = medium_bs[pix_id];
//    const float3 medium_attn_pix = medium_attn[pix_id];
//    const float3 medium_omega_pix = medium_omega[pix_id];
//    // get the smallest one of medium_attn_pix xyz
//    float min_medium_attn_pix = std::min(medium_attn_pix.x, std::min(medium_attn_pix.y, medium_attn_pix.z));
//    min_medium_attn_pix = std::min(0.f, min_medium_attn_pix);
//
//    // latter depth
//    float latter_depth = 10.f;
//    float3 latter_exp_bs = {0.f, 0.f, 0.f};
//
//    float3 v_medium_rgb_pix_local = {0.f, 0.f, 0.f};
//    float3 v_medium_bs_pix_local = {0.f, 0.f, 0.f};
//    float3 v_medium_attn_pix_local = {0.f, 0.f, 0.f};
//    float3 v_medium_omega_pix_local = {0.f, 0.f, 0.f};
//    float3 latter_B_inf_new = {0.f, 0.f, 0.f};
//
//    // collect and process batches of gaussians
//    // each thread loads one gaussian at a time before rasterizing
//    const int tr = block.thread_rank();
//    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
//    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
//    for (int b = 0; b < num_batches; ++b) {
//        // resync all threads before writing next batch of shared mem
//        block.sync();
//
//        // each thread fetch 1 gaussian from back to front
//        // 0 index will be furthest back in batch
//        // index of gaussian to load
//        // batch end is the index of the last gaussian in the batch
//        const int batch_end = range.y - 1 - block_size * b;
//        int batch_size = min(block_size, batch_end + 1 - range.x);
//        const int idx = batch_end - tr;
//        if (idx >= range.x) {
//            int32_t g_id = gaussian_ids_sorted[idx];
//            id_batch[tr] = g_id;
//            const float2 xy = xys[g_id];
//            const float opac = opacities[g_id];
//            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
//            conic_batch[tr] = conics[g_id];
//            rgbs_batch[tr] = rgbs[g_id];
//            depth_batch[tr] = depths[g_id];
//        }
//        // wait for other threads to collect the gaussians in batch
//        block.sync();
//        // process gaussians in the current batch for this pixel
//        // 0 index is the furthest back gaussian in the batch
//        for (int t = max(0,batch_end - warp_bin_final); t < batch_size; ++t) {
//            int valid = inside;
//            if (batch_end - t > bin_final) {
//                valid = 0;
//            }
//            float alpha;
//            float opac;
//            float2 delta;
//            float3 conic;
//            float vis;
//            float depth;
//            if(valid){
//                depth = depth_batch[t];
//                conic = conic_batch[t];
//                float3 xy_opac = xy_opacity_batch[t];
//                opac = xy_opac.z;
//                delta = {xy_opac.x - px, xy_opac.y - py};
//                float sigma = 0.5f * (conic.x * delta.x * delta.x +
//                                            conic.z * delta.y * delta.y) +
//                                    conic.y * delta.x * delta.y;
//                vis = __expf(-sigma);
//                alpha = min(0.99f, opac * vis);
//                if (sigma < 0.f || alpha * __expf(-min_medium_attn_pix * depth) < 1.f / 255.f) {
//                    valid = 0;
//                }
//            }
//            // if all threads are inactive in this warp, skip this loop
//            if(!warp.any(valid)){
//                continue;
//            }
//            float3 v_rgb_local = {0.f, 0.f, 0.f};
//            float3 v_conic_local = {0.f, 0.f, 0.f};
//            float2 v_xy_local = {0.f, 0.f};
//            float2 v_xy_abs_local = {0.f, 0.f};
//            float v_z_abs_local = 0.f;
//            float v_opacity_local = 0.f;
//
//            if(valid){
//                float3 exp_attn;
//                exp_attn.x = __expf(-medium_attn_pix.x * depth);
//                exp_attn.y = __expf(-medium_attn_pix.y * depth);
//                exp_attn.z = __expf(-medium_attn_pix.z * depth);
//                float3 exp_bs;
//                exp_bs.x = __expf(-medium_bs_pix.x * depth);
//                exp_bs.y = __expf(-medium_bs_pix.y * depth);
//                exp_bs.z = __expf(-medium_bs_pix.z * depth);
//
//                float depth_sq = depth * depth;
//                float3 A_depth;
//                A_depth.x = (medium_omega_pix.x / (depth_sq + medium_omega_pix.x + 1e-7f)) * exp_attn.x;
//                A_depth.y = (medium_omega_pix.y / (depth_sq + medium_omega_pix.y + 1e-7f)) * exp_attn.y;
//                A_depth.z = (medium_omega_pix.z / (depth_sq + medium_omega_pix.z + 1e-7f)) * exp_attn.z;
//                float3 B_inf_new_depth;
//                B_inf_new_depth.x = (1.0f - medium_rgb_pix.x) * A_depth.x + medium_rgb_pix.x;
//                B_inf_new_depth.y = (1.0f - medium_rgb_pix.y) * A_depth.y + medium_rgb_pix.y;
//                B_inf_new_depth.z = (1.0f - medium_rgb_pix.z) * A_depth.z + medium_rgb_pix.z;
//
//                bool is_bg_interval = (latter_exp_bs.x == 0.f && latter_exp_bs.y == 0.f && latter_exp_bs.z == 0.f);
//                float z_eval = is_bg_interval ? depth : latter_depth;
//                if (is_bg_interval) {
//                    latter_B_inf_new = B_inf_new_depth;
//                }
//
//                // +++ [替换为新代码：使用 latter_B_inf_new 进行递推] +++
//                buffer_medium.x += T * latter_B_inf_new.x * exp_bs.x;
//                buffer_medium.y += T * latter_B_inf_new.y * exp_bs.y;
//                buffer_medium.z += T * latter_B_inf_new.z * exp_bs.z;
//
//                float z_eval_sq = z_eval * z_eval;
//                // 计算评估点处的衰减 exp(-beta_D * z_eval)
//                float3 exp_attn_eval = {
//                    __expf(-medium_attn_pix.x * z_eval),
//                    __expf(-medium_attn_pix.y * z_eval),
//                    __expf(-medium_attn_pix.z * z_eval)
//                };
//                // 计算评估点处的 A(z_eval)
//                float denom_x = z_eval_sq + medium_omega_pix.x + 1e-7f;
//                float denom_y = z_eval_sq + medium_omega_pix.y + 1e-7f;
//                float denom_z = z_eval_sq + medium_omega_pix.z + 1e-7f;
//                float3 A_eval = {
//                    (medium_omega_pix.x / denom_x) * exp_attn_eval.x,
//                    (medium_omega_pix.y / denom_y) * exp_attn_eval.y,
//                    (medium_omega_pix.z / denom_z) * exp_attn_eval.z
//                };
//                // 整个体积介质项在该区间的梯度来源 (即 dL / dB_inf_new)
//                float3 V_Bnew = {
//                    v_out_med.x * T * (exp_bs.x - latter_exp_bs.x),
//                    v_out_med.y * T * (exp_bs.y - latter_exp_bs.y),
//                    v_out_med.z * T * (exp_bs.z - latter_exp_bs.z)
//                };
//                // 1. 累积到 B_inf (medium_rgb) 的梯度: dL/dB_inf_new * (1 - A)
//                v_medium_rgb_pix_local.x += V_Bnew.x * (1.0f - A_eval.x);
//                v_medium_rgb_pix_local.y += V_Bnew.y * (1.0f - A_eval.y);
//                v_medium_rgb_pix_local.z += V_Bnew.z * (1.0f - A_eval.z);
//                // 2. A(z) 的中间梯度
//                float3 V_A = {
//                    V_Bnew.x * (1.0f - medium_rgb_pix.x),
//                    V_Bnew.y * (1.0f - medium_rgb_pix.y),
//                    V_Bnew.z * (1.0f - medium_rgb_pix.z)
//                };
//                // 3. 累积到 lambda (medium_omega) 的梯度: dL/dA * [ z^2 / (z^2 + lambda)^2 ] * exp(...)
//                v_medium_omega_pix_local.x += V_A.x * (z_eval_sq / (denom_x * denom_x)) * exp_attn_eval.x;
//                v_medium_omega_pix_local.y += V_A.y * (z_eval_sq / (denom_y * denom_y)) * exp_attn_eval.y;
//                v_medium_omega_pix_local.z += V_A.z * (z_eval_sq / (denom_z * denom_z)) * exp_attn_eval.z;
//                // 4. 累积到吸收系数 beta_D (medium_attn) 的梯度: dL/dA * (-z_eval) * A
//                v_medium_attn_pix_local.x += V_A.x * (-z_eval) * A_eval.x;
//                v_medium_attn_pix_local.y += V_A.y * (-z_eval) * A_eval.y;
//                v_medium_attn_pix_local.z += V_A.z * (-z_eval) * A_eval.z;
//                // 5. 更新散射系数 beta_A (medium_bs) 的梯度，替换掉原先的 medium_rgb_pix
//                float3 B_inf_new_eval = {
//                    (1.0f - medium_rgb_pix.x) * A_eval.x + medium_rgb_pix.x,
//                    (1.0f - medium_rgb_pix.y) * A_eval.y + medium_rgb_pix.y,
//                    (1.0f - medium_rgb_pix.z) * A_eval.z + medium_rgb_pix.z
//                };
//                float3 v_exp_bs_local = {
//                    v_out_med.x * T * B_inf_new_eval.x,
//                    v_out_med.y * T * B_inf_new_eval.y,
//                    v_out_med.z * T * B_inf_new_eval.z
//                };
//                v_medium_bs_pix_local.x += v_exp_bs_local.x * (-depth * exp_bs.x + latter_depth * latter_exp_bs.x);
//                v_medium_bs_pix_local.y += v_exp_bs_local.y * (-depth * exp_bs.y + latter_depth * latter_exp_bs.y);
//                v_medium_bs_pix_local.z += v_exp_bs_local.z * (-depth * exp_bs.z + latter_depth * latter_exp_bs.z);
//                // ======== [修改 4 结束] ========
//
//
//                // compute the current T for this gaussian
//                float ra = 1.f / (1.f - alpha);
//                T *= ra;
//                v_z_abs_local += fabsf(v_exp_bs_local.x * medium_bs_pix.x * exp_bs.x * (ra + 1.f));
//                v_z_abs_local += fabsf(v_exp_bs_local.y * medium_bs_pix.y * exp_bs.y * (ra + 1.f));
//                v_z_abs_local += fabsf(v_exp_bs_local.z * medium_bs_pix.z * exp_bs.z * (ra + 1.f));
//
//                // update v_rgb for this gaussian
//                const float3 rgb = rgbs_batch[t];
//                const float fac = alpha * T;
//                float v_alpha = 0.f;
//                float3 exp_attn_fac = {fac * exp_attn.x, fac * exp_attn.y, fac * exp_attn.z};
//                v_rgb_local = {v_out.x * exp_attn_fac.x, v_out.y * exp_attn_fac.y, v_out.z * exp_attn_fac.z};
//                float3 v_exp_attn_local = {v_rgb_local.x * rgb.x, v_rgb_local.y * rgb.y, v_rgb_local.z * rgb.z};
//
//                // update v_medium_attn
//                v_medium_attn_pix_local.x += -v_exp_attn_local.x * depth;
//                v_medium_attn_pix_local.y += -v_exp_attn_local.y * depth;
//                v_medium_attn_pix_local.z += -v_exp_attn_local.z * depth;
//
//                v_z_abs_local += fabsf(v_exp_attn_local.x * medium_attn_pix.x);
//                v_z_abs_local += fabsf(v_exp_attn_local.y * medium_attn_pix.y);
//                v_z_abs_local += fabsf(v_exp_attn_local.z * medium_attn_pix.z);
//
//                // contribution from this pixel
//                v_alpha += (rgb.x * T * exp_attn.x - buffer.x * ra) * v_out.x - buffer_medium.x * ra * v_out_med.x;
//                v_alpha += (rgb.y * T * exp_attn.y - buffer.y * ra) * v_out.y - buffer_medium.y * ra * v_out_med.y;
//                v_alpha += (rgb.z * T * exp_attn.z - buffer.z * ra) * v_out.z - buffer_medium.z * ra * v_out_med.z;
//                v_alpha += T_final * ra * v_out_alpha;
//
//                // update the running sum
//                buffer.x += rgb.x * exp_attn_fac.x;
//                buffer.y += rgb.y * exp_attn_fac.y;
//                buffer.z += rgb.z * exp_attn_fac.z;
//
//                // update the running sum of medium for depth to update former alpha
//                buffer_medium.x += -T * B_inf_new_depth.x * exp_bs.x;
//                buffer_medium.y += -T * B_inf_new_depth.y * exp_bs.y;
//                buffer_medium.z += -T * B_inf_new_depth.z * exp_bs.z;
//
//                latter_depth = depth;
//                latter_exp_bs.x = exp_bs.x;
//                latter_exp_bs.y = exp_bs.y;
//                latter_exp_bs.z = exp_bs.z;
//
//                latter_B_inf_new = B_inf_new_depth;
//
//                const float v_sigma = -opac * vis * v_alpha;
//                v_conic_local = {0.5f * v_sigma * delta.x * delta.x,
//                                 v_sigma * delta.x * delta.y,
//                                 0.5f * v_sigma * delta.y * delta.y};
//                v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
//                                    v_sigma * (conic.y * delta.x + conic.z * delta.y)};
//                v_xy_abs_local = {fabsf(v_xy_local.x), fabsf(v_xy_local.y)};
//                v_opacity_local = vis * v_alpha;
//            }
//            warpSum3(v_rgb_local, warp);
//            warpSum3(v_conic_local, warp);
//            warpSum2(v_xy_local, warp);
//            warpSum2(v_xy_abs_local, warp);
//            warpSum(v_z_abs_local, warp);
//            warpSum(v_opacity_local, warp);
//
//
//            if (warp.thread_rank() == 0) {
//                int32_t g = id_batch[t];
//                float* v_rgb_ptr = (float*)(v_rgb);
//                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
//                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
//                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);
//
//                float* v_conic_ptr = (float*)(v_conic);
//                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
//                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
//                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
//
//                float* v_xy_ptr = (float*)(v_xy);
//                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
//                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);
//
//                float* v_xy_abs_ptr = (float*)(xys_grad_abs);
//                atomicAdd(v_xy_abs_ptr + 2*g + 0, v_xy_abs_local.x);
//                atomicAdd(v_xy_abs_ptr + 2*g + 1, v_xy_abs_local.y);
//
//                atomicAdd(v_opacity + g, v_opacity_local);
//            }
//        }
//    }
//
//    block.sync();
//
//    // +++ [替换为以下完整的新 if (inside) 代码块] +++
//    if (inside) {
//        float3 exp_bs = {1.f, 1.f, 1.f};
//        T = 1.f;
//
//        float z_eval = latter_depth;
//        float z_eval_sq = z_eval * z_eval;
//        float3 exp_attn_eval = {
//            __expf(-medium_attn_pix.x * z_eval),
//            __expf(-medium_attn_pix.y * z_eval),
//            __expf(-medium_attn_pix.z * z_eval)
//        };
//        float denom_x = z_eval_sq + medium_omega_pix.x + 1e-7f;
//        float denom_y = z_eval_sq + medium_omega_pix.y + 1e-7f;
//        float denom_z = z_eval_sq + medium_omega_pix.z + 1e-7f;
//        float3 A_eval = {
//            (medium_omega_pix.x / denom_x) * exp_attn_eval.x,
//            (medium_omega_pix.y / denom_y) * exp_attn_eval.y,
//            (medium_omega_pix.z / denom_z) * exp_attn_eval.z
//        };
//        float3 V_Bnew = {
//            v_out_med.x * T * (exp_bs.x - latter_exp_bs.x),
//            v_out_med.y * T * (exp_bs.y - latter_exp_bs.y),
//            v_out_med.z * T * (exp_bs.z - latter_exp_bs.z)
//        };
//
//        v_medium_rgb_pix_local.x += V_Bnew.x * (1.0f - A_eval.x);
//        v_medium_rgb_pix_local.y += V_Bnew.y * (1.0f - A_eval.y);
//        v_medium_rgb_pix_local.z += V_Bnew.z * (1.0f - A_eval.z);
//
//        float3 V_A = {
//            V_Bnew.x * (1.0f - medium_rgb_pix.x),
//            V_Bnew.y * (1.0f - medium_rgb_pix.y),
//            V_Bnew.z * (1.0f - medium_rgb_pix.z)
//        };
//
//        v_medium_omega_pix_local.x += V_A.x * (z_eval_sq / (denom_x * denom_x)) * exp_attn_eval.x;
//        v_medium_omega_pix_local.y += V_A.y * (z_eval_sq / (denom_y * denom_y)) * exp_attn_eval.y;
//        v_medium_omega_pix_local.z += V_A.z * (z_eval_sq / (denom_z * denom_z)) * exp_attn_eval.z;
//
//        v_medium_attn_pix_local.x += V_A.x * (-z_eval) * A_eval.x;
//        v_medium_attn_pix_local.y += V_A.y * (-z_eval) * A_eval.y;
//        v_medium_attn_pix_local.z += V_A.z * (-z_eval) * A_eval.z;
//
//        float3 B_inf_new_eval = {
//            (1.0f - medium_rgb_pix.x) * A_eval.x + medium_rgb_pix.x,
//            (1.0f - medium_rgb_pix.y) * A_eval.y + medium_rgb_pix.y,
//            (1.0f - medium_rgb_pix.z) * A_eval.z + medium_rgb_pix.z
//        };
//        float3 v_exp_bs_local = {
//            v_out_med.x * T * B_inf_new_eval.x,
//            v_out_med.y * T * B_inf_new_eval.y,
//            v_out_med.z * T * B_inf_new_eval.z
//        };
//        v_medium_bs_pix_local.x += v_exp_bs_local.x * latter_depth * latter_exp_bs.x;
//        v_medium_bs_pix_local.y += v_exp_bs_local.y * latter_depth * latter_exp_bs.y;
//        v_medium_bs_pix_local.z += v_exp_bs_local.z * latter_depth * latter_exp_bs.z;
//
//        // 统一写回显存
//        float* v_medium_rgb_ptr = (float*)(v_medium_rgb);
//        atomicAdd(v_medium_rgb_ptr + 3*pix_id + 0, v_medium_rgb_pix_local.x);
//        atomicAdd(v_medium_rgb_ptr + 3*pix_id + 1, v_medium_rgb_pix_local.y);
//        atomicAdd(v_medium_rgb_ptr + 3*pix_id + 2, v_medium_rgb_pix_local.z);
//
//        float* v_medium_omega_ptr = (float*)(v_medium_omega);
//        atomicAdd(v_medium_omega_ptr + 3*pix_id + 0, v_medium_omega_pix_local.x);
//        atomicAdd(v_medium_omega_ptr + 3*pix_id + 1, v_medium_omega_pix_local.y);
//        atomicAdd(v_medium_omega_ptr + 3*pix_id + 2, v_medium_omega_pix_local.z);
//
//        float* v_medium_bs_ptr = (float*)(v_medium_bs);
//        atomicAdd(v_medium_bs_ptr + 3*pix_id + 0, v_medium_bs_pix_local.x);
//        atomicAdd(v_medium_bs_ptr + 3*pix_id + 1, v_medium_bs_pix_local.y);
//        atomicAdd(v_medium_bs_ptr + 3*pix_id + 2, v_medium_bs_pix_local.z);
//
//        float* v_medium_attn_ptr = (float*)(v_medium_attn);
//        atomicAdd(v_medium_attn_ptr + 3*pix_id + 0, v_medium_attn_pix_local.x);
//        atomicAdd(v_medium_attn_ptr + 3*pix_id + 1, v_medium_attn_pix_local.y);
//        atomicAdd(v_medium_attn_ptr + 3*pix_id + 2, v_medium_attn_pix_local.z);
//    }
//}

/**
 * rasterize_backward_kernel — 最终修正版 v4
 *
 * 在 v3 基础上的新增优化：
 *
 * [优化5] 逐像素介质梯度写回：atomicAdd → 直接显存写入
 *
 *   每个线程与其 pix_id 严格一一对应（Thread-to-Pixel 映射），
 *   不存在多个线程写同一 pix_id 的情况。
 *   atomicAdd 在此场景下会引入不必要的 L2 Cache 序列化开销。
 *   改为直接指针赋值，消除原子操作流水线阻塞。
 *
 *   高斯参数（v_rgb / v_conic / v_xy / v_opacity）仍保留 atomicAdd，
 *   因为 warp 内多个线程可能对同一 g_id 并发写入。
 *
 * 沿用 v1/v2/v3 的全部修正：
 *   [修正1] dz = former_depth - depth > 0
 *   [修正2] slab T 权重使用 former_T = T_{i+1}
 *   [修正3] sum_act_T_A 先更新再计算 v_alpha_act_vec
 *   [修正4] 介质梯度使用线程私有持久累加器，不参与 warpSum
 */
//-----version2------
//__global__ void rasterize_backward_kernel(
//    const dim3 tile_bounds,
//    const dim3 img_size,
//    const int32_t* __restrict__ gaussian_ids_sorted,
//    const int2* __restrict__ tile_bins,
//    const float2* __restrict__ xys,
//    float2* __restrict__ xys_grad_abs,
//    const float3* __restrict__ conics,
//    const float3* __restrict__ rgbs,
//    const float* __restrict__ opacities,
//    const float3* __restrict__ medium_rgb,
//    const float3* __restrict__ medium_bs,
//    const float3* __restrict__ medium_attn,
//    const float3* __restrict__ medium_omega,
//    const float* __restrict__ depths,
//    const float3& __restrict__ background,
//    const float* __restrict__ final_Ts,
//    const int* __restrict__ final_index,
//    const int* __restrict__ first_index,
//    const float3* __restrict__ v_output,
//    const float3* __restrict__ v_out_medium,
//    const float* __restrict__ v_output_alpha,
//    float2* __restrict__ v_xy,
//    float3* __restrict__ v_conic,
//    float3* __restrict__ v_rgb,
//    float* __restrict__ v_opacity,
//    float3* __restrict__ v_medium_rgb,
//    float3* __restrict__ v_medium_bs,
//    float3* __restrict__ v_medium_attn,
//    float3* __restrict__ v_medium_omega
//) {
//    auto block = cg::this_thread_block();
//    int32_t tile_id =
//        block.group_index().y * tile_bounds.x + block.group_index().x;
//    unsigned i =
//        block.group_index().y * block.group_dim().y + block.thread_index().y;
//    unsigned j =
//        block.group_index().x * block.group_dim().x + block.thread_index().x;
//
//    const float px = (float)j;
//    const float py = (float)i;
//    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);
//
//    const bool inside = (i < img_size.y && j < img_size.x);
//
//    float T_final = final_Ts[pix_id];
//    float T = T_final;
//
//    float3 buffer = {0.f, 0.f, 0.f};
//
//    const int bin_final = inside ? final_index[pix_id] : 0;
//
//    const int2 range = tile_bins[tile_id];
//    const int block_size = block.size();
//    const int num_batches = (range.y - range.x + block_size - 1) / block_size;
//
//    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
//    __shared__ float3 xy_opacity_batch[MAX_BLOCK_SIZE];
//    __shared__ float3 conic_batch[MAX_BLOCK_SIZE];
//    __shared__ float3 rgbs_batch[MAX_BLOCK_SIZE];
//    __shared__ float depth_batch[MAX_BLOCK_SIZE];
//
//    const float3 v_out      = v_output[pix_id];
//    const float3 v_out_med  = v_out_medium[pix_id];
//    const float  v_out_alpha = v_output_alpha[pix_id];
//
//    const float3 medium_rgb_pix   = medium_rgb[pix_id];
//    const float3 medium_bs_pix    = medium_bs[pix_id];
//    const float3 medium_attn_pix  = medium_attn[pix_id];
//    const float3 medium_omega_pix = medium_omega[pix_id];
//
//    float min_medium_attn_pix = std::min(medium_attn_pix.x,
//                                std::min(medium_attn_pix.y, medium_attn_pix.z));
//    min_medium_attn_pix = std::min(0.f, min_medium_attn_pix);
//
//    // 自然后向散射后缀累加器
//    float3 sum_alpha_T_exp_bs   = {0.f, 0.f, 0.f};
//    float3 sum_alpha_T_z_exp_bs = {0.f, 0.f, 0.f};
//
//    // 主动后向散射后缀累加器
//    float3 sum_act_T_A = {0.f, 0.f, 0.f};
//
//    // former 状态
//    float former_depth = 0.f;
//    float former_T     = 0.f;
//    bool  has_former   = false;
//
//    // 逐像素介质梯度持久累加器（线程私有，不参与 warpSum）
//    float3 accum_v_medium_bs    = {0.f, 0.f, 0.f};
//    float3 accum_v_medium_attn  = {0.f, 0.f, 0.f};
//    float3 accum_v_medium_omega = {0.f, 0.f, 0.f};
//
//    const int tr = block.thread_rank();
//    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
//    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
//
//    for (int b = 0; b < num_batches; ++b) {
//        block.sync();
//
//        const int batch_end = range.y - 1 - block_size * b;
//        int batch_size = min(block_size, batch_end + 1 - range.x);
//        const int idx = batch_end - tr;
//        if (idx >= range.x) {
//            int32_t g_id = gaussian_ids_sorted[idx];
//            id_batch[tr]         = g_id;
//            const float2 xy      = xys[g_id];
//            const float  opac    = opacities[g_id];
//            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
//            conic_batch[tr]      = conics[g_id];
//            rgbs_batch[tr]       = rgbs[g_id];
//            depth_batch[tr]      = depths[g_id];
//        }
//        block.sync();
//
//        for (int t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
//            int valid = inside;
//            if (batch_end - t > bin_final) {
//                valid = 0;
//            }
//
//            float alpha, opac, vis, depth;
//            float2 delta;
//            float3 conic;
//
//            if (valid) {
//                depth = depth_batch[t];
//                conic = conic_batch[t];
//                float3 xy_opac = xy_opacity_batch[t];
//                opac  = xy_opac.z;
//                delta = {xy_opac.x - px, xy_opac.y - py};
//                float sigma = 0.5f * (conic.x * delta.x * delta.x +
//                                      conic.z * delta.y * delta.y) +
//                              conic.y * delta.x * delta.y;
//                vis   = __expf(-sigma);
//                alpha = min(0.999f, opac * vis);
//                if (sigma < 0.f || alpha * __expf(-min_medium_attn_pix * depth) < 1.f / 255.f) {
//                    valid = 0;
//                }
//            }
//
//            if (!warp.any(valid)) {
//                continue;
//            }
//
//            // 高斯参数梯度：内层局部变量，参与 warpSum + atomicAdd
//            float3 v_rgb_local    = {0.f, 0.f, 0.f};
//            float3 v_conic_local  = {0.f, 0.f, 0.f};
//            float2 v_xy_local     = {0.f, 0.f};
//            float2 v_xy_abs_local = {0.f, 0.f};
//            float  v_opacity_local = 0.f;
//
//            if (valid) {
//                float3 exp_attn = {
//                    __expf(-medium_attn_pix.x * depth),
//                    __expf(-medium_attn_pix.y * depth),
//                    __expf(-medium_attn_pix.z * depth)
//                };
//                float3 exp_bs = {
//                    __expf(-medium_bs_pix.x * depth),
//                    __expf(-medium_bs_pix.y * depth),
//                    __expf(-medium_bs_pix.z * depth)
//                };
//
//                float ra = 1.f / (1.f - alpha);
//                T *= ra;
//                // 此后 T == T_i
//
//                // -----------------------------------------------------------
//                // 主动后向散射
//                // [修正1] dz = former_depth - depth > 0
//                // [修正2] T 权重 = former_T = T_{i+1}
//                // [修正3] 先更新 sum_act_T_A，再计算 v_alpha_act_vec
//                // [修正4] 梯度写入持久累加器，不走 warpSum
//                // -----------------------------------------------------------
//                if (has_former) {
//                    float z_right    = former_depth;
//                    float dz         = z_right - depth;   // > 0
//
//                    float3 E2 = {
//                        __expf(-2.f * medium_attn_pix.x * z_right),
//                        __expf(-2.f * medium_attn_pix.y * z_right),
//                        __expf(-2.f * medium_attn_pix.z * z_right)
//                    };
//                    float z_right_sq = z_right * z_right;
//                    float3 F_right = {
//                        medium_omega_pix.x / (z_right_sq + medium_omega_pix.x + 1e-7f),
//                        medium_omega_pix.y / (z_right_sq + medium_omega_pix.y + 1e-7f),
//                        medium_omega_pix.z / (z_right_sq + medium_omega_pix.z + 1e-7f)
//                    };
//                    float3 A_slab = {
//                        medium_bs_pix.x * F_right.x * E2.x * dz,
//                        medium_bs_pix.y * F_right.y * E2.y * dz,
//                        medium_bs_pix.z * F_right.z * E2.z * dz
//                    };
//
//                    // [修正3] 先将 j=i+1 项加入 sum
//                    sum_act_T_A.x += former_T * A_slab.x;
//                    sum_act_T_A.y += former_T * A_slab.y;
//                    sum_act_T_A.z += former_T * A_slab.z;
//
//                    float3 v_A_slab = {
//                        v_out_med.x * former_T,
//                        v_out_med.y * former_T,
//                        v_out_med.z * former_T
//                    };
//
//                    // ∂L/∂β_bs
//                    accum_v_medium_bs.x += v_A_slab.x * F_right.x * E2.x * dz;
//                    accum_v_medium_bs.y += v_A_slab.y * F_right.y * E2.y * dz;
//                    accum_v_medium_bs.z += v_A_slab.z * F_right.z * E2.z * dz;
//
//                    // ∂L/∂β_attn（主动散射部分）
//                    accum_v_medium_attn.x += v_A_slab.x * medium_bs_pix.x * F_right.x * E2.x * (-2.f * z_right) * dz;
//                    accum_v_medium_attn.y += v_A_slab.y * medium_bs_pix.y * F_right.y * E2.y * (-2.f * z_right) * dz;
//                    accum_v_medium_attn.z += v_A_slab.z * medium_bs_pix.z * F_right.z * E2.z * (-2.f * z_right) * dz;
//
//                    // ∂F/∂ω
//                    float dF_domega_x = (z_right_sq + 1e-7f) /
//                        ((z_right_sq + medium_omega_pix.x + 1e-7f) *
//                         (z_right_sq + medium_omega_pix.x + 1e-7f));
//                    float dF_domega_y = (z_right_sq + 1e-7f) /
//                        ((z_right_sq + medium_omega_pix.y + 1e-7f) *
//                         (z_right_sq + medium_omega_pix.y + 1e-7f));
//                    float dF_domega_z = (z_right_sq + 1e-7f) /
//                        ((z_right_sq + medium_omega_pix.z + 1e-7f) *
//                         (z_right_sq + medium_omega_pix.z + 1e-7f));
//
//                    // ∂L/∂ω
//                    accum_v_medium_omega.x += v_A_slab.x * medium_bs_pix.x * dF_domega_x * E2.x * dz;
//                    accum_v_medium_omega.y += v_A_slab.y * medium_bs_pix.y * dF_domega_y * E2.y * dz;
//                    accum_v_medium_omega.z += v_A_slab.z * medium_bs_pix.z * dF_domega_z * E2.z * dz;
//                }
//
//                // 自然后向散射对 α_i 的梯度
//                float3 v_alpha_nat_vec = {
//                    v_out_med.x * medium_rgb_pix.x * (-T * exp_bs.x + ra * sum_alpha_T_exp_bs.x),
//                    v_out_med.y * medium_rgb_pix.y * (-T * exp_bs.y + ra * sum_alpha_T_exp_bs.y),
//                    v_out_med.z * medium_rgb_pix.z * (-T * exp_bs.z + ra * sum_alpha_T_exp_bs.z)
//                };
//
//                // [修正3] sum_act_T_A 已含 j=i+1 项，梯度完整
//                float3 v_alpha_act_vec = {
//                    v_out_med.x * (-ra * sum_act_T_A.x),
//                    v_out_med.y * (-ra * sum_act_T_A.y),
//                    v_out_med.z * (-ra * sum_act_T_A.z)
//                };
//
//                // 直接信号项
//                const float3 rgb = rgbs_batch[t];
//
//                v_rgb_local = {
//                    v_out.x * alpha * T * exp_attn.x,
//                    v_out.y * alpha * T * exp_attn.y,
//                    v_out.z * alpha * T * exp_attn.z
//                };
//
//                // ∂L/∂β_attn（直接信号部分）→ 持久累加器
//                accum_v_medium_attn.x += v_out.x * (alpha * T * rgb.x) * exp_attn.x * (-depth);
//                accum_v_medium_attn.y += v_out.y * (alpha * T * rgb.y) * exp_attn.y * (-depth);
//                accum_v_medium_attn.z += v_out.z * (alpha * T * rgb.z) * exp_attn.z * (-depth);
//
//                // ∂L/∂α_i 汇总
//                float v_alpha = 0.f;
//                v_alpha += (rgb.x * T * exp_attn.x - buffer.x * ra) * v_out.x;
//                v_alpha += (rgb.y * T * exp_attn.y - buffer.y * ra) * v_out.y;
//                v_alpha += (rgb.z * T * exp_attn.z - buffer.z * ra) * v_out.z;
//                v_alpha += v_alpha_nat_vec.x + v_alpha_nat_vec.y + v_alpha_nat_vec.z;
//                v_alpha += v_alpha_act_vec.x + v_alpha_act_vec.y + v_alpha_act_vec.z;
//                v_alpha += T_final * ra * v_out_alpha;
//
//                // 更新后缀累加器
//                buffer.x += rgb.x * alpha * T * exp_attn.x;
//                buffer.y += rgb.y * alpha * T * exp_attn.y;
//                buffer.z += rgb.z * alpha * T * exp_attn.z;
//
//                sum_alpha_T_exp_bs.x   += alpha * T * exp_bs.x;
//                sum_alpha_T_exp_bs.y   += alpha * T * exp_bs.y;
//                sum_alpha_T_exp_bs.z   += alpha * T * exp_bs.z;
//                sum_alpha_T_z_exp_bs.x += alpha * T * depth * exp_bs.x;
//                sum_alpha_T_z_exp_bs.y += alpha * T * depth * exp_bs.y;
//                sum_alpha_T_z_exp_bs.z += alpha * T * depth * exp_bs.z;
//
//                former_depth = depth;
//                former_T     = T;
//                has_former   = true;
//
//                // 高斯几何梯度
//                const float v_sigma = -opac * vis * v_alpha;
//                v_conic_local = {
//                    0.5f * v_sigma * delta.x * delta.x,
//                    v_sigma * delta.x * delta.y,
//                    0.5f * v_sigma * delta.y * delta.y
//                };
//                v_xy_local = {
//                    v_sigma * (conic.x * delta.x + conic.y * delta.y),
//                    v_sigma * (conic.y * delta.x + conic.z * delta.y)
//                };
//                v_xy_abs_local  = {fabsf(v_xy_local.x), fabsf(v_xy_local.y)};
//                v_opacity_local = vis * v_alpha;
//            }
//
//            // 高斯参数：warp 规约 + atomicAdd（多像素可能写同一 g_id，必须原子）
//            warpSum3(v_rgb_local, warp);
//            warpSum3(v_conic_local, warp);
//            warpSum2(v_xy_local, warp);
//            warpSum2(v_xy_abs_local, warp);
//            warpSum(v_opacity_local, warp);
//
//            if (warp.thread_rank() == 0) {
//                int32_t g = id_batch[t];
//
//                float* v_rgb_ptr = (float*)(v_rgb);
//                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
//                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
//                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);
//
//                float* v_conic_ptr = (float*)(v_conic);
//                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
//                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
//                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
//
//                float* v_xy_ptr = (float*)(v_xy);
//                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
//                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);
//
//                float* v_xy_abs_ptr = (float*)(xys_grad_abs);
//                atomicAdd(v_xy_abs_ptr + 2*g + 0, v_xy_abs_local.x);
//                atomicAdd(v_xy_abs_ptr + 2*g + 1, v_xy_abs_local.y);
//
//                atomicAdd(v_opacity + g, v_opacity_local);
//            }
//        }
//    }
//
//    block.sync();
//
//    if (inside) {
//        // 前景段（z=0 到 z_1）主动后向散射参数梯度
//        if (has_former) {
//            float z_right    = former_depth;
//            float dz         = z_right;
//
//            float3 E2 = {
//                __expf(-2.f * medium_attn_pix.x * z_right),
//                __expf(-2.f * medium_attn_pix.y * z_right),
//                __expf(-2.f * medium_attn_pix.z * z_right)
//            };
//            float z_right_sq = z_right * z_right;
//            float3 F_right = {
//                medium_omega_pix.x / (z_right_sq + medium_omega_pix.x + 1e-7f),
//                medium_omega_pix.y / (z_right_sq + medium_omega_pix.y + 1e-7f),
//                medium_omega_pix.z / (z_right_sq + medium_omega_pix.z + 1e-7f)
//            };
//
//            float3 v_A_slab = {
//                v_out_med.x * former_T,
//                v_out_med.y * former_T,
//                v_out_med.z * former_T
//            };
//
//            accum_v_medium_bs.x += v_A_slab.x * F_right.x * E2.x * dz;
//            accum_v_medium_bs.y += v_A_slab.y * F_right.y * E2.y * dz;
//            accum_v_medium_bs.z += v_A_slab.z * F_right.z * E2.z * dz;
//
//            accum_v_medium_attn.x += v_A_slab.x * medium_bs_pix.x * F_right.x * E2.x * (-2.f * z_right) * dz;
//            accum_v_medium_attn.y += v_A_slab.y * medium_bs_pix.y * F_right.y * E2.y * (-2.f * z_right) * dz;
//            accum_v_medium_attn.z += v_A_slab.z * medium_bs_pix.z * F_right.z * E2.z * (-2.f * z_right) * dz;
//
//            float dF_domega_x = (z_right_sq + 1e-7f) /
//                ((z_right_sq + medium_omega_pix.x + 1e-7f) *
//                 (z_right_sq + medium_omega_pix.x + 1e-7f));
//            float dF_domega_y = (z_right_sq + 1e-7f) /
//                ((z_right_sq + medium_omega_pix.y + 1e-7f) *
//                 (z_right_sq + medium_omega_pix.y + 1e-7f));
//            float dF_domega_z = (z_right_sq + 1e-7f) /
//                ((z_right_sq + medium_omega_pix.z + 1e-7f) *
//                 (z_right_sq + medium_omega_pix.z + 1e-7f));
//
//            accum_v_medium_omega.x += v_A_slab.x * medium_bs_pix.x * dF_domega_x * E2.x * dz;
//            accum_v_medium_omega.y += v_A_slab.y * medium_bs_pix.y * dF_domega_y * E2.y * dz;
//            accum_v_medium_omega.z += v_A_slab.z * medium_bs_pix.z * dF_domega_z * E2.z * dz;
//        }
//
//        // 自然后向散射全局参数梯度
//        float3 v_medium_rgb_pix = {
//            v_out_med.x * (1.f - sum_alpha_T_exp_bs.x),
//            v_out_med.y * (1.f - sum_alpha_T_exp_bs.y),
//            v_out_med.z * (1.f - sum_alpha_T_exp_bs.z)
//        };
//
//        accum_v_medium_bs.x += v_out_med.x * medium_rgb_pix.x * sum_alpha_T_z_exp_bs.x;
//        accum_v_medium_bs.y += v_out_med.y * medium_rgb_pix.y * sum_alpha_T_z_exp_bs.y;
//        accum_v_medium_bs.z += v_out_med.z * medium_rgb_pix.z * sum_alpha_T_z_exp_bs.z;
//
//        // -----------------------------------------------------------------------
//        // [优化5] 逐像素介质梯度：直接写入显存，无需原子操作
//        // 每个线程独占自己的 pix_id，不存在并发写入竞争
//        // -----------------------------------------------------------------------
//        v_medium_rgb[pix_id]   = v_medium_rgb_pix;
//        v_medium_bs[pix_id]    = accum_v_medium_bs;
//        v_medium_attn[pix_id]  = accum_v_medium_attn;
//        v_medium_omega[pix_id] = accum_v_medium_omega;
//    }
//}


__global__ void project_gaussians_backward_kernel(
    const int num_points,
    const float3* __restrict__ means3d,
    const float3* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float* __restrict__ viewmat,
    const float4 intrins,
    const dim3 img_size,
    const float* __restrict__ cov3d,
    const int* __restrict__ radii,
    const float3* __restrict__ conics,
    const float* __restrict__ compensation,
    const float2* __restrict__ v_xy,
    const float* __restrict__ v_depth,
    const float3* __restrict__ v_conic,
    const float* __restrict__ v_compensation,
    float3* __restrict__ v_cov2d,
    float* __restrict__ v_cov3d,
    float3* __restrict__ v_mean3d,
    float3* __restrict__ v_scale,
    float4* __restrict__ v_quat
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points || radii[idx] <= 0) {
        return;
    }
    float3 p_world = means3d[idx];
    float fx = intrins.x;
    float fy = intrins.y;
    float3 p_view = transform_4x3(viewmat, p_world);
    // get v_mean3d from v_xy
    v_mean3d[idx] = transform_4x3_rot_only_transposed(
        viewmat,
        project_pix_vjp({fx, fy}, p_view, v_xy[idx]));

    // get z gradient contribution to mean3d gradient
    // z = viemwat[8] * mean3d.x + viewmat[9] * mean3d.y + viewmat[10] *
    // mean3d.z + viewmat[11]
    float v_z = v_depth[idx];
    v_mean3d[idx].x += viewmat[8] * v_z;
    v_mean3d[idx].y += viewmat[9] * v_z;
    v_mean3d[idx].z += viewmat[10] * v_z;

    // get v_cov2d
    cov2d_to_conic_vjp(conics[idx], v_conic[idx], v_cov2d[idx]);
    cov2d_to_compensation_vjp(compensation[idx], conics[idx], v_compensation[idx], v_cov2d[idx]);
    // get v_cov3d (and v_mean3d contribution)
    project_cov3d_ewa_vjp(
        p_world,
        &(cov3d[6 * idx]),
        viewmat,
        fx,
        fy,
        v_cov2d[idx],
        v_mean3d[idx],
        &(v_cov3d[6 * idx])
    );
    // get v_scale and v_quat
    scale_rot_to_cov3d_vjp(
        scales[idx],
        glob_scale,
        quats[idx],
        &(v_cov3d[6 * idx]),
        v_scale[idx],
        v_quat[idx]
    );
}

// output space: 2D covariance, input space: cov3d
__device__ void project_cov3d_ewa_vjp(
    const float3& __restrict__ mean3d,
    const float* __restrict__ cov3d,
    const float* __restrict__ viewmat,
    const float fx,
    const float fy,
    const float3& __restrict__ v_cov2d,
    float3& __restrict__ v_mean3d,
    float* __restrict__ v_cov3d
) {
    // viewmat is row major, glm is column major
    // upper 3x3 submatrix
    // clang-format off
    glm::mat3 W = glm::mat3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );
    // clang-format on
    glm::vec3 p = glm::vec3(viewmat[3], viewmat[7], viewmat[11]);
    glm::vec3 t = W * glm::vec3(mean3d.x, mean3d.y, mean3d.z) + p;
    float rz = 1.f / t.z;
    float rz2 = rz * rz;

    // column major
    // we only care about the top 2x2 submatrix
    // clang-format off
    glm::mat3 J = glm::mat3(
        fx * rz,         0.f,             0.f,
        0.f,             fy * rz,         0.f,
        -fx * t.x * rz2, -fy * t.y * rz2, 0.f
    );
    glm::mat3 V = glm::mat3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );
    // cov = T * V * Tt; G = df/dcov = v_cov
    // -> d/dV = Tt * G * T
    // -> df/dT = G * T * Vt + Gt * T * V
    glm::mat3 v_cov = glm::mat3(
        v_cov2d.x,        0.5f * v_cov2d.y, 0.f,
        0.5f * v_cov2d.y, v_cov2d.z,        0.f,
        0.f,              0.f,              0.f
    );
    // clang-format on

    glm::mat3 T = J * W;
    glm::mat3 Tt = glm::transpose(T);
    glm::mat3 Vt = glm::transpose(V);
    glm::mat3 v_V = Tt * v_cov * T;
    glm::mat3 v_T = v_cov * T * Vt + glm::transpose(v_cov) * T * V;

    // vjp of cov3d parameters
    // v_cov3d_i = v_V : dV/d_cov3d_i
    // where : is frobenius inner product
    v_cov3d[0] = v_V[0][0];
    v_cov3d[1] = v_V[0][1] + v_V[1][0];
    v_cov3d[2] = v_V[0][2] + v_V[2][0];
    v_cov3d[3] = v_V[1][1];
    v_cov3d[4] = v_V[1][2] + v_V[2][1];
    v_cov3d[5] = v_V[2][2];

    // compute df/d_mean3d
    // T = J * W
    glm::mat3 v_J = v_T * glm::transpose(W);
    float rz3 = rz2 * rz;
    glm::vec3 v_t = glm::vec3(
        -fx * rz2 * v_J[2][0],
        -fy * rz2 * v_J[2][1],
        -fx * rz2 * v_J[0][0] + 2.f * fx * t.x * rz3 * v_J[2][0] -
            fy * rz2 * v_J[1][1] + 2.f * fy * t.y * rz3 * v_J[2][1]
    );
    // printf("v_t %.2f %.2f %.2f\n", v_t[0], v_t[1], v_t[2]);
    // printf("W %.2f %.2f %.2f\n", W[0][0], W[0][1], W[0][2]);
    v_mean3d.x += (float)glm::dot(v_t, W[0]);
    v_mean3d.y += (float)glm::dot(v_t, W[1]);
    v_mean3d.z += (float)glm::dot(v_t, W[2]);
}

// given cotangent v in output space (e.g. d_L/d_cov3d) in R(6)
// compute vJp for scale and rotation
__device__ void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const float* __restrict__ v_cov3d,
    float3& __restrict__ v_scale,
    float4& __restrict__ v_quat
) {
    // cov3d is upper triangular elements of matrix
    // off-diagonal elements count grads from both ij and ji elements,
    // must halve when expanding back into symmetric matrix
    glm::mat3 v_V = glm::mat3(
        v_cov3d[0],
        0.5 * v_cov3d[1],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[1],
        v_cov3d[3],
        0.5 * v_cov3d[4],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[4],
        v_cov3d[5]
    );
    glm::mat3 R = quat_to_rotmat(quat);
    glm::mat3 S = scale_to_mat(scale, glob_scale);
    glm::mat3 M = R * S;
    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    glm::mat3 v_M = 2.f * v_V * M;
    // glm::mat3 v_S = glm::transpose(R) * v_M;
    v_scale.x = (float)glm::dot(R[0], v_M[0]) * glob_scale;
    v_scale.y = (float)glm::dot(R[1], v_M[1]) * glob_scale;
    v_scale.z = (float)glm::dot(R[2], v_M[2]) * glob_scale;

    glm::mat3 v_R = v_M * S;
    v_quat = quat_to_rotmat_vjp(quat, v_R);
}
