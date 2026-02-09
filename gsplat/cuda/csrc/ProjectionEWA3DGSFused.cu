#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void projection_ewa_3dgs_fused_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,    // [B, N, 3]
    const scalar_t *__restrict__ covars,   // [B, N, 6] optional
    const scalar_t *__restrict__ quats,    // [B, N, 4] optional
    const scalar_t *__restrict__ scales,   // [B, N, 3] optional
    const scalar_t *__restrict__ opacities, // [B, N] optional
    const scalar_t *__restrict__ viewmats, // [B, C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    int32_t *__restrict__ radii,         // [B, C, N, 2]
    scalar_t *__restrict__ means2d,      // [B, C, N, 2]
    scalar_t *__restrict__ depths,       // [B, C, N]
    scalar_t *__restrict__ normals,      // [B, C, N, 3]
    scalar_t *__restrict__ conics,       // [B, C, N, 3]
    scalar_t *__restrict__ compensations // [B, C, N] optional
) {
    // parallelize over B * C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * C * N) {
        return;
    }
    const uint32_t bid = idx / (C * N); // batch id
    const uint32_t cid = (idx / N) % C; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += bid * N * 3 + gid * 3;
    viewmats += bid * C * 16 + cid * 16;
    Ks += bid * C * 9 + cid * 9;

    // glm is column-major but input is row-major
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to camera space
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // transform Gaussian covariance to camera space
    mat3 covar;
    if (covars != nullptr) {
        covars += bid * N * 6 + gid * 6;
        covar = mat3(
            covars[0],
            covars[1],
            covars[2], // 1st column
            covars[1],
            covars[3],
            covars[4], // 2nd column
            covars[2],
            covars[4],
            covars[5] // 3rd column
        );
    } else {
        // compute from quaternions and scales
        quats += bid * N * 4 + gid * 4;
        scales += bid * N * 3 + gid * 3;
        quat_scale_to_covar_preci(
            glm::make_vec4(quats), glm::make_vec3(scales), &covar, nullptr
        );
    }
    mat3 covar_c;
    covarW2C(R, covar, covar_c);

    // perspective projection
    mat2 covar2d;
    vec2 mean2d;

    switch (camera_model) {
    case CameraModelType::PINHOLE: // perspective projection
        persp_proj(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    case CameraModelType::ORTHO: // orthographic projection
        ortho_proj(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    case CameraModelType::FISHEYE: // fisheye projection
        fisheye_proj(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    }

    float compensation;
    float det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // compute the inverse of the 2d covariance
    mat2 covar2d_inv = glm::inverse(covar2d);

    float extend = 3.33f;
    if (opacities != nullptr) {
        float opacity = opacities[bid * N + gid];
        if (compensations != nullptr) {
            // we assume compensation term will be applied later on.
            opacity *= compensation;
        }
        if (opacity < ALPHA_THRESHOLD) {
            radii[idx * 2] = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }
        // Compute opacity-aware bounding box.
        // https://arxiv.org/pdf/2402.00525 Section B.2
        extend = min(extend, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
    }

    // compute tight rectangular bounding box (non differentiable)
    // https://arxiv.org/pdf/2402.00525
    float radius_x = ceilf(extend * sqrtf(covar2d[0][0]));
    float radius_y = ceilf(extend * sqrtf(covar2d[1][1]));

    if (radius_x <= radius_clip && radius_y <= radius_clip) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // mask out gaussians outside the image region
    if (mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= image_width ||
        mean2d.y + radius_y <= 0 || mean2d.y - radius_y >= image_height) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // normals (camera space)
    vec3 normal_c(0.f);
    if (quats != nullptr && scales != nullptr) {
        // NOTE: When `covars == nullptr` we already shifted `quats/scales` to the
        // current (bid, gid) above (for covar computation). Do NOT apply the
        // (bid, gid) offset again here, otherwise it will read out-of-bounds.
        mat3 R_gauss = quat_to_rotmat(glm::make_vec4(quats));

        // Find the axis with the smallest scale. This branch is non-differentiable.
        float sx = scales[0], sy = scales[1], sz = scales[2];
        int min_idx = (sy < sx) ? 1 : 0;
        min_idx = (sz < ((min_idx == 0) ? sx : sy)) ? 2 : min_idx;

        // The column vector corresponding to the smallest scale is the normal.
        vec3 normal_w = R_gauss[min_idx];
        normal_c = R * normal_w;

        // Flip if it is pointing away from the camera.
        float multiplier = glm::dot(-normal_c, mean_c) > 0 ? 1.f : -1.f;
        normal_c *= multiplier;
    }

    // write to outputs
    radii[idx * 2] = (int32_t)radius_x;
    radii[idx * 2 + 1] = (int32_t)radius_y;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = mean_c.z;
    normals[idx * 3 + 0] = normal_c.x;
    normals[idx * 3 + 1] = normal_c.y;
    normals[idx * 3 + 2] = normal_c.z;
    conics[idx * 3] = covar2d_inv[0][0];
    conics[idx * 3 + 1] = covar2d_inv[0][1];
    conics[idx * 3 + 2] = covar2d_inv[1][1];
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }
}

void launch_projection_ewa_3dgs_fused_fwd_kernel(
    // inputs
    const at::Tensor means,                // [..., N, 3]
    const at::optional<at::Tensor> covars, // [..., N, 6] optional
    const at::optional<at::Tensor> quats,  // [..., N, 4] optional
    const at::optional<at::Tensor> scales, // [..., N, 3] optional
    const at::optional<at::Tensor> opacities, // [..., N] optional
    const at::Tensor viewmats,             // [..., C, 4, 4]
    const at::Tensor Ks,                   // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    at::Tensor radii,                      // [..., C, N, 2]
    at::Tensor means2d,                    // [..., C, N, 2]
    at::Tensor depths,                     // [..., C, N]
    at::Tensor normals,                    // [..., C, N, 3]
    at::Tensor conics,                     // [..., C, N, 3]
    at::optional<at::Tensor> compensations // [..., C, N] optional
) {
    uint32_t N = means.size(-2);    // number of gaussians
    uint32_t C = viewmats.size(-3); // number of cameras
    uint32_t B = means.numel() / (N * 3);    // number of batches

    int64_t n_elements = B * C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "projection_ewa_3dgs_fused_fwd_kernel",
        [&]() {
            projection_ewa_3dgs_fused_fwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    B,
                    C,
                    N,
                    means.data_ptr<scalar_t>(),
                    covars.has_value() ? covars.value().data_ptr<scalar_t>()
                                       : nullptr,
                    quats.has_value() ? quats.value().data_ptr<scalar_t>()
                                      : nullptr,
                    scales.has_value() ? scales.value().data_ptr<scalar_t>()
                                       : nullptr,
                    opacities.has_value() ? opacities.value().data_ptr<scalar_t>()
                                         : nullptr,
                    viewmats.data_ptr<scalar_t>(),
                    Ks.data_ptr<scalar_t>(),
                    image_width,
                    image_height,
                    eps2d,
                    near_plane,
                    far_plane,
                    radius_clip,
                    camera_model,
                    radii.data_ptr<int32_t>(),
                    means2d.data_ptr<scalar_t>(),
                    depths.data_ptr<scalar_t>(),
                    normals.data_ptr<scalar_t>(),
                    conics.data_ptr<scalar_t>(),
                    compensations.has_value()
                        ? compensations.value().data_ptr<scalar_t>()
                        : nullptr
                );
        }
    );
}

template <typename scalar_t>
__global__ void projection_ewa_3dgs_fused_bwd_kernel(
    // fwd inputs
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,    // [B, N, 3]
    const scalar_t *__restrict__ covars,   // [B, N, 6] optional
    const scalar_t *__restrict__ quats,    // [B, N, 4] optional
    const scalar_t *__restrict__ scales,   // [B, N, 3] optional
    const scalar_t *__restrict__ viewmats, // [B, C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const int32_t *__restrict__ radii,          // [B, C, N, 2]
    const scalar_t *__restrict__ conics,        // [B, C, N, 3]
    const scalar_t *__restrict__ compensations, // [B, C, N] optional
    // grad outputs
    const scalar_t *__restrict__ v_means2d,       // [B, C, N, 2]
    const scalar_t *__restrict__ v_depths,        // [B, C, N]
    const scalar_t *__restrict__ v_normals,       // [B, C, N, 3]
    const scalar_t *__restrict__ v_conics,        // [B, C, N, 3]
    const scalar_t *__restrict__ v_compensations, // [B, C, N] optional
    // grad inputs
    scalar_t *__restrict__ v_means,   // [B, N, 3]
    scalar_t *__restrict__ v_covars,  // [B, N, 6] optional
    scalar_t *__restrict__ v_quats,   // [B, N, 4] optional
    scalar_t *__restrict__ v_scales,  // [B, N, 3] optional
    scalar_t *__restrict__ v_viewmats // [B, C, 4, 4] optional
) {
    // parallelize over B * C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * C * N || radii[idx * 2] <= 0 || radii[idx * 2 + 1] <= 0) {
        return;
    }
    const uint32_t bid = idx / (C * N); // batch id
    const uint32_t cid = (idx / N) % C; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += bid * N * 3 + gid * 3;
    viewmats += bid * C * 16 + cid * 16;
    Ks += bid * C * 9 + cid * 9;

    conics += idx * 3;

    v_means2d += idx * 2;
    v_depths += idx;
    v_normals += idx * 3;
    v_conics += idx * 3;

    // vjp: compute the inverse of the 2d covariance
    mat2 covar2d_inv = mat2(conics[0], conics[1], conics[1], conics[2]);
    mat2 v_covar2d_inv =
        mat2(v_conics[0], v_conics[1] * .5f, v_conics[1] * .5f, v_conics[2]);
    mat2 v_covar2d(0.f);
    inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);
    vec3 v_normal = {v_normals[0], v_normals[1], v_normals[2]};

    if (v_compensations != nullptr) {
        // vjp: compensation term
        const float compensation = compensations[idx];
        const float v_compensation = v_compensations[idx];
        add_blur_vjp(
            eps2d, covar2d_inv, compensation, v_compensation, v_covar2d
        );
    }

    // transform Gaussian to camera space
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

    mat3 covar;
    vec4 quat;
    vec3 scale;
    if (covars != nullptr) {
        covars += bid * N * 6 + gid * 6;
        covar = mat3(
            covars[0],
            covars[1],
            covars[2], // 1st column
            covars[1],
            covars[3],
            covars[4], // 2nd column
            covars[2],
            covars[4],
            covars[5] // 3rd column
        );
    } else {
        // compute from quaternions and scales
        quat = glm::make_vec4(quats + bid * N * 4 + gid * 4);
        scale = glm::make_vec3(scales + bid * N * 3 + gid * 3);
        quat_scale_to_covar_preci(quat, scale, &covar, nullptr);
    }
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);
    mat3 covar_c;
    covarW2C(R, covar, covar_c);

    // vjp: perspective projection
    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    mat3 v_covar_c(0.f);
    vec3 v_mean_c(0.f);

    switch (camera_model) {
    case CameraModelType::PINHOLE: // perspective projection
        persp_proj_vjp(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            v_covar2d,
            glm::make_vec2(v_means2d),
            v_mean_c,
            v_covar_c
        );
        break;
    case CameraModelType::ORTHO: // orthographic projection
        ortho_proj_vjp(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            v_covar2d,
            glm::make_vec2(v_means2d),
            v_mean_c,
            v_covar_c
        );
        break;
    case CameraModelType::FISHEYE: // fisheye projection
        fisheye_proj_vjp(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            v_covar2d,
            glm::make_vec2(v_means2d),
            v_mean_c,
            v_covar_c
        );
        break;
    }

    // add contribution from v_depths
    v_mean_c.z += v_depths[0];

    // vjp: transform Gaussian covariance to camera space
    vec3 v_mean(0.f);
    mat3 v_covar(0.f);
    mat3 v_R(0.f);
    vec3 v_t(0.f);
    posW2C_VJP(R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean);
    covarW2C_VJP(R, covar, v_covar_c, v_R, v_covar);

    // vjp: normal gradients (only valid when projecting from quats/scales).
    // NOTE:
    // - The min-axis selection and the flip are non-differentiable.
    // - If `covars` are provided, `quats/scales` are nullptr and this path is skipped.
    vec4 v_quat_from_normal(0.f);
    if (quats != nullptr && scales != nullptr) {
        mat3 R_gauss = quat_to_rotmat(quat);

        float sx = scale.x, sy = scale.y, sz = scale.z;
        int min_idx = (sy < sx) ? 1 : 0;
        min_idx = (sz < ((min_idx == 0) ? sx : sy)) ? 2 : min_idx;

        vec3 normal_w = R_gauss[min_idx];
        vec3 normal_c_pre_flip = R * normal_w;
        float multiplier = glm::dot(-normal_c_pre_flip, mean_c) > 0 ? 1.f : -1.f;

        vec3 v_normal_c = v_normal * multiplier;
        vec3 v_normal_w = glm::transpose(R) * v_normal_c;

        // gradient w.r.t camera rotation R
        v_R += glm::outerProduct(v_normal_c, normal_w);

        // gradient w.r.t quaternion via the selected column of R_gauss
        mat3 v_R_gauss(0.f);
        v_R_gauss[min_idx] = v_normal_w;
        quat_to_rotmat_vjp(quat, v_R_gauss, v_quat_from_normal);
    }

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (v_means != nullptr) {
        warpSum(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_means += bid * N * 3 + gid * 3;
#pragma unroll
            for (uint32_t i = 0; i < 3; i++) {
                gpuAtomicAdd(v_means + i, v_mean[i]);
            }
        }
    }
    if (v_covars != nullptr) {
        // Output gradients w.r.t. the covariance matrix
        warpSum(v_covar, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_covars += bid * N * 6 + gid * 6;
            gpuAtomicAdd(v_covars, v_covar[0][0]);
            gpuAtomicAdd(v_covars + 1, v_covar[0][1] + v_covar[1][0]);
            gpuAtomicAdd(v_covars + 2, v_covar[0][2] + v_covar[2][0]);
            gpuAtomicAdd(v_covars + 3, v_covar[1][1]);
            gpuAtomicAdd(v_covars + 4, v_covar[1][2] + v_covar[2][1]);
            gpuAtomicAdd(v_covars + 5, v_covar[2][2]);
        }
    } else {
        // Directly output gradients w.r.t. the quaternion and scale
        mat3 rotmat = quat_to_rotmat(quat);
        vec4 v_quat(0.f);
        vec3 v_scale(0.f);
        quat_scale_to_covar_vjp(quat, scale, rotmat, v_covar, v_quat, v_scale);
        v_quat += v_quat_from_normal;
        warpSum(v_quat, warp_group_g);
        warpSum(v_scale, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_quats += bid * N * 4 + gid * 4;
            v_scales += bid * N * 3 + gid * 3;
            gpuAtomicAdd(v_quats, v_quat[0]);
            gpuAtomicAdd(v_quats + 1, v_quat[1]);
            gpuAtomicAdd(v_quats + 2, v_quat[2]);
            gpuAtomicAdd(v_quats + 3, v_quat[3]);
            gpuAtomicAdd(v_scales, v_scale[0]);
            gpuAtomicAdd(v_scales + 1, v_scale[1]);
            gpuAtomicAdd(v_scales + 2, v_scale[2]);
        }
    }
    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += bid * C * 16 + cid * 16;
#pragma unroll
            for (uint32_t i = 0; i < 3; i++) { // rows
#pragma unroll
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}

void launch_projection_ewa_3dgs_fused_bwd_kernel(
    // inputs
    // fwd inputs
    const at::Tensor means,                // [..., N, 3]
    const at::optional<at::Tensor> covars, // [..., N, 6] optional
    const at::optional<at::Tensor> quats,  // [..., N, 4] optional
    const at::optional<at::Tensor> scales, // [..., N, 3] optional
    const at::Tensor viewmats,             // [..., C, 4, 4]
    const at::Tensor Ks,                   // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor radii,                       // [..., C, N, 2]
    const at::Tensor conics,                      // [..., C, N, 3]
    const at::optional<at::Tensor> compensations, // [..., C, N] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [..., C, N, 2]
    const at::Tensor v_depths,                      // [..., C, N]
    const at::Tensor v_normals,                     // [..., C, N, 3]
    const at::Tensor v_conics,                      // [..., C, N, 3]
    const at::optional<at::Tensor> v_compensations, // [..., C, N] optional
    const bool viewmats_requires_grad,
    // outputs
    at::Tensor v_means,   // [..., N, 3]
    at::Tensor v_covars,  // [..., N, 3, 3]
    at::Tensor v_quats,   // [..., N, 4]
    at::Tensor v_scales,  // [..., N, 3]
    at::Tensor v_viewmats // [..., C, 4, 4]
) {
    uint32_t N = means.size(-2);    // number of gaussians
    uint32_t C = viewmats.size(-3); // number of cameras
    uint32_t B = means.numel() / (N * 3); // number of batches

    int64_t n_elements = B * C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "projection_ewa_3dgs_fused_bwd_kernel",
        [&]() {
            projection_ewa_3dgs_fused_bwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    B, 
                    C,
                    N,
                    means.data_ptr<scalar_t>(),
                    covars.has_value() ? covars.value().data_ptr<scalar_t>()
                                       : nullptr,
                    covars.has_value() ? nullptr
                                       : quats.value().data_ptr<scalar_t>(),
                    covars.has_value() ? nullptr
                                       : scales.value().data_ptr<scalar_t>(),
                    viewmats.data_ptr<scalar_t>(),
                    Ks.data_ptr<scalar_t>(),
                    image_width,
                    image_height,
                    eps2d,
                    camera_model,
                    radii.data_ptr<int32_t>(),
                    conics.data_ptr<scalar_t>(),
                    compensations.has_value()
                        ? compensations.value().data_ptr<scalar_t>()
                        : nullptr,
                    v_means2d.data_ptr<scalar_t>(),
                    v_depths.data_ptr<scalar_t>(),
                    v_normals.data_ptr<scalar_t>(),
                    v_conics.data_ptr<scalar_t>(),
                    v_compensations.has_value()
                        ? v_compensations.value().data_ptr<scalar_t>()
                        : nullptr,
                    v_means.data_ptr<scalar_t>(),
                    covars.has_value() ? v_covars.data_ptr<scalar_t>()
                                       : nullptr,
                    covars.has_value() ? nullptr : v_quats.data_ptr<scalar_t>(),
                    covars.has_value() ? nullptr
                                       : v_scales.data_ptr<scalar_t>(),
                    viewmats_requires_grad ? v_viewmats.data_ptr<scalar_t>()
                                           : nullptr
                );
        }
    );
}

} // namespace gsplat
