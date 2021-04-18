/**
 * @file bilater_filter.cu
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-04-09 file create, add todo
 *       2021-04-11 complete todo, initial version
 * @copyright Copyright (c) 2021
 */
#include "img_proc/cuda/bilateral_filter.h"

namespace device {

/**
 * @brief compute gaussian 1D weight for bilateral filter
 * @tparam T
 * @param[in] x
 * @param[in] sigma Standard Deviation
 * @return float
 */
template <typename T> __device__ float Gauss1D(const T &x, const float &sigma) {
  return exp(-1.0f * x * x / (2.0f * sigma * sigma));
}

/**
 * @brief compute gaussian 2D weight for bilateral filter,
 *     suppose sigma = sigma11 = sigma22, sigma12 = sigma21 = 0, mu1 = mu2 = 0
 * @tparam T
 * @param[in] x1
 * @param[in] x2
 * @param[in] sigma Standard Deviation
 * @return float
 */
template <typename T>
__device__ float Gauss2D(const T &x1, const T &x2, const float &sigma) {
  return exp(-1.0f * (x1 * x1 + x2 * x2) / (2.0f * sigma * sigma));
}

__global__ void BilateralFilterKernelTexture(cudaTextureObject_t depth_texture,
                                             cudaSurfaceObject_t depth_surface,
                                             const int &cols, const int &rows,
                                             const float &sigma_d,
                                             const float &sigma_r) {
  const int x = threadIdx.x + blockDim.x * blockIdx.x;
  const int y = threadIdx.y + blockDim.y * blockIdx.y;
  if (x >= cols || y >= rows)
    return;

  const float p = static_cast<float>(tex2D<ushort>(depth_texture, x, y));
  float sum = 0.0f, sum_w = 0.0f;

  // radius
  const int r = (int)ceil(2 * sigma_d);
  for (int i = x - r; i <= x + r; ++i) {
    for (int j = y - r; j <= y + r; ++j) {
      if (i >= 0 && i < cols && j >= 0 && j < rows) {
        const float q = static_cast<float>(tex2D<ushort>(depth_texture, i, j));
        float w = Gauss2D<int>(x - i, y - j, sigma_d) *
                  Gauss1D<float>(p - q, sigma_r);
        sum += w * q;
        sum_w += w;
      }
    }
  }

  float center_filtered = sum / sum_w;
  surf2Dwrite(center_filtered, depth_surface, x * sizeof(float), y);
}

__global__ void BilateralFilterKernel(const PtrStepSz<ushort> depth_in,
                                      PtrStepSz<float> depth_out,
                                      const float &sigma_d,
                                      const float &sigma_r) {
  const int x = threadIdx.x + blockDim.x * blockIdx.x;
  const int y = threadIdx.y + blockDim.y * blockIdx.y;
  if (x >= depth_in.cols || y >= depth_in.rows)
    return;

  const float p = static_cast<float>(depth_in.ptr(y)[x]);
  float sum = 0.0f, sum_w = 0.0f;

  const int r = (int)ceil(2 * sigma_d);
  for (int i = x - r; i <= x + r; ++i) {
    for (int j = y - r; j <= y + r; ++j) {
      if (i >= 0 && i < depth_in.cols && j >= 0 && j < depth_in.rows) {
        const float q = static_cast<float>(depth_in.ptr(j)[i]);
        float w = Gauss2D<int>(x - i, y - j, sigma_d) *
                  Gauss1D<float>(p - q, sigma_r);
        sum += w * q;
        sum_w += w;
      }
    }
  }

  depth_out.ptr(y)[x] = sum / sum_w;
}

} // namespace device

void BilateralFilter(CudaTextureSurface2D<ushort> depth_in,
                     CudaTextureSurface2D<float> depth_out,
                     const float &sigma_d, const float &sigma_r,
                     cudaStream_t stream) {
  dim3 blk(16, 16);
  dim3 grid(DivideUp(depth_in.cols(), blk.x), DivideUp(depth_in.rows(), blk.y));
  device::BilateralFilterKernelTexture<<<grid, blk, 0, stream>>>(
      depth_in.texture(), depth_out.surface(), depth_in.cols(), depth_in.rows(),
      sigma_d, sigma_r);

  CudaSafeCall(cudaStreamSynchronize(stream));
  CudaSafeCall(cudaGetLastError());
}

void BilateralFilter(const DeviceArray2D<ushort> depth_in,
                     DeviceArray2D<float> depth_out, const float &sigma_d,
                     const float &sigma_r, cudaStream_t stream) {
  dim3 blk(16, 16);
  dim3 grid(DivideUp(depth_in.cols(), blk.x), DivideUp(depth_in.rows(), blk.y));
  device::BilateralFilterKernel<<<grid, blk, 0, stream>>>(depth_in, depth_out,
                                                          sigma_d, sigma_r);

  CudaSafeCall(cudaStreamSynchronize(stream));
  CudaSafeCall(cudaGetLastError());
}