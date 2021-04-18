#include "img_proc/cuda/compute_vertex.h"
#include "img_proc/cuda/cuda_snippets.hpp"

namespace device {

/**
 * @brief compute Vertex kernel using texture.
 * V_x = (u - cx) * V_z / fx
 * V_y = (v - cy) * V_z / fy
 * if (vertex_z == 0), then vertex_x = vertex_y = 0.
 *
 * @param[in] cols image cols = image width
 * @param[in] rows image rows = image height
 * @param[in] cam_intr_inv inverse intrinsic: 1/fx, 1/fy, cx, cy
 * @param[in] depth_texture
 * @param[out] vertex_surface vertex unit is millimeter(mm)
 */
__global__ void ComputeVertexKernelTexture(const unsigned cols,
                                           const unsigned rows,
                                           const CamIntrInv cam_intr_inv,
                                           cudaTextureObject_t depth_texture,
                                           cudaSurfaceObject_t vertex_surface) {
  const unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= cols || y >= rows)
    return;

  const float vertex_z = static_cast<float>(tex2D<ushort>(depth_texture, x, y));

  const float vertex_x = (x - cam_intr_inv.cx) * vertex_z * cam_intr_inv.fx_inv;
  const float vertex_y = (y - cam_intr_inv.cy) * vertex_z * cam_intr_inv.fy_inv;

  float4 vertex = make_float4(vertex_x, vertex_y, vertex_z, 1);
  surf2Dwrite(vertex, vertex_surface, x * sizeof(float4), y);
}

/**
 * @brief compute Vertex kernel using texture.
 * V_x = (u - cx) * V_z / fx
 * V_y = (v - cy) * V_z / fy
 * if (vertex_z == 0), then vertex_x = vertex_y = 0.
 *
 * @tparam T depth date type, it's expected to be either ushort or float
 * @param[in] depth depth image
 * @param[out] vertex vertex map
 * @param[in] cam_intr_inv inverse camera intrinsic: 1/fx, 1/fy, cx, cy
 */
template <typename T>
__global__ void ComputeVertexKernel(const PtrStepSz<T> depth,
                                    PtrStepSz<float> vertex,
                                    const CamIntrInv cam_intr_inv) {
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= depth.cols || y >= depth.rows)
    return;

  // Note: Even if (vertex_z == 0); then vertex_x = vertex_y = 0.
  const float vertex_z = static_cast<float>(depth.ptr(y)[x]);

  // V_x = (u - cx) * V_z / fx
  // V_y = (v - cy) * V_z / fy
  const float vertex_x = (x - cam_intr_inv.cx) * vertex_z * cam_intr_inv.fx_inv;
  const float vertex_y = (y - cam_intr_inv.cy) * vertex_z * cam_intr_inv.fy_inv;

  vertex.ptr(y)[x] = vertex_x;
  vertex.ptr(y + depth.rows)[x] = vertex_y;
  vertex.ptr(y + depth.rows * 2)[x] = vertex_z;
}

} // namespace device

void ComputeVertex(CudaTextureSurface2D<ushort>::Ptr depth,
                   CudaTextureSurface2D<float4>::Ptr vertex,
                   const CamIntrInv &cam_intr_inv, cudaStream_t stream) {
  dim3 blk(16, 16);
  dim3 grid(DivideUp(depth->cols(), blk.x), DivideUp(depth->rows(), blk.y));

  device::ComputeVertexKernelTexture<<<grid, blk, 0, stream>>>(
      depth->cols(), depth->rows(), cam_intr_inv, depth->texture(),
      vertex->surface());
  CudaSafeCall(cudaStreamSynchronize(stream));
  CudaSafeCall(cudaGetLastError());
}

void ComputeVertex(const DeviceArray2D<ushort> depth,
                   DeviceArray2D<float> vertex, const CamIntrInv &cam_intr_inv,
                   cudaStream_t stream) {
  dim3 blk(16, 16);
  dim3 grid(DivideUp(depth.cols(), blk.x), DivideUp(depth.rows(), blk.y));

  device::ComputeVertexKernel<<<grid, blk, 0, stream>>>(
      PtrStepSz<ushort>(depth), vertex, cam_intr_inv);

  CudaSafeCall(cudaStreamSynchronize(stream));
  CudaSafeCall(cudaGetLastError());
}

void ComputeVertex(const DeviceArray2D<float> depth,
                   DeviceArray2D<float> vertex, const CamIntrInv &cam_intr_inv,
                   cudaStream_t stream) {
  dim3 blk(16, 16);
  dim3 grid(DivideUp(depth.cols(), blk.x), DivideUp(depth.rows(), blk.y));

  device::ComputeVertexKernel<<<grid, blk, 0, stream>>>(PtrStepSz<float>(depth),
                                                        vertex, cam_intr_inv);

  CudaSafeCall(cudaStreamSynchronize(stream));
  CudaSafeCall(cudaGetLastError());
}