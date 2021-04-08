/**
 * cuda_texture_surface.cu
 * @author: xiaotaw
 * @email:
 * @date: 2020/07/09 19:04
 */
#include "img_proc/cuda/cuda_texture_surface.h"

namespace device {
// nearest interpolation
template <typename T>
__global__ void
ResizeDownNearestKernel(cudaTextureObject_t src_texture,
                        const unsigned src_cols, const unsigned src_rows,
                        const unsigned dst_cols, const unsigned dst_rows,
                        const float col_factor_inv, const float row_factor_inv,
                        cudaSurfaceObject_t dst_surface) {

  unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= dst_cols || y >= dst_rows)
    return;

  unsigned src_x = __float2uint_rn(col_factor_inv * x);
  unsigned src_y = __float2uint_rn(row_factor_inv * y);

  const T val = tex2D<T>(src_texture, src_x, src_y);

  surf2Dwrite(val, dst_surface, x * sizeof(T), y);
}

// ******* DO NOT USE: Not Finished Yet ********
// template<typename T>
// __global__ void ResizeDownInterAreaKernel(cudaTextureObject_t src_texture,
//     const unsigned src_cols, const unsigned src_rows,
//     const unsigned dst_cols, const unsigned dst_rows,
//     const float col_factor_inv, const float row_factor_inv,
//     cudaSurfaceObject_t dst_surface){

//     unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
//     unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
//     if (x >= dst_cols || y >= dst_rows) return;

//     T sum = 0;
//     for(int i=0; i<col_factor;){
//         for(int j=0; j<row_factor;){
//             unsigned src_x = __float2uint_rn(col_factor_inv * x + i);
//             unsigned src_y = __float2uint_rn(row_factor_inv * y + j);
//             sum += tex2D<T>(src_texture, src_x, src_y);
//         }
//     }
//     const T val = sum / (col_factor * row_factor);
//     surf2Dwrite(val, dst_surface, x * sizeof(T), y);
// }

} // namespace device

template <typename T>
void CudaTextureSurface2D<T>::ResizeDownNearest(CudaTextureSurface2D<T> &dst,
                                                cudaStream_t stream) {
  float col_factor_inv = static_cast<float>(cols_ / dst.cols_);
  float row_factor_inv = static_cast<float>(rows_ / dst.rows_);
  dim3 blk(16, 16);
  dim3 grid(DivideUp(dst.cols_, blk.x), DivideUp(dst.rows_, blk.y));
  device::ResizeDownNearestKernel<T>
      <<<grid, blk, 0, stream>>>(texture_, cols_, rows_, dst.cols_, dst.rows_,
                                 col_factor_inv, row_factor_inv, dst.surface_);
  CudaSafeCall(cudaStreamSynchronize(stream));
  CudaSafeCall(cudaGetLastError());
}

template <typename T>
void CudaTextureSurface2D<T>::ResizeDownInterArea(CudaTextureSurface2D<T> &dst,
                                                  cudaStream_t stream) {
  // float col_factor_inv = static_cast<float>(cols_ / dst.cols_);
  // float row_factor_inv = static_cast<float>(rows_ / dst.rows_);
  // dim3 blk(16, 16);
  // dim3 grid(DivideUp(dst.cols_, blk.x), DivideUp(dst.rows_, blk.y));
  // device::ResizeDownInterAreaKernel<T><<<grid, blk, 0, stream>>>(
  //     texture_,
  //     cols_, rows_,
  //     dst.cols_, dst.rows_,
  //     col_factor_inv, row_factor_inv,
  //     dst.surface_
  // );
  // CudaSafeCall(cudaStreamSynchronize(stream));
  // CudaSafeCall(cudaGetLastError());
}

template <typename T>
void CudaTextureSurface2D<T>::ResizeDown(CudaTextureSurface2D<T> &dst,
                                         cudaStream_t stream,
                                         InterpolationMode mode) {
  if (mode == kNearest) {
    CudaTextureSurface2D<T>::ResizeDownNearest(dst, stream);
  } else if (mode == kInterArea) {
    // ResizeDownInterArea(dst, stream);
    std::cout << "Unsupported Interpolation Mode: " << mode << std::endl;
  } else {
    std::cout << "Unsupported Interpolation Mode: " << mode << std::endl;
  }
}

template <>
void CudaTextureSurface2D<ushort>::ResizeDown(CudaTextureSurface2D<ushort> &dst,
                                              cudaStream_t stream,
                                              InterpolationMode mode) {
  if (mode == kNearest) {
    CudaTextureSurface2D<ushort>::ResizeDownNearest(dst, stream);
  } else if (mode == kInterArea) {
    // ResizeDownInterArea(dst, stream);
    std::cout << "Unsupported Interpolation Mode: " << mode << std::endl;
  } else {
    std::cout << "Unsupported Interpolation Mode: " << mode << std::endl;
  }
}

template <>
void CudaTextureSurface2D<uchar4>::ResizeDown(CudaTextureSurface2D<uchar4> &dst,
                                              cudaStream_t stream,
                                              InterpolationMode mode) {
  if (mode == kNearest) {
    CudaTextureSurface2D<uchar4>::ResizeDownNearest(dst, stream);
  } else if (mode == kInterArea) {
    // ResizeDownInterArea(dst, stream);
    std::cout << "Unsupported Interpolation Mode: " << mode << std::endl;
  } else {
    std::cout << "Unsupported Interpolation Mode: " << mode << std::endl;
  }
}
