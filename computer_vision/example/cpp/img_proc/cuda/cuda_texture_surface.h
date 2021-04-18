/**
 * cuda texture surface
 * @author: xiaotaw
 * @email:
 * @date: 2020/06/29 22:35
 */
#pragma once
#include "common/disable_copy_assign_move.h"
#include "img_proc/cuda/cuda_snippets.hpp"
#include <channel_descriptor.h> // for cudaCreateChannelDesc<T>()
#include <cstring>              // for memcpy and memset using in template
#include <memory>

// (xt) TODO: move enum outside of template class, because it's independent of
// the template bilinear is not supported as filterMode only support
// cudaFilterModePoint, when data type is not float kInterArea is not supported
// yet.
enum InterpolationMode {
  kNearest = 0, // cv::INTER_NEAREST
  // kBilinear = 1,   // cv::INTER_LINEAR not supported now
  kInterArea = 3 // cv::INTER_AREA
};

template <typename T> class CudaTextureSurface2D {
public:
  using Ptr = std::shared_ptr<CudaTextureSurface2D<T>>;

  DISABLE_COPY_ASSIGN_MOVE(CudaTextureSurface2D);

  // ctor
  CudaTextureSurface2D() = default;

  // ctor
  CudaTextureSurface2D(const unsigned rows, const unsigned cols)
      : rows_(rows), cols_(cols) {
    m_channel_desc_ = cudaCreateChannelDesc<T>();
    // allocate cuda array
    CudaSafeCall(cudaMallocArray(&d_array_, &m_channel_desc_, cols_, rows_));
    // create resource desc
    memset(&m_resource_desc_, 0, sizeof(m_resource_desc_));
    m_resource_desc_.resType = cudaResourceTypeArray;
    m_resource_desc_.res.array.array = d_array_;
    // create surface
    CudaSafeCall(cudaCreateSurfaceObject(&surface_, &m_resource_desc_));
    // create texture desc
    memset(&m_texture_desc_, 0, sizeof(m_texture_desc_));
    m_texture_desc_.addressMode[0] = m_texture_desc_.addressMode[1] =
        m_texture_desc_.addressMode[2] = cudaAddressModeBorder;
    m_texture_desc_.filterMode = cudaFilterModePoint;
    m_texture_desc_.readMode = cudaReadModeElementType;
    m_texture_desc_.normalizedCoords = 0;
    // create texture
    CudaSafeCall(cudaCreateTextureObject(&texture_, &m_resource_desc_,
                                         &m_texture_desc_, 0));
  }

  // dtor
  ~CudaTextureSurface2D() {
    CudaSafeCall(cudaDestroyTextureObject(texture_));
    CudaSafeCall(cudaDestroySurfaceObject(surface_));
    CudaSafeCall(cudaFreeArray(d_array_));
  }

  // upload from host to device
  void Upload(const void *data_h, cudaStream_t stream) {
    CudaSafeCall(cudaMemcpyToArrayAsync(d_array_, 0, 0, data_h,
                                        sizeof(T) * cols_ * rows_,
                                        cudaMemcpyHostToDevice, stream));
  }

  // download from device to host
  void Download(void *data_h, cudaStream_t stream) {
    CudaSafeCall(cudaMemcpyFromArrayAsync(data_h, d_array_, 0, 0,
                                          sizeof(T) * cols_ * rows_,
                                          cudaMemcpyDeviceToHost, stream));
  }

  // for build pyramid
  void ResizeDown(CudaTextureSurface2D &dst, cudaStream_t stream,
                  InterpolationMode mode);
  void ResizeDownNearest(CudaTextureSurface2D &dst, cudaStream_t stream);
  void ResizeDownInterArea(CudaTextureSurface2D &dst, cudaStream_t stream);

  // private member access
  cudaTextureObject_t texture() const { return texture_; }
  cudaTextureObject_t surface() const { return surface_; }
  cudaArray_t d_array() const { return d_array_; }
  unsigned rows() const { return rows_; }
  unsigned cols() const { return cols_; }

private:
  cudaChannelFormatDesc m_channel_desc_;
  cudaResourceDesc m_resource_desc_;
  cudaTextureDesc m_texture_desc_;

  cudaTextureObject_t texture_;
  cudaSurfaceObject_t surface_;
  cudaArray_t d_array_;
  unsigned rows_;
  unsigned cols_;
};
