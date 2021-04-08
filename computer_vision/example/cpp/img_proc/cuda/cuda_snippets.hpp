/**
 * some snippets, often defined in MACROs
 * @author: xiaotaw
 * @email:
 * @date: 2020/06/29 22:33
 */
#pragma once
#include "common/disable_copy_assign_move.h"
#include <cstring> // for memcpy
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>

#define CudaSafeCall(expr) __cudaSafeCall(expr, __FILE__, __LINE__, __func__)

static inline void __cudaSafeCall(cudaError_t err, const char *file,
                                  const int line, const char *func = "") {
  if (cudaSuccess != err) {
    std::cout << "Error: " << file << ":" << line << " in " << func
              << std::endl;
  }
}

static inline unsigned DivideUp(unsigned dividend, unsigned divisor) {
  return (dividend + divisor - 1) / divisor;
}

/**
 * simple wrap for automatical garbage collection(gc)
 */
class CudaStream {
public:
  cudaStream_t stream_;
  CudaStream() { CudaSafeCall(cudaStreamCreate(&stream_)); }
  ~CudaStream() { CudaSafeCall(cudaStreamDestroy(stream_)); }

  DISABLE_COPY_ASSIGN_MOVE(CudaStream);

  inline void Synchronize() { CudaSafeCall(cudaStreamSynchronize(stream_)); }

  operator cudaStream_t() { return stream_; }
};

/**
 * Desc:
 *  Data transfer between device and host via pagelock memory
 * Usage:
 *  Host2Device: memory      -----HostCopyFrom-----> PagelockMemory
 * --UploadToDevice--> cudaArray_t Device2Host: cudaArray_t
 * --DownloadFromDevice--> PagelockMemory ----HostCopyTo----> memory
 */
class PagelockMemory {
public:
  void *data_;
  size_t size_;
  PagelockMemory(size_t size) {
    CudaSafeCall(cudaMallocHost(&data_, size));
    size_ = size;
  }
  ~PagelockMemory() {
    CudaSafeCall(cudaFreeHost(data_));
    size_ = 0;
  }

  using Ptr = std::shared_ptr<PagelockMemory>;

  DISABLE_COPY_ASSIGN_MOVE(PagelockMemory);

  inline void UploadToDevice(cudaArray_t d_array, cudaStream_t stream) const {
    CudaSafeCall(cudaMemcpyToArrayAsync(d_array, 0, 0, data_, size_,
                                        cudaMemcpyHostToDevice, stream));
  }

  inline void DownloadFromDevice(cudaArray_t d_array, cudaStream_t stream) {

    CudaSafeCall(cudaMemcpyFromArrayAsync(data_, d_array, 0, 0, size_,
                                          cudaMemcpyDeviceToHost, stream));
  }

  inline void HostCopyFrom(void *src) { std::memcpy(data_, src, size_); }

  inline void HostCopyTo(void *dst) { std::memcpy(dst, data_, size_); }

  inline void Clear() { std::memset(data_, 0, size_); }
};