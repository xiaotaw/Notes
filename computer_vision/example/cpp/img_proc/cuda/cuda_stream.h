/**
 * @file cuda_stream.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2020-06-29 init version
 *       2021-04-17 Separate from cuda_snippets.hpp and form an independent
 * module
 * @copyright Copyright (c) 2021
 */
#pragma once
#include "common/disable_copy_assign_move.h"
#include "img_proc/cuda/cuda_snippets.hpp"
#include <cuda_runtime_api.h>

/**
 * @brief cudaStream_t with simple wrap of automatical garbage collection(gc)
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