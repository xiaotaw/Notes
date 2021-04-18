/**
 * @file pagelocked_memory.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date
 *   2020-06-29 init version
 *   2021-04-11 Separate from cuda_snippets.hpp and form an independent module
 * @copyright Copyright (c) 2021
 */
#pragma once
#include "common/disable_copy_assign_move.h"
#include <cstring> // for memcpy
#include <cuda_runtime_api.h>
#include <memory>

/**
 * @brief Pagelocked memory or pinned memory, which could speedup data transfer
 * between device and host.
 */
class PagelockedMemory {
public:
  using Ptr = std::shared_ptr<PagelockedMemory>;
  DISABLE_COPY_ASSIGN_MOVE(PagelockedMemory);

  PagelockedMemory(const size_t &size) {
    CudaSafeCall(cudaMallocHost(&data_, size));
    size_ = size;
  }

  ~PagelockedMemory() {
    CudaSafeCall(cudaFreeHost(data_));
    size_ = 0;
  }

  /**
   * @brief copy data from another address in host.
   * @param[in] src pointer to buffer to copy from
   */
  inline void CopyFrom(const void *src) { std::memcpy(data_, src, size_); }

  /**
   * @brief copy data to another address in host
   * @param[out] dst pointer to buffer to copy to
   */
  inline void CopyTo(void *dst) const { std::memcpy(dst, data_, size_); }

  inline void Clear() { std::memset(data_, 0, size_); }

  // private access
  inline void *data() const { return data_; }
  inline size_t size() const { return size_; }

private:
  void *data_;
  size_t size_;
};