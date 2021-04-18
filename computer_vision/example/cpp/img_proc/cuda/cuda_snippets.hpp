/**
 * some snippets, often defined in MACROs
 * @author: xiaotaw
 * @email:
 * @date: 2020/06/29 22:33
 */
#pragma once
#include <cuda_runtime_api.h>
#include <iostream>

#define CudaSafeCall(expr) __cudaSafeCall(expr, __FILE__, __LINE__, __func__)

static inline void __cudaSafeCall(cudaError_t err, const char *file,
                                  const int line, const char *func = "") {
  if (cudaSuccess != err) {
    std::cout << "Error: " << file << ":" << line << " in " << func
              << std::endl;
  }
}

static inline unsigned int DivideUp(unsigned int dividend,
                                    unsigned int divisor) {
  return (dividend + divisor - 1) / divisor;
}

__device__ __host__ inline bool IsZeroVertex(const float &x, const float &y,
                                             const float &z) {
  return (std::abs(x) <= 1e-5) || (std::abs(y) <= 1e-5) ||
         (std::abs(z) <= 1e-5);
}

__device__ __host__ inline bool IsZeroVertex(const float3 &v) {
  return (std::abs(v.x) <= 1e-5) || (std::abs(v.y) <= 1e-5) ||
         (std::abs(v.z) <= 1e-5);
}