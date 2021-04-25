/**
 * @file compute_normal.cu
 * @author xiaotaw (you@domain.com)
 * @brief compute normal map, reference: https://github.com/weigao95/surfelwarp
 * @version 0.1
 * @date 2021-04-18
 * @copyright Copyright (c) 2021
 */
#include "img_proc/cuda/compute_normal.h"
#include "img_proc/cuda/cuda_snippets.hpp"
#include "img_proc/cuda/utils/eigen.hpp"

namespace device {

__device__ float3 ReadFloat3(const PtrStepSz<float> map, const int &u,
                             const int &v) {
  const float x = map.ptr(v, 0)[u];
  const float y = map.ptr(v, 1)[u];
  const float z = map.ptr(v, 2)[u];
  return make_float3(x, y, z);
}

__device__ void WriteFloat4(PtrStepSz<float> map, const float4 &val,
                            const int &u, const int &v) {
  map.ptr(v, 0)[u] = val.x;
  map.ptr(v, 1)[u] = val.y;
  map.ptr(v, 2)[u] = val.z;
  map.ptr(v, 3)[u] = val.w;
}

/**
 * @brief 计算法向量的核函数
 * @param vertex_map 顶点
 * @param normal_map 法向量
 * @todo 1. try to use shared memory to speed up,
 *       2. try fast mean & std caculation， in a iterative way
 */
__global__ void ComputeNormalKernel(const PtrStepSz<float> vertex_map,
                                    PtrStepSz<float> normal_map) {
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int cols = vertex_map.cols;
  const int rows = vertex_map.rows;
  if (x >= cols || y >= rows) {
    return;
  }

  // 矩形范围内，均为临近点
  const int neighbor_radius = 3;
  const int neighbor_points_number =
      (2 * neighbor_radius + 1) * (2 * neighbor_radius + 1);

  float4 normal = {0, 0, 0, 0};
  float3 vertex = ReadFloat3(vertex_map, x, y);

  if (!IsZeroVertex(vertex)) {
    // 统计周围点的信息
    int cnt = 0;
    float3 centroid = make_float3(0, 0, 0);
    for (int i = -neighbor_radius; i <= neighbor_radius; ++i) {
      for (int j = -neighbor_radius; j <= neighbor_radius; ++j) {
        if (x + i < 0 || x + i >= cols || y + j < 0 || y + j >= rows) {
          continue;
        }
        float3 v = ReadFloat3(vertex_map, x + i, y + j);
        if (!IsZeroVertex(v)) {
          centroid += v;
          cnt++;
        }
      }
    }

    // 点数过半，才计算法向量
    if (cnt * 2 > neighbor_points_number) {
      centroid *= (1.0f / cnt);

      float cov_half[6] = {0};
      for (int i = -neighbor_radius; i <= neighbor_radius; ++i) {
        for (int j = -neighbor_radius; j <= neighbor_radius; ++j) {
          if (x + i < 0 || x + i >= cols || y + j < 0 || y + j >= rows) {
            continue;
          }
          float3 v = ReadFloat3(vertex_map, x + i, y + j);
          float3 d = v - centroid;
          cov_half[0] += d.x * d.x;
          cov_half[1] += d.x * d.y;
          cov_half[2] += d.x * d.z;
          cov_half[3] += d.y * d.y;
          cov_half[4] += d.y * d.z;
          cov_half[5] += d.z * d.z;
        }
      }

      // 特征值最小的特征向量即为法向量
      Eigen33::Mat33 tmp, vec_tmp, evecs;
      float3 evals;
      Eigen33 eigen33(cov_half);
      eigen33.compute(tmp, vec_tmp, evecs, evals);

      // 特征值evals中最小的为evals[0]，对应的特征向量为evecs[0]
      normal.x = evecs[0].x;
      normal.y = evecs[0].y;
      normal.z = evecs[0].z;

      // 法向量朝向纠正，使之朝向相机
      if (dot(vertex, make_float3(normal.x, normal.y, normal.z)) > 0) {
        normal *= -1.0;
      }

      // curvature surface change
      float evals_sum = evals.x + evals.y + evals.z;
      normal.w = (evals_sum == 0) ? 0 : std::fabs(evals.x / evals_sum);
    }
  }
  WriteFloat4(normal_map, normal, x, y);
}

} // namespace device

void ComputeNormal(const DeviceArray3D<float> vertex_map,
                   DeviceArray3D<float> normal_map, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid(DivideUp(vertex_map.cols(), block.x),
            DivideUp(vertex_map.rows(), block.y));
  device::ComputeNormalKernel<<<grid, block, 0, stream>>>(vertex_map,
                                                          normal_map);
  CudaSafeCall(cudaStreamSynchronize(stream));
  CudaSafeCall(cudaGetLastError());
}