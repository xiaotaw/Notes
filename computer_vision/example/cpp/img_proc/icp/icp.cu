/**
 * @file icp.cu
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-04-25 initial version
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "icp.h"
#include "img_proc/cuda/cuda_snippets.hpp"

namespace icp {

namespace rigid_icp {

namespace device {

__inline__ __device__ EdaVector3f ReadEdaVector3f(const PtrStepSz<float> &map,
                                                  const int &u, const int &v) {
  return EdaVector3f(map.ptr(v, 0)[u], map.ptr(v, 1)[u], map.ptr(v, 2)[u]);
}

__inline__ __device__ bool IsZeroVertex(EdaVector3f v) {
  return (std::abs(v.x()) <= 1e-5) || (std::abs(v.y()) <= 1e-5) ||
         (std::abs(v.z()) <= 1e-5);
}

__inline__ __device__ void
ComposeGnState(EdaGnState &gn_state, const float (&J)[6], const float &res) {
  // clang-format off
  // lower triangle of JtJ
  gn_state(0)  = J[0] * J[0]; 
  gn_state(1)  = J[0] * J[1]; gn_state(2)  = J[1] * J[1];
  gn_state(3)  = J[0] * J[2]; gn_state(4)  = J[1] * J[2]; gn_state(5)  = J[2] * J[2];
  gn_state(6)  = J[0] * J[3]; gn_state(7)  = J[1] * J[3]; gn_state(8)  = J[2] * J[3]; gn_state(9) = J[3] * J[3];
  gn_state(10) = J[0] * J[4]; gn_state(11) = J[1] * J[4]; gn_state(12) = J[2] * J[4]; gn_state(13) = J[3] * J[4]; gn_state(14) = J[4] * J[4];
  gn_state(15) = J[0] * J[5]; gn_state(16) = J[1] * J[5]; gn_state(17) = J[2] * J[5]; gn_state(18) = J[3] * J[5]; gn_state(19) = J[4] * J[5]; gn_state(20) = J[5] * J[5];
  // J*residual
  gn_state(21) = J[0] * res;  gn_state(22) = J[1] * res;  gn_state(23) = J[2] * res;  gn_state(24) = J[3] * res;  gn_state(25) = J[4] * res;  gn_state(26) = J[5] * res;
  // clang-format on

  // src vertex and tgt vertex is paired;
  gn_state(27) = 1;
  // the residual
  gn_state(28) = res;
}

/**
 * @brief warp reduce sum
 * @param[in out] gn_state
 * @note https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
 */
__device__ void WarpReduceSum(EdaGnState &gn_state) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < kGnStateSize; i++) {
      gn_state[i] += __shfl_down_sync(0xFFFFFFFF, gn_state[i], offset);
    }
  }
}

/**
 * @brief 2D Block Reduce sum
 * @param[in out] gn_state
 *
 * @note Block reduce sum的计算过程分为两步： 1. block中的每一个warp中进行reduce
 * sum，结果在该warp中第一个线程（lane_id为零）中。 2. 将每个warp执行reduce
 * sum后的结果，经过shared memory转移至同一个warp中，执行第二次warp reduce
 * sum。
 *
 * Block中最大线程数目为1024，一个warp的大小warpSize=32， 因此Block中最多有
 * 1024/32 = 32 个warp。 因此第二次warp reduce sum只需一个warp即可。
 *
 * Block中每一个thread有一个id，计算方式为：
 * The index of a thread and its thread ID relate to each other in a
 * straightforward way: For a one-dimensional block, they are the same; for a
 * two-dimensional block of size (Dx, Dy),the thread ID of a thread of index (x,
 * y) is (x + y Dx); for a three-dimensional block of size (Dx, Dy, Dz), the
 * thread ID of a thread of index (x, y, z) is (x + y Dx + z Dx Dy).
 * （来源 https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html 中的
 * 2.2 thread-hierarchy章节）
 */
__device__ void BlockReduceSum(EdaGnState &gn_state) {

  static __shared__ unsigned char shared_mem[32 * sizeof(EdaGnState)];
  EdaGnState(&shared)[32] = reinterpret_cast<EdaGnState(&)[32]>(shared_mem);

  int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
  int lane_id = thread_id % warpSize;
  int warp_id = thread_id / warpSize;

  WarpReduceSum(gn_state);

  if (lane_id == 0) {
    shared[warp_id] = gn_state;
  }

  __syncthreads();

  gn_state = (thread_id < blockDim.x * blockDim.y / warpSize)
                 ? shared[lane_id]
                 : EdaGnState::Zero();

  if (warp_id == 0) {
    WarpReduceSum(gn_state);
  }
}

__global__ void
SingleBlockReduceSum(const PtrStepSz<EdaGnState> gn_state_reduce_tmp,
                     PtrStepSz<EdaGnState> d_gn_state_sum) {
  const int u = threadIdx.x;
  const int v = threadIdx.y;
  EdaGnState val = gn_state_reduce_tmp.ptr(v)[u];

  BlockReduceSum(val);

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    d_gn_state_sum.ptr(0)[0] = val;
  }
}

__global__ void IcpAlignStepKernel(
    const PtrStepSz<float> vmap_src, const PtrStepSz<float> nmap_src,
    const PtrStepSz<float> vmap_tgt, const PtrStepSz<float> nmap_tgt,
    const CamIntr cam_intr, const EdaMatrix3f rot, const EdaVector3f trans,
    PtrStepSz<EdaGnState> gn_state_reduce_tmp) {
  const int u = threadIdx.x + blockDim.x * blockIdx.x;
  const int v = threadIdx.y + blockDim.y * blockIdx.y;

  EdaGnState gn_state = EdaGnState::Zero();

  if (u < vmap_src.cols && v < vmap_src.rows) {
    const EdaVector3f v_src = ReadEdaVector3f(vmap_src, u, v);
    const EdaVector3f n_src = ReadEdaVector3f(nmap_src, u, v);

    if (!IsZeroVertex(v_src)) {
      const EdaVector3f vp = rot * v_src + trans;
      int u_proj = __float2int_rn(vp(0) / vp(2) * cam_intr.fx + cam_intr.cx);
      int v_proj = __float2int_rn(vp(1) / vp(2) * cam_intr.fy + cam_intr.cy);

      if (u_proj >= 0 && u_proj < vmap_tgt.cols && v_proj > 0 &&
          v_proj < vmap_tgt.rows) {
        EdaVector3f v_tgt = ReadEdaVector3f(vmap_tgt, u_proj, v_proj);
        EdaVector3f n_tgt = ReadEdaVector3f(nmap_tgt, u_proj, v_proj);

        if (!IsZeroVertex(v_tgt)) {
          float J[6] = {0};
          *((EdaVector3f *)(&J[0])) = n_tgt;
          *((EdaVector3f *)(&J[3])) = vp.cross(n_tgt);
          float residual = n_tgt.dot(v_tgt - vp);
          ComposeGnState(gn_state, J, residual);
        }
      }
    }
  }

  BlockReduceSum(gn_state);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    gn_state_reduce_tmp.ptr(blockIdx.y)[blockIdx.x] = gn_state;
  }
}

} // namespace device

void IcpAlignDeviceStep(const DeviceArray3D<float> &vmap_src,
                        const DeviceArray3D<float> &nmap_src,
                        const DeviceArray3D<float> &vmap_tgt,
                        const DeviceArray3D<float> &nmap_tgt,
                        const CamIntr &cam_intr, const Eigen::Matrix3f &rot,
                        const Eigen::Vector3f &trans,
                        EdaGnState &h_gn_state_sum, cudaStream_t stream) {
  const int &cols = vmap_src.cols();
  const int &rows = vmap_src.rows();
  dim3 blk(32, 16);
  dim3 grid(DivideUp(cols, blk.x), DivideUp(rows, blk.y));

  DeviceArray2D<EdaGnState> gn_state_reduce_tmp(grid.y, grid.x);
  DeviceArray2D<EdaGnState> d_gn_state_sum(1, 1);

  EdaMatrix3f eda_rot = rot.cast<float>().eval();
  EdaVector3f eda_trans = trans.cast<float>().eval();
  device::IcpAlignStepKernel<<<grid, blk, 0, stream>>>(
      vmap_src, nmap_src, vmap_tgt, nmap_tgt, cam_intr, eda_rot, eda_trans,
      gn_state_reduce_tmp);

  CudaSafeCall(cudaStreamSynchronize(stream));
  CudaSafeCall(cudaGetLastError());

  assert(grid.x * grid.y <= 1024);
  device::SingleBlockReduceSum<<<dim3(1, 1, 1), grid, 0, stream>>>(
      gn_state_reduce_tmp, d_gn_state_sum);

  CudaSafeCall(cudaStreamSynchronize(stream));
  CudaSafeCall(cudaGetLastError());

  d_gn_state_sum.download(&h_gn_state_sum, sizeof(EdaGnState));

  CudaSafeCall(cudaStreamSynchronize(stream));
  CudaSafeCall(cudaGetLastError());
}

} // namespace rigid_icp

} // namespace icp
