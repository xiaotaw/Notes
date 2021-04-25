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

namespace device {

__inline__ __device__ EdaVector3f ReadEdaVector3f(const PtrStepSz<float> &map,
                                                  const int &u, const int &v) {
  return EdaVector3f(map.ptr(v, 0)[u], map.ptr(v, 1)[u], map.ptr(v, 2)[u]);
}

__device__ void ComposeJtJ_JR(EdaVectorf<JTJ_JR_SIZE> &JtJ_JR,
                              const float (&J)[6], const float &res) {
  // clang-format off
  // lower triangle of H
  JtJ_JR(0)  = J[0] * J[0]; 
  JtJ_JR(1)  = J[0] * J[1]; JtJ_JR(2)  = J[1] * J[1];
  JtJ_JR(3)  = J[0] * J[2]; JtJ_JR(4)  = J[1] * J[2]; JtJ_JR(5)  = J[2] * J[2];
  JtJ_JR(6)  = J[0] * J[3]; JtJ_JR(7)  = J[1] * J[3]; JtJ_JR(8)  = J[2] * J[3]; JtJ_JR(9) = J[3] * J[3];
  JtJ_JR(10) = J[0] * J[4]; JtJ_JR(11) = J[1] * J[4]; JtJ_JR(12) = J[2] * J[4]; JtJ_JR(13) = J[3] * J[4]; JtJ_JR(14) = J[4] * J[4];
  JtJ_JR(15) = J[0] * J[5]; JtJ_JR(16) = J[1] * J[5]; JtJ_JR(17) = J[2] * J[5]; JtJ_JR(18) = J[3] * J[5]; JtJ_JR(19) = J[4] * J[5]; JtJ_JR(20) = J[5] * J[5];
  // J*residual
  JtJ_JR(21) = J[0] * res;  JtJ_JR(22) = J[1] * res;  JtJ_JR(23) = J[2] * res;  JtJ_JR(24) = J[3] * res;  JtJ_JR(25) = J[4] * res;  JtJ_JR(26) = J[5] * res;
  // clang-format on
}

/**
 * @brief warp reduce sum
 * @param[in out] JtJ_JR
 * @note https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
 */
__device__ void WarpReduceSum(EdaVectorf<JTJ_JR_SIZE> &JtJ_JR) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < JTJ_JR_SIZE; i++) {
      JtJ_JR[i] += __shfl_down_sync(0xFFFFFFFF, JtJ_JR[i], offset);
    }
  }
}

/**
 * @brief 2D Block Reduce sum
 * @param[in out] JtJ_JR
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
__device__ void BlockReduceSum(EdaVectorf<JTJ_JR_SIZE> &JtJ_JR) {

  static __shared__ unsigned char
      shared_mem[32 * sizeof(EdaVectorf<JTJ_JR_SIZE>)];
  EdaVectorf<JTJ_JR_SIZE>(&shared)[32] =
      reinterpret_cast<EdaVectorf<JTJ_JR_SIZE>(&)[32]>(shared_mem);

  int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
  int lane_id = thread_id % warpSize;
  int warp_id = thread_id / warpSize;

  WarpReduceSum(JtJ_JR);

  if (lane_id == 0) {
    shared[warp_id] = JtJ_JR;
  }

  __syncthreads();

  JtJ_JR = (thread_id < blockDim.x * blockDim.y / warpSize)
               ? shared[lane_id]
               : EdaVectorf<JTJ_JR_SIZE>::Zero();

  if (warp_id == 0) {
    WarpReduceSum(JtJ_JR);
  }
}

__global__ void
SingleBlockReduceSum(const PtrStepSz<EdaVectorf<JTJ_JR_SIZE>> jtj_jr_reduce_tmp,
                     PtrStepSz<EdaVectorf<JTJ_JR_SIZE>> jtj_jr) {
  const int u = threadIdx.x;
  const int v = threadIdx.y;
  EdaVectorf<JTJ_JR_SIZE> val = jtj_jr_reduce_tmp.ptr(v)[u];

  BlockReduceSum(val);

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    jtj_jr.ptr(0)[0] = val;
  }
}

__global__ void
IcpAlignStepKernel(const PtrStepSz<float> vmap_src,
                   const PtrStepSz<float> nmap_src,
                   const PtrStepSz<float> vmap_tgt,
                   const PtrStepSz<float> nmap_tgt, const CamIntr cam_intr,
                   const Eigen::Matrix3d rot, const Eigen::Vector3d trans,
                   PtrStepSz<EdaVectorf<JTJ_JR_SIZE>> jtj_jr_reduce_tmp) {
  const int u = threadIdx.x + blockDim.x * blockIdx.x;
  const int v = threadIdx.y + blockDim.y * blockIdx.y;

  EdaVectorf<JTJ_JR_SIZE> JtJ_JR = EdaVectorf<JTJ_JR_SIZE>::Zero();

  if (u < vmap_src.cols && v < vmap_src.rows) {
    EdaVector3f v_src = ReadEdaVector3f(vmap_src, u, v);
    EdaVector3f n_src = ReadEdaVector3f(nmap_src, u, v);

    if (!IsZeroVertex(v_src)) {
      EdaVector3f vp = rot.cast<float>() * v_src + trans.cast<float>();
      int u_proj = std::round(vp(0) / vp(2) * cam_intr.fx + cam_intr.cx);
      int v_proj = std::round(vp(1) / vp(2) * cam_intr.fy + cam_intr.cy);

      if (u_proj >= 0 && u_proj < vmap_tgt.cols && v_proj > 0 &&
          v_proj < vmap_tgt.rows) {
        EdaVector3f v_tgt = ReadEdaVector3f(vmap_tgt, u_proj, v_proj);
        EdaVector3f n_tgt = ReadEdaVector3f(nmap_tgt, u_proj, v_proj);

        if (!IsZeroVertex(v_tgt)) {
          float J[6] = {0};
          *((EdaVector3f *)(&J[0])) = n_tgt;
          *((EdaVector3f *)(&J[3])) = vp.cross(n_tgt);
          float residual = n_tgt.dot(v_tgt - vp);
          ComposeJtJ_JR(JtJ_JR, J, residual);
        }
      }
    }
  }
  BlockReduceSum(JtJ_JR);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    jtj_jr_reduce_tmp.ptr(blockIdx.y)[blockIdx.x] = JtJ_JR;
  }
}

} // namespace device

void IcpAlignDeviceStep(const DeviceArray3D<float> &vmap_src,
                        const DeviceArray3D<float> &nmap_src,
                        const DeviceArray3D<float> &vmap_tgt,
                        const DeviceArray3D<float> &nmap_tgt,
                        const CamIntr &cam_intr, const Eigen::Matrix3d &rot,
                        const Eigen::Vector3d &trans,
                        EdaVectorf<JTJ_JR_SIZE> &h_jtj_jr,
                        cudaStream_t stream) {
  const int &cols = vmap_src.cols();
  const int &rows = vmap_src.rows();
  dim3 blk(16, 16);
  dim3 grid(DivideUp(cols, blk.x), DivideUp(rows, blk.y));

  DeviceArray2D<EdaVectorf<JTJ_JR_SIZE>> jtj_jr_reduce_tmp(grid.x, grid.y);
  DeviceArray2D<EdaVectorf<JTJ_JR_SIZE>> jtj_jr(1, 1);
  device::IcpAlignStepKernel<<<grid, blk, 0, stream>>>(
      vmap_src, nmap_src, vmap_tgt, nmap_tgt, cam_intr, rot, trans,
      jtj_jr_reduce_tmp);
  device::SingleBlockReduceSum<<<dim3(1, 1, 1), grid, 0, stream>>>(
      jtj_jr_reduce_tmp, jtj_jr);

  CudaSafeCall(cudaStreamSynchronize(stream));
  CudaSafeCall(cudaGetLastError());

  jtj_jr.download(&h_jtj_jr, sizeof(EdaVectorf<JTJ_JR_SIZE>));
}

} // namespace icp