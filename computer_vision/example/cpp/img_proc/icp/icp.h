/**
 * @file icp.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-04-24 declear functions
 *       2021-04-25 initial version
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include "img_proc/cam_intr.h"
#include "img_proc/cuda/containers/device_array.hpp"
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <cuda_runtime_api.h>

namespace icp {

/**
 * @brief 中间状态量，用于高斯牛顿法求解point-to-plane icp，其中：
 *        前21个元素，储存6x6的JtJ的上三角，(1+6)*6/2=21;
 *        中间6个元素，储存6x1的J*residual；
 *        1个元素，储存score；
 *        1个元素，储存vertex是否配对
 */
const int kGnStateSize = 29;
using EdaGnState = Eigen::Matrix<float, kGnStateSize, 1, Eigen::DontAlign>;

using EdaVector3f = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using EdaMatrix3f = Eigen::Matrix<float, 3, 3, Eigen::DontAlign>;

namespace rigid_icp {
/**
 * @brief icp align from src to tgt
 * @param[in] vertex_src
 * @param[in] normal_src
 * @param[in] vertex_tgt
 * @param[in] normal_tgt
 * @param[in] cam_intr
 * @param[in out] rot
 * @param[in out] trans
 * @param[out] score
 * @param[in] stream
 * @todo score is not calculated
 */
void IcpAlign(const DeviceArray3D<float> &vmap_src,
              const DeviceArray3D<float> &nmap_src,
              const DeviceArray3D<float> &vmap_tgt,
              const DeviceArray3D<float> &nmap_tgt, const CamIntr &cam_intr,
              Eigen::Matrix3d &rot, Eigen::Vector3d &trans, float &score,
              cudaStream_t stream = 0);

/**
 * @brief device step of icp align from src to tgt
 * @param[in] vertex_src
 * @param[in] normal_src
 * @param[in] vertex_tgt
 * @param[in] normal_tgt
 * @param[in] cam_intr
 * @param[in] rot
 * @param[in] trans
 * @param[out] h_jtj_jr
 * @param[out] score
 * @param[in] stream
 * @todo score is not calculated
 */
void IcpAlignDeviceStep(const DeviceArray3D<float> &vmap_src,
                        const DeviceArray3D<float> &nmap_src,
                        const DeviceArray3D<float> &vmap_tgt,
                        const DeviceArray3D<float> &nmap_tgt,
                        const CamIntr &cam_intr, const Eigen::Matrix3f &rot,
                        const Eigen::Vector3f &trans,
                        EdaGnState &h_gn_state_sum, cudaStream_t stream = 0);

} // namespace rigid_icp

} // namespace icp
