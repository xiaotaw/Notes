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

const int JTJ_JR_SIZE = 27;

using EdaVector3f = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;

template <int DIM>
using EdaVectorf = Eigen::Matrix<float, DIM, 1, Eigen::DontAlign>;

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
                        const CamIntr &cam_intr, const Eigen::Matrix3d &rot,
                        const Eigen::Vector3d &trans,
                        EdaVectorf<JTJ_JR_SIZE> &h_jtj_jr, float &score,
                        cudaStream_t stream = 0);

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

} // namespace icp
