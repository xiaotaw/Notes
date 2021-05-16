/**
 * @file icp.cpp
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-04-25 initial version
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "icp.h"
#include <iostream>

namespace icp {

namespace rigid_icp {

/**
 * @brief 向量对应的反对称矩阵
 *
 * @param[in] t 向量
 * @return Eigen::Matrix3d 反对称矩阵
 */
static Eigen::Matrix3d hat(const Eigen::Vector3d &t) {
  Eigen::Matrix3d t_hat;
  // clang-format off
  t_hat <<    0, -t(2),  t(1),
           t(2),     0, -t(0),
          -t(1),  t(0),     0;
  // clang-format on
  return t_hat;
}

/**
 * @brief Exponential Map of se3
 * 理论公式见文章 Lie Groups for 2D and 3D Transformations, Ethan Eade, 2017.
 * 中的第3.2节 Exponential Map 的（77-84）。 代码实现参见：
 * https://github.com/RainerKuemmerle/g2o 中的 g2o/types/slam3d/se3quat.h
 *
 * @param[in] omega 李代数se3中的与旋转对应的量（so3）
 * @param[in] upsilon 李代数se3中与平移对应的量
 * @param[out] rot 李群SE3的旋转量（SO3）
 * @param[out] trans 李群SE3中的平移量
 */
static void se3exp(const Eigen::Vector3d &omega, const Eigen::Vector3d &upsilon,
                   Eigen::Matrix3d &rot, Eigen::Vector3d &trans) {
  double theta = omega.norm();
  double theta2 = theta * theta;

  // Ethan Eade(2017): For implementation purposes, Taylor expansions of A, B,
  // and C should be used when θ*θ is small.
  // For A, sin(theta) / theta = 1, when theta -> 0;
  // For B, 1 - cos(theta) = cos(0) - cos(theta)
  //                       = -2 * sin((0 + theta) / 2) * sin((0 - theta) / 2)
  //                       = 2 * sin(theta / 2) * sin(theta / 2),
  //  so B = 1/2, when theta -> 0;
  // For C, according formula (75), just leave oout any term with theta,
  //  so C = 1 / (3!) = 1/6, when theta -> 0;
  double A = 1.0;
  double B = 0.5;
  double C = 1.0 / 6.0;
  if (theta2 > std::numeric_limits<double>::epsilon()) {
    A = std::sin(theta) / theta;
    B = (1 - std::cos(theta)) / (theta * theta);
    C = (1 - A) / (theta * theta);
  }

  Eigen::Matrix3d omega_hat = hat(omega);
  rot = Eigen::Matrix3d::Identity() + A * omega_hat + B * omega_hat * omega_hat;
  Eigen::Matrix3d V =
      Eigen::Matrix3d::Identity() + B * omega_hat + C * omega_hat * omega_hat;
  trans = V * upsilon;
}

void IcpAlign(const DeviceArray3D<float> &vmap_src,
              const DeviceArray3D<float> &nmap_src,
              const DeviceArray3D<float> &vmap_tgt,
              const DeviceArray3D<float> &nmap_tgt, const CamIntr &cam_intr,
              Eigen::Matrix3d &rot, Eigen::Vector3d &trans, float &score,
              cudaStream_t stream) {

  for (int iter = 0; iter < 20; ++iter) {
    // 高斯牛顿法，计算JtJ和J*Residual。
    EdaGnState h_gn_state_sum = EdaGnState::Zero();
    IcpAlignDeviceStep(vmap_src, nmap_src, vmap_tgt, nmap_tgt, cam_intr, rot.cast<float>(),
                       trans.cast<float>(), h_gn_state_sum, stream);
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> JtJ_host;
    Eigen::Matrix<double, 6, 1> JR_host;

    int i = 0;
    for (int u = 0; u < 6; ++u) {
      for (int v = 0; v < u; ++v) {
        JtJ_host(u, v) = h_gn_state_sum(i);
        JtJ_host(v, u) = h_gn_state_sum(i);
        i++;
      }
      JtJ_host(u, u) = h_gn_state_sum(i++);
    }
    for (int u = 0; u < 6; ++u) {
      JR_host(u) = h_gn_state_sum(i++);
    }

    // 计算score
    double cnt = h_gn_state_sum(i++);
    double score_sum = h_gn_state_sum(i++);
    score = cnt > 0 ? score_sum / cnt : 0;

    // JtJ为对称矩阵，采用ldlt分解，求方程组的解
    Eigen::Matrix<double, 6, 1> delta = JtJ_host.ldlt().solve(JR_host);

    // se3更新
    Eigen::Matrix3d delta_R;
    Eigen::Vector3d delta_t;
    se3exp(Eigen::Vector3d(delta(0), delta(1), delta(2)),
           Eigen::Vector3d(delta(3), delta(4), delta(5)), delta_R, delta_t);
    rot = delta_R * rot;
    trans = delta_t + delta_R * trans;
  }
}

} // namespace rigid_icp
} // namespace icp