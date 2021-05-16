/**
 * @file frame.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-04-25
 *
 * @copyright Copyright (c) 2021
 */
#pragma once
#include "common/disable_copy_assign_move.h"
#include "img_proc/cuda/containers/device_array.hpp"
#include <Eigen/Core>

struct Frame {
  DISABLE_COPY_ASSIGN_MOVE(Frame);
  using Ptr = std::shared_ptr<Frame>;

  Frame() {}
  Frame(const int &rows, const int &cols) {
    d_color.create(rows, cols);
    d_depth.create(rows, cols);
    d_vertex.create(rows, kVertexChannels, cols);
    d_normal.create(rows, kNormalChannels, cols);
  }

  ~Frame() {
    d_color.release();
    d_depth.release();
    d_vertex.release();
    d_normal.release();
  }

  // size of frame
  int cols;
  int rows;

  // device memory of frame data
  DeviceArray2D<unsigned int> d_color;
  DeviceArray2D<unsigned short> d_depth;
  DeviceArray3D<float> d_vertex;
  DeviceArray3D<float> d_normal;

  // camera pose
  Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
  Eigen::Vector3d trans = Eigen::Vector3d::Zero();

  // 3 channels in vertex: x, y, z
  const static int kVertexChannels = 3;

  // 4 channels in normal: normal_x, normal_y, normal_z, curventure
  const static int kNormalChannels = 4;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};