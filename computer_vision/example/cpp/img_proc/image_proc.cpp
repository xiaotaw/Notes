/**
 * @file image_proc.cpp
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2020-08-26
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "img_proc/image_proc.h"
#include "img_proc/cuda/compute_normal.h"
#include "img_proc/cuda/compute_vertex.h"
#include "img_proc/icp/icp.h"
#include <boost/make_shared.hpp>

ImageProcessor::ImageProcessor(const int &cols, const int &rows,
                               const double &fx, const double &fy,
                               const double &cx, const double &cy)
    : cols_(cols), rows_(rows) {
  cam_intr_ = CamIntr(fx, fy, cx, cy);
  cam_intr_inv_ = CamIntrInv(cam_intr_);
  AllocateBuffer();
}

ImageProcessor::ImageProcessor(const CameraParams &params)
    : cols_(params.cols_), rows_(params.rows_) {
  cam_intr_ = CamIntr(params.fx_, params.fy_, params.cx_, params.cy_);
  cam_intr_inv_ = CamIntrInv(cam_intr_);
  AllocateBuffer();
}

void ImageProcessor::DownloadCurrentFrame(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
    pcl::PointCloud<pcl::Normal>::Ptr &normal, const cv::Mat &color_img) {
  stream_.Synchronize();
  CudaSafeCall(cudaGetLastError());

  cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  normal = boost::make_shared<pcl::PointCloud<pcl::Normal>>();
  cloud->points.reserve(cols_ * rows_);
  normal->points.reserve(cols_ * rows_);

  // download vertex map from device to host
  std::vector<float> vertex_tmp(cols_ * rows_ * 3);
  std::vector<float> normal_tmp(cols_ * rows_ * 4);
  cur_frame_->d_vertex.download(&vertex_tmp[0], sizeof(float) * cols_);
  CudaSafeCall(cudaGetLastError());
  cur_frame_->d_normal.download(&normal_tmp[0], sizeof(float) * cols_);
  CudaSafeCall(cudaGetLastError());

  // convert vertex and normal into pointcloud
  for (auto v = 0; v < rows_; v++) {
    for (auto u = 0; u < cols_; u++) {
      const int i = v * cols_ + u;
      const float &x = vertex_tmp[i];
      const float &y = vertex_tmp[i + cols_ * rows_];
      const float &z = vertex_tmp[i + cols_ * rows_ * 2];
      if (!IsZeroVertex(x, y, z)) {
        pcl::PointXYZRGB point;
        point.x = x;
        point.y = y;
        point.z = z;
        if (!color_img.empty()) {
          auto color = color_img.at<cv::Vec3b>(v, u);
          point.b = color[0];
          point.g = color[1];
          point.r = color[2];
        }
        cloud->points.emplace_back(point);

        pcl::Normal n;
        n.normal_x = normal_tmp[i];
        n.normal_y = normal_tmp[i + cols_ * rows_];
        n.normal_z = normal_tmp[i + cols_ * rows_ * 2];
        n.curvature = normal_tmp[i + cols_ * rows_ * 3];
        normal->points.emplace_back(n);
      }
    }
  }
  // set cloud->width
  cloud->resize(cloud->points.size());
  normal->resize(normal->points.size());
}

void ImageProcessor::ProcessImage(const cv::Mat &depth_img,
                                  const cv::Mat &color_img) {
  assert(depth_img.rows == rows_);
  assert(depth_img.cols == cols_);
  assert(color_img.rows == rows_);
  assert(color_img.cols == cols_);
  // "depth image type is expected to be CV_16UC1"
  assert(depth_img.type() == 2);

  depth_h_pagelocked_->CopyFrom(static_cast<void *>(depth_img.data));

  cur_frame_ = std::make_shared<Frame>(rows_, cols_);
  cur_frame_->d_depth.upload(depth_h_pagelocked_->data(),
                             sizeof(ushort) * cols_, rows_, cols_);
  ComputeVertex(cur_frame_->d_depth, cur_frame_->d_vertex, cam_intr_inv_,
                stream_);
  ComputeNormal(cur_frame_->d_vertex, cur_frame_->d_normal, stream_);

  if (!frames_.empty()) {
    Frame::Ptr &last_frame_ = frames_.back();
    Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
    Eigen::Vector3d trans = Eigen::Vector3d::Zero();
    float score = 0;
    icp::rigid_icp::IcpAlign(cur_frame_->d_vertex, cur_frame_->d_normal,
                             last_frame_->d_vertex, last_frame_->d_normal,
                             cam_intr_, rot, trans, score, stream_);
    cur_frame_->rot = rot * last_frame_->rot;
    cur_frame_->trans = rot * last_frame_->trans + trans;
  }

  frames_.emplace_back(cur_frame_);
  if (frames_.size() > 15) {
    frames_.pop_front();
  }

  // TODO: for color image
}

void ImageProcessor::Synchronize() {
  stream_.Synchronize();
  CudaSafeCall(cudaGetLastError());
}

void ImageProcessor::AllocateBuffer() {
  depth_h_pagelocked_ =
      std::make_shared<PagelockedMemory>(sizeof(ushort) * rows_ * cols_);
}