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

ImageProcessor::~ImageProcessor() { ReleaseBuffer(); }

void ImageProcessor::BuildVertexMap(const cv::Mat &depth_img) {
  assert(depth_img.cols == cols_);
  assert(depth_img.rows == rows_);
  // "depth image type is expected to be CV_16UC1"
  assert(depth_img.type() == 2);

  depth_h_pagelocked_->CopyFrom(static_cast<void *>(depth_img.data));

#ifdef USING_TEXTURE
  // Notes: not synchronized
  depth_d_->Upload(depth_h_pagelocked_->data(), stream_);
#else
  // Notes: synchronized
  depth_d_.upload(depth_h_pagelocked_->data(), sizeof(ushort) * cols_, rows_,
                  cols_);
#endif

  ComputeVertex(depth_d_, vertex_d_, cam_intr_inv_, stream_);
}

void ImageProcessor::BuildNormalMap(bool sync) {
#ifdef USING_TEXTURE
  LOG(WARNING) << "Not implemented yet!";
#else
  ComputeNormal(vertex_d_, normal_d_, stream_);
#endif
  if (sync) {
    stream_.Synchronize();
    CudaSafeCall(cudaGetLastError());
  }
}

void ImageProcessor::DownloadVertexNormal(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
    pcl::PointCloud<pcl::Normal>::Ptr &normal, const cv::Mat &color_img) {

  cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  normal = boost::make_shared<pcl::PointCloud<pcl::Normal>>();
  cloud->points.reserve(cols_ * rows_);
  normal->points.reserve(cols_ * rows_);

  auto IsValidVertex = [](const float &x, const float &y, const float &z) {
    return (abs(x) > 1e-5) && (abs(y) > 1e-5) && (abs(z) > 1e-5);
  };

#ifdef USING_TEXTURE
  // download vertex map from device to host
  cv::Mat vertex_tmp = cv::Mat(depth_img.size(), CV_32FC4);
  vertex_d_->Download(vertex_tmp.data, stream_);
  stream_.Synchronize();
  CudaSafeCall(cudaGetLastError());

  LOG(WARNING)
      << "It's Not implemented yet to compute Normal via Using Texture!";

  // convert vertex map into pointcloud
  for (auto v = 0; v < rows_; v++) {
    for (auto u = 0; u < cols_; u++) {
      auto vertex = vertex_tmp.at<float4>(v, u);
      const float &x = vertex.x;
      const float &y = vertex.y;
      const float &z = vertex.z;
      if (IsValidVertex(x, y, z)) {
        auto color = color_map.at<cv::Vec3b>(y, x);
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
      }
    }
  }
  cloud->resize(cloud->points.size());
#else
  stream_.Synchronize();
  CudaSafeCall(cudaGetLastError());

  // download vertex map from device to host
  std::vector<float> vertex_tmp(cols_ * rows_ * 3);
  std::vector<float> normal_tmp(cols_ * rows_ * 4);
  vertex_d_.download(&vertex_tmp[0], sizeof(float) * cols_);
  CudaSafeCall(cudaGetLastError());
  normal_d_.download(&normal_tmp[0], sizeof(float) * cols_);
  CudaSafeCall(cudaGetLastError());

  // convert vertex and normal into pointcloud
  for (auto v = 0; v < rows_; v++) {
    for (auto u = 0; u < cols_; u++) {
      const int i = v * cols_ + u;
      const float &x = vertex_tmp[i];
      const float &y = vertex_tmp[i + cols_ * rows_];
      const float &z = vertex_tmp[i + cols_ * rows_ * 2];
      if (IsValidVertex(x, y, z)) {
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
#endif
}

void ImageProcessor::AllocateBuffer() {
#ifdef USING_TEXTURE
  depth_d_ = std::make_shared<CudaTextureSurface2D<ushort>>(rows_, cols_);
  vertex_d_ = std::make_shared<CudaTextureSurface2D<float4>>(rows_, cols_);
#else
  depth_d_.create(rows_, cols_);
  // Structure of Array(SoA): vertex x, y, z
  vertex_d_.create(rows_ * 3, cols_);
  normal_d_.create(rows_ * 4, cols_);
#endif
  depth_h_pagelocked_ =
      std::make_shared<PagelockedMemory>(sizeof(ushort) * rows_ * cols_);
  // necessary to sync after cudaMallocHost?
  CudaSafeCall(cudaDeviceSynchronize());
  CudaSafeCall(cudaGetLastError());
}

void ImageProcessor::ReleaseBuffer() {
#ifdef USING_TEXTURE
#else
  depth_d_.release();
  vertex_d_.release();
  normal_d_.release();
#endif
}