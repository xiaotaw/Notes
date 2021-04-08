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

void ImageProcessor::AllocateBuffer() {
  depth_texture_surface_ =
      std::make_shared<CudaTextureSurface2D<ushort>>(rows_, cols_);
  vertex_texture_surface_ =
      std::make_shared<CudaTextureSurface2D<float4>>(rows_, cols_);
  depth_buffer_pagelock_ =
      std::make_shared<PagelockMemory>(sizeof(uint16_t) * rows_ * cols_);
  vertex_buffer_pagelock_ =
      std::make_shared<PagelockMemory>(sizeof(float4) * rows_ * cols_);
  // necessary to sync after cudaMallocHost?
  CudaSafeCall(cudaDeviceSynchronize());
  CudaSafeCall(cudaGetLastError());
}

void ImageProcessor::BuildVertexMap(const cv::Mat &depth_img) {
  assert(depth_img.size().width == cols_);
  assert(depth_img.size().height == rows_);
  // "depth image type is expected to be CV_16UC1"
  assert(depth_img.type() == 2);

  depth_buffer_pagelock_->HostCopyFrom(static_cast<void *>(depth_img.data));
  depth_buffer_pagelock_->UploadToDevice(depth_texture_surface_->d_array(),
                                         stream_);
  ComputeVertex(depth_texture_surface_, vertex_texture_surface_, cam_intr_inv_,
                stream_);
}

// inside synchronize
void ImageProcessor::BuildVertexMap(const cv::Mat &depth_img,
                                    cv::Mat &vertex_map) {
  BuildVertexMap(depth_img);
  vertex_buffer_pagelock_->DownloadFromDevice(
      vertex_texture_surface_->d_array(), stream_);
  stream_.Synchronize();
  CudaSafeCall(cudaGetLastError());
  vertex_buffer_pagelock_->HostCopyTo(vertex_map.data);
}