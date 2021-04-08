/**
 * image processor
 * @author: xiaotaw
 * @email:
 * @date: 2020/08/25 04:15
 */
#pragma once
#include "dataset/camera_params.hpp"
#include "img_proc/cam_intr.h"
#include "img_proc/cuda/compute_vertex.h"
#include "img_proc/cuda/cuda_texture_surface.h"
#include "img_proc/cuda/cuda_timmer.h"
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <vector_functions.hpp> // for make_ushort2

class ImageProcessor {
public:
  DISABLE_COPY_ASSIGN_MOVE(ImageProcessor);

  /**
   * @brief ctor
   */
  ImageProcessor(const CameraParams &camera_params);
  ImageProcessor(const int &cols, const int &rows, const double &fx,
                 const double &fy, const double &cx, const double &cy);

  /**
   * @brief
   *
   * @param[in] depth_img
   */
  void BuildVertexMap(const cv::Mat &depth_img);

  int cols() const { return cols_; }
  int rows() const { return rows_; }

public:
  /////////////////////////// DEBUG Method /////////////////////////////
  // inside synchronize
  void BuildVertexMap(const cv::Mat &depth_img, cv::Mat &vertex_map);

private:
  int cols_, rows_;
  CamIntr cam_intr_;
  CamIntrInv cam_intr_inv_;

  CudaTextureSurface2D<ushort>::Ptr depth_texture_surface_;
  CudaTextureSurface2D<float4>::Ptr vertex_texture_surface_;

  PagelockMemory::Ptr depth_buffer_pagelock_;
  PagelockMemory::Ptr vertex_buffer_pagelock_;

  CudaStream stream_;
  CUDATimer timer_;

  void AllocateBuffer();
  void ReleaseBuffer();
};