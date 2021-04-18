/**
 * @file image_proc.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2020-08-25 initial version
 *
 * @copyright Copyright (c) 2021
 */

#pragma once
#include "dataset/camera_params.hpp"
#include "img_proc/cam_intr.h"
#include "img_proc/cuda/cuda_stream.h"
#include "img_proc/cuda/cuda_timmer.h"
#include "img_proc/cuda/pagelocked_memory.h"
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector_functions.hpp> // for make_ushort2

//#define USING_TEXTURE
#ifdef USING_TEXTURE
#include "img_proc/cuda/cuda_texture_surface.h"
#else
#include "img_proc/cuda/containers/device_array.hpp"
#endif

class ImageProcessor {
public:
  DISABLE_COPY_ASSIGN_MOVE(ImageProcessor);

  ImageProcessor(const CameraParams &camera_params);
  ImageProcessor(const int &cols, const int &rows, const double &fx,
                 const double &fy, const double &cx, const double &cy);

  ~ImageProcessor();

  /**
   * @brief compute vertex map from depth image
   * @param[in] depth_img depth image
   */
  void BuildVertexMap(const cv::Mat &depth_img);

  void BuildNormalMap(bool sync = false);

  /**
   * @brief Download vertex and normal data as pcl::PointCloud
   * @param[out] cloud the vertex (pcl::PointCloud)
   * @param[out] normal the normal of each vertex
   * @param[in] color_img the color of vertex
   * @note it's a debug method
   */
  void DownloadVertexNormal(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                            pcl::PointCloud<pcl::Normal>::Ptr &normal,
                            const cv::Mat &color_img = cv::Mat());

  int cols() const { return cols_; }
  int rows() const { return rows_; }

private:
  int cols_, rows_;
  CamIntr cam_intr_;
  CamIntrInv cam_intr_inv_;

#ifdef USING_TEXTURE
  CudaTextureSurface2D<ushort>::Ptr depth_d_;
  CudaTextureSurface2D<float4>::Ptr vertex_d_;
#else
  DeviceArray2D<ushort> depth_d_;

  ///////////////////////// SHOULD USING DeviceArray3D //////////////////////
  DeviceArray2D<float> vertex_d_;
  DeviceArray2D<float> normal_d_;
#endif

  PagelockedMemory::Ptr depth_h_pagelocked_;
  CudaStream stream_;
  CUDATimer timer_;

private:
  void AllocateBuffer();
  void ReleaseBuffer();
};