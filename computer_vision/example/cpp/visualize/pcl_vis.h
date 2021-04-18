/**
 * @file pcl_vis.h
 * @author xiaotaw (you@domain.com)
 * @brief A 3D visualizer for RGBD data.
 * @version 0.1
 * @date 2020-08-21
 * @copyright Copyright (c) 2021
 */
#pragma once
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

class PCLVis {
public:
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_;
  pcl::PointCloud<pcl::Normal>::Ptr normal_;
  pcl::visualization::PCLVisualizer::Ptr viewer_;
  std::string cloud_name_;
  std::string normal_name_;

  struct float4 {
    float x, y, z, w;
  };

  PCLVis();

  void UpdatePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

  void UpdatePointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

  void UpdatePointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                        const pcl::PointCloud<pcl::Normal>::Ptr normal);

  static void
  keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
                        void *nothing);

  static std::atomic<bool> update;
};
