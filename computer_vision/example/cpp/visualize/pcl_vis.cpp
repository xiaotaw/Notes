/**
 * @file pcl_vis.cpp
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-08-22
 * @copyright Copyright (c) 2021
 */
#include "pcl_vis.h"

PCLVis::PCLVis() {
  // build point cloud
  point_cloud_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  auto point = pcl::PointXYZRGB(255, 255, 255);
  point_cloud_->points.push_back(point);
  cloud_name_ = "cloud";

  // build normal
  normal_ = boost::make_shared<pcl::PointCloud<pcl::Normal>>();
  auto n = pcl::Normal(0, 0, 0);
  normal_->push_back(n);
  normal_name_ = "normal";

  // visualize point cloud
  viewer_ = boost::make_shared<pcl::visualization::PCLVisualizer>(
      "simple point cloud viewer");
  viewer_->addPointCloud<pcl::PointXYZRGB>(point_cloud_, cloud_name_);
  viewer_->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(point_cloud_, normal_, 10, 0.05, normal_name_);
  viewer_->addCoordinateSystem(2.0, cloud_name_, 0);
  viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cloud_name_);
  viewer_->setCameraPosition(-493.926, -2538.05, -4271.43, 0.0244369, -0.907735,
                             0.418832, 0);
  viewer_->registerKeyboardCallback(&keyboardEventOccurred, (void *)NULL);
}

void PCLVis::UpdatePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
  viewer_->updatePointCloud(cloud->makeShared(), cloud_name_);
  update = false;
}

void PCLVis::UpdatePointCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
  viewer_->updatePointCloud(cloud->makeShared(), cloud_name_);
  update = false;
}

void PCLVis::UpdatePointCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normal) {
  viewer_->updatePointCloud(cloud->makeShared(), cloud_name_);
  if (!normal->empty()) {
    viewer_->removePointCloud(normal_name_);
    viewer_->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normal, 10, 80, normal_name_);
  }
  update = false;
}

void PCLVis::keyboardEventOccurred(
    const pcl::visualization::KeyboardEvent &event, void *nothing) {
  if (event.keyDown()) {
    //打印出按下的按键信息
    cout << event.getKeySym() << endl;
    if (event.getKeySym() == "n") {
      update = true;
    }
  }
}

std::atomic<bool> PCLVis::update(false);