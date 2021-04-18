/**
 * test pcl_visualizer
 * @author: xiaotaw
 * @email:
 * @date: 2020/08/26 10:52
 */
#include "common/logging.h"
#include "common/time_logger.h"
#include "dataset/dataset.h"
#include "img_proc/image_proc.h"
#include "visualize/pcl_vis.h"
#include <algorithm>
#include <cctype> // for std::tolower
#include <iostream>
#include <string>

static void HelpInfo() {
  std::cout << "Usage 1: " << std::endl;
  std::cout << "    ./executable data_dir data_type" << std::endl;
  std::cout << "data_type could be: azure_kinect, nyu_v2_raw, nyu_v2_labeled"
            << std::endl;
}

static void PrintTodo() {
  std::cout << "TODO: " << std::endl;
  std::cout << "    1. apply bilateral filtering on depth image" << std::endl;
  std::cout << "    2. try to build point cloud for VolumnDeform dataset, to "
               "check if mismatching between color and depth images"
            << std::endl;
}

int main(int argc, char **argv) {
  PrintTodo();
  // args
  std::string data_dir = "/media/xt/8T/DATASETS/KinectDkDataset/20200701/";
  std::string data_type = "azure_kinect";
  if (argc == 1) {
  } else if (argc == 3) {
    data_dir = argv[1];
    data_type = argv[2];
  } else {
    HelpInfo();
    exit(EXIT_FAILURE);
  }

  RgbdDataset::Ptr dataset = CreateDataset(data_type, data_dir);
  ImageProcessor image_proc(dataset->color_camera_params());
  PCLVis vis;

  TimeLogger::printTimeLog("start");
  for (int i = 0; i < int(dataset->num());) {
    if ((vis.viewer_)->wasStopped()) {
      break;
    }

    (vis.viewer_)->spinOnce();
    if (PCLVis::update) {
      TimeLogger::printTimeLog("new update");

      cv::Mat color_img, depth_img;
      dataset->FetchNextFrame(depth_img, color_img);
      TimeLogger::printTimeLog("fetch images");

      image_proc.BuildVertexMap(depth_img);
      TimeLogger::printTimeLog("compute vertex");

      image_proc.BuildNormalMap(true);
      TimeLogger::printTimeLog("compute normal");

      auto cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
      auto normal = boost::make_shared<pcl::PointCloud<pcl::Normal>>();
      image_proc.DownloadVertexNormal(cloud, normal, color_img);

      if (cloud->size() == 0) {
        LOG(WARNING) << "point cloud size is ZERO.";
      }

      vis.UpdatePointCloud(cloud, normal);

      i++;
      TimeLogger::printTimeLog("Loop Done");
    }
  }
  return 0;
}
