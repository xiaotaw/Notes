/**
 * @file dataset_azure_kinect.cpp
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2020-08-24 initial version
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "dataset_azure_kinect.h"
#include "common/logging.h"
#include "common/opencv_cross_version.h"
#include "common/safe_open.hpp"
#include "json.hpp"
#include <unistd.h>

AzureKinectDataset::AzureKinectDataset(const std::string &data_path)
    : RgbdDataset(data_path) {
  //
  ReadImageList((fs::path(data_path_) / "depth.txt").string(),
                depth_filenames_);
  ReadImageList((fs::path(data_path_) / "rgb.txt").string(), color_filenames_);
  LOG(INFO) << "Init AzureKinect Dataset, color images = "
            << color_filenames_.size() << ", depth images = "
            << depth_filenames_.size() << std::endl;

  StartPreloadThread();

  // dataset info in json format
  std::ifstream inf;
  OpenFileOrExit(inf, (fs::path(data_path_) / "image_info.json").string());
  nlohmann::json j = nlohmann::json::parse(inf);
  inf.close();
  auto img_size =
      cv::Size(j["depth image"]["width"], j["depth image"]["height"]);
  double fx = j["depth image"]["intrinsic"]["fx"];
  double fy = j["depth image"]["intrinsic"]["fy"];
  double cx = j["depth image"]["intrinsic"]["cx"];
  double cy = j["depth image"]["intrinsic"]["cy"];
  color_camera_params_ =
      CameraParams(img_size.height, img_size.width, fx, fy, cx, cy);
}

int AzureKinectDataset::ReadImageList(const std::string &image_list_fn,
                                      std::vector<std::string> &fn_list) {
  std::ifstream inf;
  OpenFileOrExit(inf, image_list_fn);
  std::string line, timestamp, filename;
  while (getline(inf, line)) {
    if (line[0] == '#') {
      continue;
    }
    size_t n = line.find_first_of(" ");
    size_t t = line.find_first_not_of("\r\n");
    if (n != std::string::npos && t != std::string::npos) {
      timestamp = line.substr(0, n);
      filename = line.substr(n + 1, t - n);
    }
    fn_list.push_back(filename);
  }
  return fn_list.size();
}
