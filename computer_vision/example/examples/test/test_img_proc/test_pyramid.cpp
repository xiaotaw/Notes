/**
 * Create point cloud from depth image and camera intrinsic, and visualize it
 * @author: xiaotaw
 * @email:
 * @date: 2020/06/29 22:32
 */
#include "common/safe_open.hpp"
#include "img_proc/cuda/cuda_texture_surface.h"
#include "img_proc/cuda/pyramid.hpp"
#include "json.hpp" // for nlohmann::json
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory> // for shared_ptr
#include <opencv2/opencv.hpp>
#include <vector_functions.hpp> // for make_ushort2

#if CV_VERSION_MAJOR >= 4
const int CV_ANYCOLOR = cv::IMREAD_ANYCOLOR;
const int CV_ANYDEPTH = cv::IMREAD_ANYDEPTH;
#else
const int CV_ANYCOLOR = CV_LOAD_IMAGE_ANYCOLOR;
const int CV_ANYDEPTH = CV_LOAD_IMAGE_ANYDEPTH;
#endif

static int ReadImageList(const std::string image_list_fn,
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

static void DrawDepthImage(const std::string &win_name,
                           const cv::Mat &depth_img) {
  double max_depth, min_depth;
  cv::minMaxIdx(depth_img, &min_depth, &max_depth);
  // std::cout << "depth min and max: " << min_depth << " : " << max_depth <<
  // std::endl;
  cv::Mat depth_scale;
  cv::convertScaleAbs(depth_img, depth_scale, 255 / max_depth);
  cv::imshow(win_name, depth_scale);
}

static void HelpInfo() {
  std::cout << "Usage 1: " << std::endl;
  std::cout << "    ./executable data_dir" << std::endl;
}

int main(int argc, char **argv) {
  std::cout << "TODO: " << std::endl;
  std::cout << "    1. apply bilateral filtering on depth image" << std::endl;
  std::cout << "    2. try to build point cloud for VolumnDeform dataset,"
            << std::endl;
  std::cout << "       to check if mismatching between color and depth images"
            << std::endl;
  // args
  std::string data_dir;
  if (argc == 1) {
    data_dir = "/data/DATASETS/KinectDK/20200630/";
  } else if (argc == 2) {
    data_dir = argv[1];
  } else {
    HelpInfo();
    exit(EXIT_FAILURE);
  }

  // image list
  std::vector<std::string> color_fn_list, depth_fn_list;
  ReadImageList(data_dir + "rgb.txt", color_fn_list);
  ReadImageList(data_dir + "depth.txt", depth_fn_list);

  // read image info
  std::ifstream inf;
  OpenFileOrExit(inf, data_dir + "/image_info.json");
  nlohmann::json j = nlohmann::json::parse(inf);
  inf.close();
  auto img_size =
      cv::Size(j["depth image"]["width"], j["depth image"]["height"]);
  double fx = j["depth image"]["intrinsic"]["fx"];
  double fy = j["depth image"]["intrinsic"]["fy"];
  double cx = j["depth image"]["intrinsic"]["cx"];
  double cy = j["depth image"]["intrinsic"]["cy"];
  //float4 camera_intrinsic_inv = make_float4(1.0 / fx, 1.0 / fy, cx, cy);

  // cuda resource
  CudaStream stream;
  // pagelock memory
  PagelockMemory depth_buffer_pagelock(sizeof(uint16_t) * img_size.area());
  PagelockMemory color_buffer_pagelock(sizeof(uchar4) * img_size.area());

  // necessary to sync after cudaMallocHost?
  CudaSafeCall(cudaDeviceSynchronize());
  CudaSafeCall(cudaGetLastError());

  // pyramid
  Pyramid<ushort, 5> depth_pyramid(img_size.height, img_size.width, kNearest);
  Pyramid<uchar4, 5> color_pyramid(img_size.height, img_size.width, kNearest);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaSafeCall(cudaGetLastError());

  for (unsigned i = 0; i < color_fn_list.size(); i++) {
    std::cout << i << " " << depth_fn_list[i] << std::endl;
    cv::Mat color_img_ =
        cv::imread(data_dir + color_fn_list[i], CV_ANYCOLOR | CV_ANYDEPTH);
    cv::Mat color_img;
    // (xt) can we upload a RGB(3 channel) image into texture<uchar4> ?
    cv::cvtColor(color_img_, color_img, cv::COLOR_BGR2BGRA);
    cv::Mat depth_img =
        cv::imread(data_dir + depth_fn_list[i], CV_ANYCOLOR | CV_ANYDEPTH);
    if (color_img.empty()) {
      std::cout << "Error failed to read " << data_dir + color_fn_list[i]
                << std::endl;
      exit(EXIT_FAILURE);
    }
    if (depth_img.empty()) {
      std::cout << "Error failed to read " << data_dir + depth_fn_list[i]
                << std::endl;
      exit(EXIT_FAILURE);
    }

    // assert(color_img.type() == 16); // "color image type is expected to be
    // CV_8UC3"
    assert(depth_img.type() ==
           2); // "depth image type is expected to be CV_16UC1"
    assert(img_size == color_img.size());
    assert(img_size == depth_img.size());

    // memory to pagelock memory
    depth_buffer_pagelock.HostCopyFrom(static_cast<void *>(depth_img.data));
    color_buffer_pagelock.HostCopyFrom(static_cast<void *>(color_img.data));

    // pagelock memory to device
    depth_pyramid.UploadToPyramid(depth_buffer_pagelock, stream);
    color_pyramid.UploadToPyramid(color_buffer_pagelock, stream);

    // build pyramid level by level
    depth_pyramid.BuildPyramid(stream);
    color_pyramid.BuildPyramid(stream);
    CudaSafeCall(cudaStreamSynchronize(stream));
    CudaSafeCall(cudaGetLastError());

    // debug draw
    cv::Mat depth_helix = depth_pyramid.DownloadHelix(stream);
    cv::Mat color_helix = color_pyramid.DownloadHelix(stream);
    DrawDepthImage("depth helix", depth_helix);
    cv::imshow("color helix", color_helix);
    cv::waitKey(0);
  }
  return 0;
}
