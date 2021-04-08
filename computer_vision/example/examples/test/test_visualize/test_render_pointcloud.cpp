/**
 * Create point cloud from depth image and camera intrinsic, and visualize it
 * @author: xiaotaw
 * @email:
 * @date: 2020/08/23 15:02
 */
#include "common/safe_open.hpp"
#include "img_proc/cuda/compute_vertex.h"
#include "img_proc/cuda/cuda_texture_surface.h"
#include "json.hpp" // for nlohmann::json
#include "visualize/gl_render.h"
#include <atomic>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory> // for shared_ptr
#include <opencv2/opencv.hpp>
#include <vector_functions.hpp> // cuda header file, for make_ushort2

#if CV_VERSION_MAJOR >= 4
const int CV_ANYCOLOR = cv::IMREAD_ANYCOLOR;
const int CV_ANYDEPTH = cv::IMREAD_ANYDEPTH;
#else
const int CV_ANYCOLOR = CV_LOAD_IMAGE_ANYCOLOR;
const int CV_ANYDEPTH = CV_LOAD_IMAGE_ANYDEPTH;
#endif

//
std::atomic<bool> update(true);

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

static bool IsValidVertex(float4 vertex) {
  return (abs(vertex.x) > 1e-5) && (abs(vertex.y) > 1e-5) &&
         (abs(vertex.z) > 1e-5);
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
    data_dir = "/media/xt/8T/DATASETS/KinectDkDataset/20200630/";
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
  CamIntrInv cam_intr_inv(1.0 / fx, 1.0 / fy, cx, cy);

  // cuda resource
  CudaStream stream;
  auto depth_texture_surface = std::make_shared<CudaTextureSurface2D<ushort>>(
      img_size.height, img_size.width);
  auto vertex_texture_surface = std::make_shared<CudaTextureSurface2D<float4>>(
      img_size.height, img_size.width);
  // pagelock memory
  PagelockMemory depth_buffer_pagelock(sizeof(uint16_t) * img_size.area());
  PagelockMemory vertex_buffer_pagelock(sizeof(float4) * img_size.area());
  // necessary to sync after cudaMallocHost?
  CudaSafeCall(cudaDeviceSynchronize());
  CudaSafeCall(cudaGetLastError());

  std::string project_dir = "/data/Notes/others/computer_vision/example";
  std::string vert_shader_filename =
      project_dir + "/visualize/shaders/pointcloud.vert";
  std::string frag_shader_filename =
      project_dir + "/visualize/shaders/pointcloud.frag";
  GLRender render(vert_shader_filename, frag_shader_filename);

  for (unsigned i = 0; (!glfwWindowShouldClose(render.window_)) &&
                       (i < color_fn_list.size());) {
    if (update) {
      std::cout << i << " " << depth_fn_list[i] << std::endl;
      cv::Mat color_img =
          cv::imread(data_dir + color_fn_list[i], CV_ANYCOLOR | CV_ANYDEPTH);
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

      // std::cout << "color image type: " << color_img.type() << "; depth image
      // type: " << depth_img.type() << std::endl;
      assert(color_img.type() ==
             16); // "color image type is expected to be CV_8UC3"
      assert(depth_img.type() ==
             2); // "depth image type is expected to be CV_16UC1"
      assert(img_size == color_img.size());
      assert(img_size == depth_img.size());

      // memory to pagelock memory
      depth_buffer_pagelock.HostCopyFrom(static_cast<void *>(depth_img.data));
      // pagelock memory to device
      depth_buffer_pagelock.UploadToDevice(depth_texture_surface->d_array(),
                                           stream);

      // compute vertex from depth image on device
      ComputeVertex(depth_texture_surface, vertex_texture_surface, cam_intr_inv,
                    stream);
      // download vertex from device to host
      vertex_buffer_pagelock.DownloadFromDevice(
          vertex_texture_surface->d_array(), stream);

      stream.Synchronize();
      CudaSafeCall(cudaGetLastError());

      cv::Mat vertex_map = cv::Mat(img_size, CV_32FC4);
      vertex_buffer_pagelock.HostCopyTo(vertex_map.data);

      auto m = Eigen::AngleAxisf(-M_PI, Eigen::Vector3f::UnitZ());

      std::vector<Eigen::Vector3f> vertexs, colors;
      vertexs.resize(img_size.area());
      colors.resize(img_size.area());
      int i = 0;
      for (auto y = 0; y < img_size.height; y++) {
        for (auto x = 0; x < img_size.width; x++) {
          auto vertex = vertex_map.at<float4>(y, x);
          if (IsValidVertex(vertex)) {
            auto color = color_img.at<cv::Vec3b>(y, x);

            vertexs[i] =
                m * Eigen::Vector3f(vertex.x / 1000.0f, vertex.y / 1000.0f,
                                    vertex.z / 1000.0f);
            colors[i] = Eigen::Vector3f(color[2] / 256.0f, color[1] / 256.0f,
                                        color[0] / 256.0f);
            i++;
          }
        }
      }
      vertexs.resize(i);
      colors.resize(i);
      std::cout << "point cloud size: " << i << std::endl;

      GLRender::ProcessInput(render.window_);
      glfwPollEvents();
      render.UpdatePointCloud(vertexs, colors);
      render.RenderPointCloud();
      glfwSwapBuffers(render.window_);

      // update = false;
      // i++;

      // int a;
      // std::cin >> a;
    }
  }
  return 0;
}
