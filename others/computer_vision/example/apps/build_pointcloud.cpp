/**
 * Create point cloud from depth image and camera intrinsic, and visualize it
 * @author: xiaotaw
 * @email: 
 * @date: 2020/06/29 22:32
 */
#include <iostream>
#include <memory> // for shared_ptr
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "cuda_runtime_api.h"
#include "vector_functions.hpp" // for make_ushort2

#include "common/json.hpp" // for nlohmann::json
#include "common/path.hpp"
#include "common/cuda_texture_surface.h"
#include "common/compute_vertex.h"

#if CV_VERSION_MAJOR >= 4
const int CV_ANYCOLOR = cv::IMREAD_ANYCOLOR;
const int CV_ANYDEPTH = cv::IMREAD_ANYDEPTH;
#else
const int CV_ANYCOLOR = CV_LOAD_IMAGE_ANYCOLOR;
const int CV_ANYDEPTH = CV_LOAD_IMAGE_ANYDEPTH;
#endif

//
std::atomic<bool> update(false);

/**
 * (xt) TODO: make it easier to save/read intrinsic matrix
 */
static cv::Mat ReadCameraIntrinsic(const std::string fn)
{
    cv::Mat camera_matrix = cv::Mat(3, 3, CV_64F);
    double *_data = reinterpret_cast<double *>(camera_matrix.data);

    std::ifstream inf;
    OpenFileOrExit(inf, fn);
    for (int i = 0; !inf.eof() && i < 9;)
    {
        int x = inf.peek();
        if (x == '[' || x == ',' || x == ' ' || x == ';')
        {
            inf.ignore();
            continue;
        }
        else if (x == ']')
        {
            break;
        }
        else
        {
            inf >> _data[i++];
        }
    }
    inf.close();
    return camera_matrix;
}

static int ReadImageList(const std::string image_list_fn, std::vector<std::string> &fn_list)
{
    std::ifstream inf;
    OpenFileOrExit(inf, image_list_fn);
    std::string line, timestamp, filename;
    while (getline(inf, line))
    {
        if (line[0] == '#')
        {
            continue;
        }
        size_t n = line.find_first_of(" ");
        size_t t = line.find_first_not_of("\r\n");
        if (n != std::string::npos && t != std::string::npos)
        {
            timestamp = line.substr(0, n);
            filename = line.substr(n + 1, t - n);
        }
        fn_list.push_back(filename);
    }
    return fn_list.size();
}

static bool IsValidVertex(float4 vertex)
{
    return (abs(vertex.x) > 1e-5) && (abs(vertex.y) > 1e-5) && (abs(vertex.z) > 1e-5);
}

static void HelpInfo()
{
    std::cout << "Usage 1: " << std::endl;
    std::cout << "    ./executable data_dir" << std::endl;
}

static void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void *nothing)
{
    if (event.keyDown())
    {
        //打印出按下的按键信息
        cout << event.getKeySym() << endl;
        if (event.getKeySym() == "n")
        {
            update = true;
        }
    }
}

int main(int argc, char **argv)
{
    std::cout << "TODO: " << std::endl;
    std::cout << "    1. apply bilateral filtering on depth image" << std::endl;
    std::cout << "    2. try to build point cloud for VolumnDeform dataset," << std::endl;
    std::cout << "       to check if mismatching between color and depth images" << std::endl;
    // args
    std::string data_dir;
    if (argc == 1)
    {
        data_dir = "/data/DATASETS/KinectDK/20200630/";
    }
    else if (argc == 2)
    {
        data_dir = argv[1];
    }
    else
    {
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
    auto img_size = cv::Size(j["depth image"]["width"], j["depth image"]["height"]);
    double fx = j["depth image"]["intrinsic"]["fx"];
    double fy = j["depth image"]["intrinsic"]["fy"];
    double cx = j["depth image"]["intrinsic"]["cx"];
    double cy = j["depth image"]["intrinsic"]["cy"];
    float4 camera_intrinsic_inv = make_float4(1.0 / fx, 1.0 / fy, cx, cy);

    // cuda resource
    CudaStream stream;
    auto depth_texture_surface = std::make_shared<CudaTextureSurface2D<ushort>>(img_size.height, img_size.width);
    auto vertex_texture_surface = std::make_shared<CudaTextureSurface2D<float4>>(img_size.height, img_size.width);
    // pagelock memory
    PagelockMemory depth_buffer_pagelock(sizeof(uint16_t) * img_size.area());
    PagelockMemory vertex_buffer_pagelock(sizeof(float4) * img_size.area());
    // necessary to sync after cudaMallocHost?
    CudaSafeCall(cudaDeviceSynchronize());
    CudaSafeCall(cudaGetLastError());

    // build point cloud
    auto point_cloud = pcl::PointCloud<pcl::PointXYZRGB>();
    auto point = pcl::PointXYZRGB(255, 255, 255);
    point_cloud.points.push_back(point);
    std::string cloud_name = "cloud";

    // visualize point cloud
    auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>("simple point cloud viewer");
    viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud.makeShared(), cloud_name);
    viewer->addCoordinateSystem(2.0, cloud_name, 0);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cloud_name);
    viewer->setCameraPosition(-493.926, -2538.05, -4271.43, 0.0244369, -0.907735, 0.418832, 0);
    viewer->registerKeyboardCallback(&keyboardEventOccurred, (void *)NULL);

    for (unsigned i = 0; (!viewer->wasStopped()) && (i < color_fn_list.size());)
    {

        viewer->spinOnce(100);
        if (update)
        {
            std::cout << i << " " << depth_fn_list[i] << std::endl;
            cv::Mat color_img = cv::imread(data_dir + color_fn_list[i], CV_ANYCOLOR | CV_ANYDEPTH);
            cv::Mat depth_img = cv::imread(data_dir + depth_fn_list[i], CV_ANYCOLOR | CV_ANYDEPTH);
            if (color_img.empty())
            {
                std::cout << "Error failed to read " << data_dir + color_fn_list[i] << std::endl;
                exit(EXIT_FAILURE);
            }
            if (depth_img.empty())
            {
                std::cout << "Error failed to read " << data_dir + depth_fn_list[i] << std::endl;
                exit(EXIT_FAILURE);
            }

            //std::cout << "color image type: " << color_img.type() << "; depth image type: " << depth_img.type() << std::endl;
            assert(color_img.type() == 16); // "color image type is expected to be CV_8UC3"
            assert(depth_img.type() == 2);  // "depth image type is expected to be CV_16UC1"
            assert(img_size == color_img.size());
            assert(img_size == depth_img.size());

            // memory to pagelock memory
            depth_buffer_pagelock.HostCopyFrom(static_cast<void *>(depth_img.data));
            // pagelock memory to device
            depth_buffer_pagelock.UploadToDevice(depth_texture_surface->d_array_, stream);

            // compute vertex from depth image on device
            ComputeVertex(make_ushort2(img_size.width, img_size.height),
                          camera_intrinsic_inv,
                          depth_texture_surface->texture_,
                          vertex_texture_surface->surface_,
                          stream);
            // download vertex from device to host
            vertex_buffer_pagelock.DownloadFromDevice(vertex_texture_surface->d_array_, stream);

            stream.Synchronize();
            CudaSafeCall(cudaGetLastError());

            cv::Mat vertex_map = cv::Mat(img_size, CV_32FC4);
            vertex_buffer_pagelock.HostCopyTo(vertex_map.data);

            point_cloud.clear();
            for (auto x = 0; x < img_size.width; x++)
            {
                for (auto y = 0; y < img_size.height; y++)
                {
                    auto vertex = vertex_map.at<float4>(y, x);
                    if (IsValidVertex(vertex))
                    {
                        auto color = color_img.at<cv::Vec3b>(y, x);
                        pcl::PointXYZRGB point;
                        point.x = vertex.x;
                        point.y = vertex.y;
                        point.z = vertex.z;
                        point.b = color[0];
                        point.g = color[1];
                        point.r = color[2];
                        point_cloud.points.push_back(point);
                    }
                }
            }
            std::cout << "point cloud size: " << point_cloud.points.size() << std::endl;

            viewer->updatePointCloud(point_cloud.makeShared(), cloud_name);
            update = false;
            i++;
        }
    }
    return 0;
}
