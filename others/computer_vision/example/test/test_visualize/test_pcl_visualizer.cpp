/**
 * test pcl_visualizer
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/26 10:52
 */
#include <cctype> // for std::tolower
#include <string>
#include <iostream>
#include <algorithm>
#include "common/time_logger.h"
#include "common/image_proc.h"
#include "dataset/dataset_azure_kinect.h"
#include "dataset/dataset_nyu_v2.h"
#include "visualize/pcl_vis.h"

static void HelpInfo()
{
    std::cout << "Usage 1: " << std::endl;
    std::cout << "    ./executable data_dir data_type" << std::endl;
    std::cout << "        data_type could be: azure_kinect, nyu_v2_raw, nyu_v2_labeled" << std::endl;
}

int main(int argc, char **argv)
{
    std::cout << "TODO: " << std::endl;
    std::cout << "    1. apply bilateral filtering on depth image" << std::endl;
    std::cout << "    2. try to build point cloud for VolumnDeform dataset," << std::endl;
    std::cout << "       to check if mismatching between color and depth images" << std::endl;
    // args
    std::string data_dir, data_type;
    if (argc == 1)
    {
        data_dir = "/media/xt/8T/DATASETS/KinectDkDataset/20200701/";
        data_type = "azure_kinect";
    }
    else if (argc == 3)
    {
        data_dir = argv[1];
        data_type = argv[2];
        // 使用全局函数tolower，避免std命名空间中因重载导致的问题
        std::transform(data_type.begin(), data_type.end(), data_type.begin(), ::tolower);
    }
    else
    {
        HelpInfo();
        exit(EXIT_FAILURE);
    }

    RgbdDataset::Ptr dataset;
    if (data_type == "azure_kinect")
    {
        dataset = std::make_shared<AzureKinectDataset>(data_dir);
    }
    else if (data_type == "nyu_v2_labeled")
    {
        dataset = std::make_shared<NyuV2LabeledDataset>(data_dir);
    }
    else if (data_type == "nyu_v2_raw")
    {
        dataset = std::make_shared<NyuV2RawDataset>(data_dir);
    }
    else
    {
        HelpInfo();
        exit(EXIT_FAILURE);
    }

    // preload images
    dataset->StartPreloadThread();

    ImageProcessor image_proc(dataset->camera_params_);

    PCLVis vis;

    TimeLogger::printTimeLog("start");

    for (unsigned i = 0; (!(vis.viewer_)->wasStopped()) && (i < dataset->color_filenames_.size());)
    {
        (vis.viewer_)->spinOnce(1);
        if (PCLVis::update)
        {
            TimeLogger::printTimeLog("new update");
            std::cout << i << " " << dataset->color_filenames_[i];
            std::cout << " " << dataset->depth_filenames_[i] << std::endl;
            cv::Mat color_img, depth_img;

            dataset->FetchNextFrame(depth_img, color_img);

            //std::cout << "color image type: " << color_img.type() << "; depth image type: " << depth_img.type() << std::endl;
            assert(color_img.type() == 16); // "color image type is expected to be CV_8UC3"
            assert(depth_img.type() == 2);  // "depth image type is expected to be CV_16UC1"
            auto img_size = depth_img.size();

            TimeLogger::printTimeLog("fetch images");

            cv::Mat vertex_map = cv::Mat(img_size, CV_32FC4);
            image_proc.BuildVertexMap(depth_img, vertex_map);

            TimeLogger::printTimeLog("compute vertex");

            vis.UpdatePointCloud(vertex_map, color_img);

            TimeLogger::printTimeLog("compact and update point cloud");

            PCLVis::update = false;
            i++;
            TimeLogger::printTimeLog("Loop Done");
        }
    }
    return 0;
}
