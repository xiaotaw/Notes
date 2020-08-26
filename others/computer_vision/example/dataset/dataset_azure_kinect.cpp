/**
 * Azure Kinect dataset
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/24 17:43
 */
#include <unistd.h>
#include "dataset_azure_kinect.h"
#include "common/json.hpp"
#include "common/path.hpp"
#include "utils/logging.h"
#include "utils/opencv_cross_version.h"

AzureKinectDataset::AzureKinectDataset(const std::string &data_path) : RgbdDataset(data_path)
{
    //
    ReadImageList((fs::path(data_path_) / "depth.txt").string(), depth_filenames_);
    ReadImageList((fs::path(data_path_) / "rgb.txt").string(), color_filenames_);

    // dataset info in json format
    std::ifstream inf;
    OpenFileOrExit(inf, (fs::path(data_path_) / "image_info.json").string());
    nlohmann::json j = nlohmann::json::parse(inf);
    inf.close();
    auto img_size = cv::Size(j["depth image"]["width"], j["depth image"]["height"]);
    double fx = j["depth image"]["intrinsic"]["fx"];
    double fy = j["depth image"]["intrinsic"]["fy"];
    double cx = j["depth image"]["intrinsic"]["cx"];
    double cy = j["depth image"]["intrinsic"]["cy"];
    camera_params_ = CameraParams(img_size.height, img_size.width, fx, fy, cx, cy);

}

int AzureKinectDataset::ReadImageList(const std::string &image_list_fn, std::vector<std::string> &fn_list)
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
