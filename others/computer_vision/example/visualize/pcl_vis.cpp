/**
 * visualizer
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/22 07:50
 */
#include "pcl_vis.h"

PCLVis::PCLVis()
{
    // build point cloud
    point_cloud_ = pcl::PointCloud<pcl::PointXYZRGB>();
    auto point = pcl::PointXYZRGB(255, 255, 255);
    point_cloud_.points.push_back(point);
    cloud_name_ = "cloud";

    // visualize point cloud
    viewer_ = boost::make_shared<pcl::visualization::PCLVisualizer>("simple point cloud viewer");
    viewer_->addPointCloud<pcl::PointXYZRGB>(point_cloud_.makeShared(), cloud_name_);
    viewer_->addCoordinateSystem(2.0, cloud_name_, 0);
    viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cloud_name_);
    viewer_->setCameraPosition(-493.926, -2538.05, -4271.43, 0.0244369, -0.907735, 0.418832, 0);
    viewer_->registerKeyboardCallback(&keyboardEventOccurred, (void *)NULL);
}

void PCLVis::UpdatePointCloud(const cv::Mat &vertex_map, const cv::Mat &color_map)
{
    point_cloud_.clear();
    auto img_size = vertex_map.size();
    for (auto y = 0; y < img_size.height; y++)
    {
        for (auto x = 0; x < img_size.width; x++)
        {
            auto vertex = vertex_map.at<float4>(y, x);
            if (IsValidVertex(vertex))
            {
                auto color = color_map.at<cv::Vec3b>(y, x);
                pcl::PointXYZRGB point;
                point.x = vertex.x;
                point.y = vertex.y;
                point.z = vertex.z;
                point.b = color[0];
                point.g = color[1];
                point.r = color[2];
                point_cloud_.points.push_back(point);
            }
        }
    }

    std::cout << "point cloud size: " << point_cloud_.points.size() << std::endl;

    viewer_->updatePointCloud(point_cloud_.makeShared(), cloud_name_);
    update = false;
}

bool PCLVis::IsValidVertex(float4 vertex)
{
    return (abs(vertex.x) > 1e-5) && (abs(vertex.y) > 1e-5) && (abs(vertex.z) > 1e-5);
}

void PCLVis::keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void *nothing)
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

std::atomic<bool> PCLVis::update(false);