/**
 * A 3D visualizer for RGBD data.
 * @author: xiaotaw
 * @email:
 * @date: 2020/08/21 16:22
 */
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

class PCLVis {
public:
  pcl::PointCloud<pcl::PointXYZRGB> point_cloud_;
  pcl::visualization::PCLVisualizer::Ptr viewer_;
  std::string cloud_name_;

  struct float4 {
    float x, y, z, w;
  };

  PCLVis();

  void UpdatePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

  void UpdatePointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

  void UpdatePointCloud(const cv::Mat &vertex_map, const cv::Mat &color_map);

  static bool IsValidVertex(float4 vertex);

  static void
  keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
                        void *nothing);

  static std::atomic<bool> update;
};
