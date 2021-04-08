#include "common/opencv_cross_version.h"
#include "common/static_image_properties.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

const std::string dir =
    "/home/xt/Documents/data/DATASETS/VolumeDeform/boxing/data/";

int main() {
  int i = 1;
  char fn[50];
  sprintf(fn, "frame-%06d.depth.png", i);
  cv::Mat depth_img =
      cv::imread(dir + std::string(fn), CV_ANYCOLOR | CV_ANYDEPTH);

  std::vector<Pixel> buffer;

  std::pair<uint16_t, uint16_t> expectedValueRange = {(uint16_t)200,
                                                      (uint16_t)800};
  ColorizeDepthImage(depth_img, DepthPixelColorizer::ColorizeBlueToRed,
                     expectedValueRange, &buffer);

  cv::Mat colorized_depth_img = cv::Mat(
      depth_img.size().height, depth_img.size().width, CV_8UC4, buffer.data());
  cv::imshow("colorized", colorized_depth_img);
  cv::waitKey(0);

  return 0;
}