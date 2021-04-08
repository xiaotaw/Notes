#include "dataset/dataset_nyu_v2.h"
#include <iostream>

int main() {
  /*
  std::string raw_data_path =
  "/media/xt/8T/DATASETS/NYU_Depth_Dataset_V2/nyu_depth_v2_raw"; NyuV2RawDataset
  raw(raw_data_path); std::cout << "The number of scenes in nyu_v2 dataset: " <<
  raw.scenes_.size() << std::endl;

  auto a_scene = raw.scenes_[0];
  a_scene.ShowSceneInfo();
  */

  std::string labeled_data_path =
      "/media/xt/8T/DATASETS/NYU_Depth_Dataset_V2/nyu_depth_v2_labeled";
  NyuV2LabeledDataset labeled(labeled_data_path);
  std::cout << "The number of depth images: " << labeled.num_depth() << " ";
  std::cout << "The number of color images: " << labeled.num_color()
            << std::endl;

  cv::Mat depth_img, color_img;
  labeled.FetchNextFrame(depth_img, color_img);

  std::cout << "1. show the first depth and color image" << std::endl;
  cv::imshow("depth", depth_img);
  cv::imshow("color", color_img);
  cv::waitKey();
  return 0;
}