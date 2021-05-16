/**
 * @file test_dataset.cpp
 * @author xiaotaw (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2021-04-24
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include "dataset/dataset.h"


static void HelpInfo() {
  std::cout << "Usage 1: " << std::endl;
  std::cout << "    ./executable data_dir data_type" << std::endl;
  std::cout << "data_type could be: azure_kinect, nyu_v2_raw, nyu_v2_labeled"
            << std::endl;
}

int main(int argc, char **argv) {
  std::string data_dir, data_type;
  if (argc == 3) {
    data_dir = argv[1];
    data_type = argv[2];
  } else {
    HelpInfo();
    exit(EXIT_FAILURE);
  }

  RgbdDataset::Ptr dataset = CreateDataset(data_type, data_dir);

  cv::Mat d_img, c_img;
  for (int i = 0; i < 100; i++) {
    dataset->FetchNextFrame(d_img, c_img);
    cv::imshow("color", c_img);
    cv::imshow("depth", d_img);
    cv::waitKey();
  }
  return 0;

}