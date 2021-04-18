/**
 * test AzureKinectDK dataset
 * @author: xiaotaw
 * @email:
 * @date: 2020/08/25 09:34
 */

#include "dataset/dataset_azure_kinect.h"
#include <iostream>

int main() {
  std::string d_path = "/media/xt/8T/DATASETS/KinectDkDataset/20200701/";
  AzureKinectDataset d(d_path);

  cv::Mat d_img, c_img;
  for (int i = 0; i < 100; i++) {
    d.FetchNextFrame(d_img, c_img);
    cv::imshow("color", c_img);
    cv::waitKey();
  }
  return 0;
}