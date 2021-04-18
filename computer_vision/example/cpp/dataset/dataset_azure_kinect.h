/**
 * @file dataset_azure_kinect.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2020-08-24
 * @copyright Copyright (c) 2021
 */
#pragma once
#include "dataset_base.h"

class AzureKinectDataset : public RgbdDataset {
public:
  AzureKinectDataset(const std::string &data_path);

  using Ptr = std::shared_ptr<AzureKinectDataset>;

private:
  int ReadImageList(const std::string &image_list_fn,
                    std::vector<std::string> &fn_list);
};