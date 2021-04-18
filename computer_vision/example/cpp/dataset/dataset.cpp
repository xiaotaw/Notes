/**
 * @file dataset.cpp
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-04-11
 * @copyright Copyright (c) 2021
 */
#include "dataset/dataset.h"
#include "common/logging.h"

RgbdDataset::Ptr CreateDataset(const std::string &data_type,
                               const std::string &data_dir) {
  RgbdDataset::Ptr dataset;
  std::string data_dype_ = data_type;
  // 使用全局函数tolower，避免std命名空间中因重载导致的问题
  std::transform(data_type.begin(), data_type.end(), data_dype_.begin(),
                 ::tolower);
  if (data_dype_ == "azure_kinect") {
    dataset = std::make_shared<AzureKinectDataset>(data_dir);
  } else if (data_dype_ == "nyu_v2_labeled") {
    dataset = std::make_shared<NyuV2LabeledDataset>(data_dir);
  } else if (data_dype_ == "nyu_v2_raw") {
    dataset = std::make_shared<NyuV2RawDataset>(data_dir);
  } else {
    LOG(FATAL) << "Unknown data_type: " << data_dype_;
  }
  return dataset;
}
