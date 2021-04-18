/**
 * @file dataset.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-04-11
 * @copyright Copyright (c) 2021
 */
#pragma once
#include "dataset/dataset_azure_kinect.h"
#include "dataset/dataset_nyu_v2.h"
#include <algorithm>
#include <cctype> // for std::tolower
#include <string>

RgbdDataset::Ptr CreateDataset(const std::string &data_type,
                               const std::string &data_dir);
