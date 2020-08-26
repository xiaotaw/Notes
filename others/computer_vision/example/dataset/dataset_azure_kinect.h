/**
 * Azure Kinect dataset
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/24 17:06
 */
#pragma once
#include "dataset_base.h"

class AzureKinectDataset : public RgbdDataset
{
public:
    AzureKinectDataset(const std::string &data_path);


    using Ptr = std::shared_ptr<AzureKinectDataset>;

private:
    int ReadImageList(const std::string &image_list_fn, std::vector<std::string> &fn_list);
};