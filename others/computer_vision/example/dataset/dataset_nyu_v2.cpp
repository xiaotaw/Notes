/**
 * nyu depth v2 dataset
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/18 07:23
 */
#include <numeric>
#include <iostream>
#include <algorithm>
#include <boost/lexical_cast.hpp>                    // for boost::lexical_cast
#include <boost/algorithm/string/split.hpp>          // for boost::split
#include <boost/algorithm/string/predicate.hpp>      // for boost::starts_with
#include <boost/algorithm/string/classification.hpp> // for boost::is_any_of
#include "dataset_nyu_v2.h"

namespace fs = boost::filesystem;

NyuV2RawDataset::NyuV2RawDataset(const std::string &data_path) : RgbdDataset(data_path)
{
    is_synced_ = false;
    FindImageList();
    SyncDepthColorImage();
    is_synced_ = true;
}

int NyuV2RawDataset::FindImageList()
{
    if (!fs::exists(data_path_))
    {
        std::cout << "[Error] path not exists: " << data_path_ << std::endl;
        return 0;
    }
    fs::path d_path(data_path_);
    fs::directory_iterator end_iter;
    for (fs::directory_iterator iter(d_path); iter != end_iter; iter++)
    {
        if (fs::is_regular_file(*iter))
        {
            const std::string str = iter->path().filename().string();
            if (str[0] == 'a')
            {
                // accelerometer data
                raw_accel_filenames_.push_back(str);
            }
            else if (str[0] == 'd')
            {
                // depth image
                raw_depth_filenames_.push_back(str);
            }
            else if (str[0] == 'r')
            {
                // color image
                raw_color_filenames_.push_back(str);
            }
            else
            {
                // INDEX.txt seems broken
            }
        }
    }
    std::sort(raw_depth_filenames_.begin(), raw_depth_filenames_.end());
    std::sort(raw_color_filenames_.begin(), raw_color_filenames_.end());
    std::sort(raw_accel_filenames_.begin(), raw_accel_filenames_.end());
    return raw_depth_filenames_.size() + raw_color_filenames_.size() + raw_accel_filenames_.size();
}

int NyuV2RawDataset::SyncDepthColorImage(bool verbose)
{
    // timestamps
    std::vector<double> dt, ct, at;
    for (auto it : raw_depth_filenames_)
    {
        dt.push_back(FilenameToTimestamp(it));
    }
    for (auto it : raw_color_filenames_)
    {
        ct.push_back(FilenameToTimestamp(it));
    }

    std::vector<double> diff;

    // for each depth filename, find the corresponding color filename
    // the difference of timestamps when depth and color images were taken should be less than TIME_ERROR_LIMIT
    for (unsigned i = 0; i < raw_depth_filenames_.size(); i++)
    {
        std::string d_filename = raw_depth_filenames_[i];
        double d = dt[i];
        double d_prev = i >= 1 ? dt[i - 1] : d - TIME_ERROR_LIMIT;
        double d_nect = i + 1 < dt.size() ? dt[i + 1] : d + TIME_ERROR_LIMIT;

        double min_diff = d;
        int min_index = -1;
        std::string c_filename;
        for (unsigned j = 0; j < raw_color_filenames_.size(); j++)
        {
            c_filename = raw_color_filenames_[j];
            double c = ct[j];
            if (c < d_prev)
            {
                continue;
            }
            else if (c <= d_nect)
            {
                double temp = std::abs(d - c);
                if (temp <= min_diff)
                {
                    min_diff = temp;
                    min_index = j;
                }
                else
                {
                    break;
                }
            }
            else
            {
                break;
            }
        }

        if (min_index != -1 && min_diff < TIME_ERROR_LIMIT)
        {
            depth_filenames_.push_back(d_filename);
            color_filenames_.push_back(c_filename);
            diff.push_back(min_diff);
        }
    }
    if (verbose)
    {
        if (diff.size() > 0)
        {
            double sum = std::accumulate(std::begin(diff), std::end(diff), 0.0);
            double mean = sum / diff.size();
            std::cout << "Sync depth and color for scene: " << data_path_ << std::endl;
            std::cout << "  depth images: " << raw_depth_filenames_.size();
            std::cout << "; color images: " << raw_color_filenames_.size() << std::endl;
            std::cout << "  synced images: " << diff.size() << std::endl;
            std::cout << "  mean time diff: " << mean << std::endl;
        }
        else
        {
            std::cout << "[Error] no synced images for scene: " << data_path_ << std::endl;
        }
    }
    return depth_filenames_.size();
}

void NyuV2RawDataset::ShowCurrentSceneInfo() const
{
    std::cout << "In " << data_path_ << ": " << std::endl;
    std::cout << "    depth images: " << raw_depth_filenames_.size() << std::endl;
    std::cout << "    color images: " << raw_color_filenames_.size() << std::endl;
    std::cout << "    accel files: " << raw_accel_filenames_.size() << std::endl;
    std::cout << "    synced images: " << depth_filenames_.size() << std::endl;
}

double NyuV2RawDataset::FilenameToTimestamp(const std::string &filename)
{
    std::vector<std::string> tokens;
    boost::split(tokens, filename, boost::is_any_of("-"));
    if (tokens.size() != 3)
    {
        std::cout << "[Error] cannot extract timestamp from " << filename << std::endl;
        return -1;
    }
    else
    {
        return boost::lexical_cast<double>(tokens[1]);
    }
}

// NyuV2RawDataset::NyuV2RawDataset(std::string data_path) : data_path_(data_path)
// {
//     FindAllScenes();
// }

int NyuV2RawDataset::FindAllScenes(const std::string &root_path, std::vector<std::string> &scenes)
{
    fs::path d_path(root_path);
    fs::directory_iterator end_iter;
    for (fs::directory_iterator iter(d_path); iter != end_iter; iter++)
    {
        if (fs::is_directory(*iter))
        {
            scenes.push_back(iter->path().filename().string());
        }
    }
    return scenes.size();
}

NyuV2LabeledDataset::NyuV2LabeledDataset(std::string data_path) : RgbdDataset(data_path)
{
    FindAllImages();
    // nyu v2 标注数据集是transposed，读取图片时需转置。
    is_transpose_ = true;
    camera_params_ = CameraParams(
        480,
        640,
        5.1885790117450188e+02,
        5.1946961112127485e+02,
        3.2558244941119034e+02,
        2.5373616633400465e+02);
}

int NyuV2LabeledDataset::FindAllImages()
{
    fs::path d_path(data_path_);
    fs::directory_iterator end_iter;
    for (fs::directory_iterator iter(d_path); iter != end_iter; iter++)
    {
        if (fs::is_regular_file(*iter))
        {
            std::string filename = iter->path().filename().string();
            if (boost::starts_with(filename, "depth"))
            {
                depth_filenames_.push_back(filename);
            }
            else if (boost::starts_with(filename, "color"))
            {
                color_filenames_.push_back(filename);
            }
            else
            {
            }
        }
    }
    std::sort(color_filenames_.begin(), color_filenames_.end());
    std::sort(depth_filenames_.begin(), depth_filenames_.end());
    return depth_filenames_.size() + color_filenames_.size();
}