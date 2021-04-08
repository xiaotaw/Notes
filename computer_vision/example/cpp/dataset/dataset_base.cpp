/**
 * RGBD dataset
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/24 17:43
 */
#include <unistd.h>
#include "dataset_base.h"
#include "json.hpp"
#include "common/safe_open.hpp"
#include "common/logging.h"
#include "common/opencv_cross_version.h"

void RgbdDataset::PreloadImages()
{
    while (1)
    {
        bool should_load = true;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            bool d_empty = depth_images_.empty();
            bool c_empty = color_images_.empty();
            if (d_empty != c_empty)
            {
                LOG(FATAL) << "depth images and color images count differently" << std::endl;
                break;
            }
            should_load = d_empty && c_empty;
        }
        if (should_load)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            img_index_++;

            if (img_index_ >= depth_filenames_.size() || img_index_ >= color_filenames_.size())
            {
                break;
            }

            std::string d_filename = (fs::path(data_path_) / depth_filenames_[img_index_]).string();
            cv::Mat d_img = cv::imread(d_filename, CV_ANYCOLOR | CV_ANYDEPTH);
            if (d_img.empty())
            {
                LOG(FATAL) << "Read " << d_filename << " failed" << std::endl;
                break;
            }
            std::string c_filename = (fs::path(data_path_) / color_filenames_[img_index_]).string();
            cv::Mat c_img = cv::imread(c_filename, CV_ANYCOLOR | CV_ANYDEPTH);
            if (d_img.empty())
            {
                LOG(FATAL) << "Read " << c_filename << " failed" << std::endl;
                break;
            }
            if (is_transpose_){
                cv::transpose(d_img, d_img);
                cv::transpose(c_img, c_img);
            }
            std::cout << "depth img size: " << d_img.size() << std::endl;
            depth_images_.emplace_back(std::move(d_img));
            color_images_.emplace_back(std::move(c_img));
        }
        else
        {
            usleep(3000);
        }
    }
}

void RgbdDataset::StartPreloadThread()
{
    // start Preload thread
    thread_ = std::thread(&RgbdDataset::PreloadImages, this);
    thread_.detach();
}

size_t RgbdDataset::FetchNextFrame(cv::Mat &depth_img, cv::Mat &color_img)
{
    assert(is_synced_);
    for (int i = 0; i < 3; i++) // try 3 times
    {
        bool is_empty = true;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            bool d_empty = depth_images_.empty();
            bool c_empty = color_images_.empty();
            is_empty = d_empty || c_empty;
        }
        if (is_empty)
        {
            usleep(3000);
        }
        else
        {
            std::unique_lock<std::mutex> lock(mutex_);
            depth_img = depth_images_.front();
            color_img = color_images_.front();
            depth_images_.pop_front();
            color_images_.pop_front();
            return img_index_;
        }
    }
    LOG(FATAL) << "time out to fetch next frame" << std::endl;
    return -1;
}