/**
 * RGBD dataset
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/24 17:06
 */
#pragma once
#include <list>
#include <mutex>
#include <memory>
#include <string>
#include <thread>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include "camera_params.hpp"

// TODO: is it safe?
namespace fs = boost::filesystem;

/** Interface class RgbdDataset
 * 1. Do NOT use RgbdDataset directly, instead, you should derive your own dataset class.
 * 2. In derived Dataset class, remember to initialize necessary data members, 
 *    e.g. depth_filenames_, color_filenames_, color_camera_params_, 
 * 3. Before calling FetchNextFrame, remember to StartPreloadThread.
 */
class RgbdDataset
{
public:
    // info
    std::string data_path_;
    std::vector<std::string> depth_filenames_;
    std::vector<std::string> color_filenames_;
    CameraParams color_camera_params_;
    CameraParams depth_camera_params_;

    // if depth and color images are synchronized
    bool is_synced_ = true;
    // if transpose image
    bool is_transpose_ = false;

    using Ptr = std::shared_ptr<RgbdDataset>;

    // ctor
    RgbdDataset() = default;
    RgbdDataset(const std::string &data_path) : data_path_(data_path) {}
    RgbdDataset(const std::string &data_path, bool is_transpose) : data_path_(data_path), is_transpose_(is_transpose) {}
    // dtor
    ~RgbdDataset() = default;

    void StartPreloadThread();

    // interface func, Note depth and color images must be synced
    size_t FetchNextFrame(cv::Mat &depth_img, cv::Mat &color_img);

protected:
    /** preload one frame in a separate thread
     * 1. whenever list.empty(), try to load the next depth and color images.
     * 2. using lock to deal with race condition.
     */
    std::list<cv::Mat> color_images_;
    std::list<cv::Mat> depth_images_;
    size_t img_index_ = -1;
    std::mutex mutex_;
    std::thread thread_;

    void PreloadImages();
};