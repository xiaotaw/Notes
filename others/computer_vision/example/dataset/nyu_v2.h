/**
 * nuy depth v2 dataset
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/18 07:22
 */
#pragma once
#include <string>
#include <boost/filesystem.hpp>

#include "dataset/camera_params.h"

//
using SyncPair = std::pair<std::string, std::string>;

class NyuV2Scene
{
public:
    std::string scene_path_;
    std::vector<std::string> depth_filenames_;
    std::vector<std::string> color_filenames_;
    std::vector<std::string> accel_filenames_;
    std::vector<SyncPair> synced_;

    NyuV2Scene(std::string scene_path);

    /**
     * Desc:
     * Input: scene_path_
     * Output: depth_filenames_, color_filenames_, accel_filenames_
     */
    int FindImageList();

    static constexpr double TIME_ERROR_LIMIT = 0.05;

    /**
     * Desc: sync depth and color images for a scene, just follow nyu_v2_sync_color_depth_image.py
     *     However, in fact, we cannot sync depth and color, as color and depth images are recoreded independently.
     *     NOTES: this function NEED optimization.
     * Input: depth_filenames_, color_filenames_
     * Output: synced_
     */
    int SyncDepthColorImage(bool verbose = false);

    /**
     * Desc:
     */
    void ShowSceneInfo();

private:
    // extract timestamp from filename, e.g. d-1315332943.792050-3485245013.pgm -> 1315332943.792050
    double FilenameToTimestamp(std::string filename);
};

class NyuV2RawDataset
{
public:
    std::string data_path_;
    std::vector<NyuV2Scene> scenes_;

    // ctor
    NyuV2RawDataset(std::string data_path);

private:
    /**
     * Desc:
     * Input: data_path_
     * Output: scenes_
     */
    int FindAllScenes();
};

/**
 * Desc: 
 *     The Nyu V2 labeled dataset:
 *     |-- data_path_
 *         |-- color-0.png
 *         |-- depth-0.png
 *         ...
 *     Notes: The original labeled dataset was in a *.mat file. 
 *            and the color and depth files were extracted out by using mat_to_png.py.
 */

class NyuV2LabeledDataset
{
public:
    std::string data_path_;
    std::vector<std::string> depth_filenames_;
    std::vector<std::string> color_filenames_;

    // the color camera's 
    CameraParams camera_params_;

    // ctor
    NyuV2LabeledDataset(std::string data_path);

    /**
     * Desc:
     * Input: data_path_
     * Output: depth_filenames_, color_filenames_
     */
    int FindAllImages();



};