/**
 * nuy depth v2 dataset
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/18 07:22
 */
#pragma once
#include <string>
#include <boost/filesystem.hpp>

#include "dataset_base.h"

class NyuV2RawDataset : public RgbdDataset
{
public:
    NyuV2RawDataset(const std::string &data_path);

    using Ptr = std::shared_ptr<NyuV2RawDataset>;

    void ShowCurrentSceneInfo() const;
    
    /**
     * Desc:
     * Input: root_path_
     * Output: scenes_
     */
    static int FindAllScenes(const std::string &root_path, std::vector<std::string> &scenes);

    static constexpr double TIME_ERROR_LIMIT = 0.05;

private:
    // not synced filenames
    std::vector<std::string> raw_color_filenames_;
    std::vector<std::string> raw_depth_filenames_;
    std::vector<std::string> raw_accel_filenames_;

    // extract timestamp from filename, e.g. d-1315332943.792050-3485245013.pgm -> 1315332943.792050
    double FilenameToTimestamp(const std::string &filename);

    /**
     * Desc:
     * Input: scene_path_
     * Output: depth_filenames_, color_filenames_, accel_filenames_
     */
    int FindImageList();    
    
    /**
     * Desc: sync depth and color images for a scene, just follow nyu_v2_sync_color_depth_image.py
     *     However, in fact, we cannot sync depth and color, as color and depth images are recoreded independently.
     *     NOTES: this function NEED optimization.
     * Input: depth_filenames_, color_filenames_
     * Output: synced_
     */
    int SyncDepthColorImage(bool verbose = false);
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

class NyuV2LabeledDataset : public RgbdDataset
{
public:
    // ctor
    NyuV2LabeledDataset(std::string data_path);

    using Ptr = std::shared_ptr<NyuV2LabeledDataset>;

private:
    /**
     * Desc:
     * Input: data_path_
     * Output: depth_filenames_, color_filenames_
     */
    int FindAllImages();



};