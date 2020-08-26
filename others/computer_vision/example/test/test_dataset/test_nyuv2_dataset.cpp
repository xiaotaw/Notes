#include <iostream>
#include "dataset/dataset_nyu_v2.h"


int main(){
    /*
    std::string raw_data_path = "/media/xt/8T/DATASETS/NYU_Depth_Dataset_V2/nyu_depth_v2_raw";
    NyuV2RawDataset raw(raw_data_path);
    std::cout << "The number of scenes in nyu_v2 dataset: " << raw.scenes_.size() << std::endl;

    auto a_scene = raw.scenes_[0];
    a_scene.ShowSceneInfo();
    */

    std::string labeled_data_path = "/media/xt/8T/DATASETS/NYU_Depth_Dataset_V2/nyu_depth_v2_labeled";
    NyuV2LabeledDataset labeled(labeled_data_path);
    std::cout << "The number of depth images: " << labeled.depth_filenames_.size() << " ";
    std::cout << "The number of color images: " << labeled.color_filenames_.size() << std::endl;
    
    labeled.StartPreloadThread();

    std::cout << "1. " << labeled.color_filenames_[0] << "\t" << labeled.depth_filenames_[0] << std::endl;
    cv::Mat depth_img, color_img;
    labeled.FetchNextFrame(depth_img, color_img);

    cv::imshow("depth", depth_img);
    cv::imshow("color", color_img);
    cv::waitKey();
    return 0;
}