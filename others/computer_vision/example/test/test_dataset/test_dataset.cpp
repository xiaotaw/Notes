#include <iostream>
#include "dataset/nyu_v2.h"


int main(){
    std::string data_path = "/media/xt/8T/DATASETS/NYU_Depth_Dataset_V2/nyu_depth_v2_raw";
    NyuV2RawDataset raw(data_path);
    std::cout << "The number of scenes in nyu_v2 dataset: " << raw.scenes_.size() << std::endl;

    auto a_scene = raw.scenes_[0];
    a_scene.ShowSceneInfo();

    data_path = "/media/xt/8T/DATASETS/NYU_Depth_Dataset_V2/nyu_depth_v2_labeled";
    NyuV2LabeledDataset labeled(data_path);
    std::cout << "The number of depth images: " << labeled.depth_filenames_.size() << " ";
    std::cout << "The number of color images: " << labeled.color_filenames_.size() << std::endl;
    return 0;
}