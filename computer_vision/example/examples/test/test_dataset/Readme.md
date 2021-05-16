
nyu v2的测试集可以运行测试test_dataset，但不适合测试img_proc。

```bash
# azure kinect dk dataset
./test_dataset /media/xt/8T/DATASETS/KinectDkDataset/20200701 azure_kinect

# nyu v2 labeled dataset
./test_dataset /media/xt/8T/DATASETS/NYU_Depth_Dataset_V2/nyu_depth_v2_labeled nyu_v2_labeled

# nyu v2 raw dataset
./test_dataset /media/xt/8T/DATASETS/NYU_Depth_Dataset_V2/nyu_depth_v2_raw/bedroom_0001 nyu_v2_raw
```