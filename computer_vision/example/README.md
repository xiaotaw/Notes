## 3D reconstruction based on RGBD camera
1. using gpu to speedup, as the massive pointcloud and pixels
2. be aware of dynamic objects

## Dependency
Eigen >= 3.3.9

## Compilation
```bash
mkdir build && cd build
cmake .. -DOpenCV_DIR=/usr/local/opencv-4.2.0/lib/cmake/opencv4
```

## Image process
### bilateral filtering of depth image 

### compute vertex map from depth image, i.e. point cloud

### compute normal map from vertex map

## ICP


## Speed test
- timmer


## Plan
| No  |  type        | item                                           | deadline | status |
| --- | ---          | --------------                                 | -------- | ------ |
| 1   | basic func   | cuda container and texture                     | 2021.4   | 100%   | 
| 2   | image process| bilateral filter                               | 2021.4   | 50% (done but not tested yet) |
| 3   | 3D           | icp(cuda) version                              | 2021.4   | 100% (finished at 2021-05-16)   |
| 4   | 2D           | optical flow + tv l1                           | 2021.5   | 0%     |
| 5   | 3D           | 'icp + tv l1': define the question, math model | 2021.6   | 0%     |
| 6   | 3D           | 'icp + tv l1': implementation                  | 2021.7   | 0%     |
