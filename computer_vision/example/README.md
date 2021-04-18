## 3D reconstruction based on RGBD camera
1. using gpu to speedup, as the massive pointcloud and pixels
2. be aware of dynamic objects
 
## Image process
### bilateral filtering of depth image 

### compute vertex map from depth image, i.e. point cloud

### compute normal map from vertex map

## ICP


## Speed test
- timmer


## Plan
| No  |  type        | item                       | deadline | status |
| --- | ---          | --------------             | -------- | ------ |
| 1   | basic func   | cuda container and texture | 2021.4   | 100%   | 
| 1   | image process| bilateral filter           | 2021.4   | 50%    |
| 2   | 3D           | icp(cuda) version          | 2021.4   | x      |
| 3   | 2D           | tv l1                      | 2021.5   | x      |