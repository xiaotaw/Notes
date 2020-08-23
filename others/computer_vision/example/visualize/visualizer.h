/**
 * A 3D visualizer for RGBD data.
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/21 16:22
 */
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "gl_render.h"

class Visualizer{
public:
    GLRender render_;
    Visualizer(){ 
        render_.InitWindow();
        //render_.InitShader();
    }
    using Vector3d = Eigen::Vector3d;

    void Draw(std::vector<Vector3d> points, std::vector<Vector3d> colors);
    
};

