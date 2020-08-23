/**
 * visualizer
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/22 07:50
 */
#include "visualizer.h"

void Visualizer::Draw(std::vector<Vector3d> points, std::vector<Vector3d> colors)
{
    if (points.size() != colors.size())
    {
        std::cout << "[ERROR]" << std::endl;
    }
}