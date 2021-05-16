/**
 * @file motion_to_color.h
 * @author xiaotaw (you@domain.com)
 * @brief convert motion to color,
 *        ref: https://blog.csdn.net/zouxy09/article/details/8683859
 *        ref: http://members.shaw.ca/quadibloc/other/colint.htm
 *        ref: http://vision.middlebury.edu/flow/data/
 * @version 0.1
 * @date 2021-05-9
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <opencv2/opencv.hpp>
#include <vector>

void MakeColorWheel(std::vector<cv::Scalar> &colorwheel);

void MotionToColor(const cv::Mat &flow, cv::Mat &color);