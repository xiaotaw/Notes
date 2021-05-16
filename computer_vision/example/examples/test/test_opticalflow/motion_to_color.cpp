/**
 * @file motion_to_color.cpp
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-05-9
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "motion_to_color.h"

#define UNKNOWN_FLOW_THRESH 1e9

void MakeColorWheel(std::vector<cv::Scalar> &colorwheel) {
  int RY = 15;
  int YG = 6;
  int GC = 4;
  int CB = 11;
  int BM = 13;
  int MR = 6;

  for (int i = 0; i < RY; i++)
    colorwheel.push_back(cv::Scalar(255, 255 * i / RY, 0));
  for (int i = 0; i < YG; i++)
    colorwheel.push_back(cv::Scalar(255 - 255 * i / YG, 255, 0));
  for (int i = 0; i < GC; i++)
    colorwheel.push_back(cv::Scalar(0, 255, 255 * i / GC));
  for (int i = 0; i < CB; i++)
    colorwheel.push_back(cv::Scalar(0, 255 - 255 * i / CB, 255));
  for (int i = 0; i < BM; i++)
    colorwheel.push_back(cv::Scalar(255 * i / BM, 0, 255));
  for (int i = 0; i < MR; i++)
    colorwheel.push_back(cv::Scalar(255, 0, 255 - 255 * i / MR));
}

void MotionToColor(const cv::Mat &flow, cv::Mat &color) {
  if (color.empty())
    color.create(flow.rows, flow.cols, CV_8UC3);

  static std::vector<cv::Scalar> colorwheel; // Scalar r,g,b
  if (colorwheel.empty())
    MakeColorWheel(colorwheel);

  // determine motion range:
  float maxrad = -1;

  // Find max flow to normalize fx and fy
  for (int i = 0; i < flow.rows; ++i) {
    for (int j = 0; j < flow.cols; ++j) {
      cv::Vec2f flow_at_point = flow.at<cv::Vec2f>(i, j);
      float fx = flow_at_point[0];
      float fy = flow_at_point[1];
      if ((fabs(fx) > UNKNOWN_FLOW_THRESH) || (fabs(fy) > UNKNOWN_FLOW_THRESH))
        continue;
      float rad = sqrt(fx * fx + fy * fy);
      //float rad = log(1 + sqrt(fx * fx + fy * fy));
      maxrad = maxrad > rad ? maxrad : rad;
    }
  }
  {
    // debug
    static float m_maxrad = -1;
    if (m_maxrad < maxrad) {
      m_maxrad = maxrad;
    }
    std::cout << "m_maxrad： " << m_maxrad << " maxrad: " << maxrad
              << std::endl;
  }

  for (int i = 0; i < flow.rows; ++i) {
    for (int j = 0; j < flow.cols; ++j) {
      uchar *data = color.data + color.step[0] * i + color.step[1] * j;
      cv::Vec2f flow_at_point = flow.at<cv::Vec2f>(i, j);

      float fx = flow_at_point[0];
      float fy = flow_at_point[1];
      if ((fabs(fx) > UNKNOWN_FLOW_THRESH) ||
          (fabs(fy) > UNKNOWN_FLOW_THRESH)) {
        data[0] = data[1] = data[2] = 0;
        continue;
      }
      float rad = sqrt(fx * fx + fy * fy) / maxrad;
      //float rad = log(1 + sqrt(fx * fx + fy * fy)) / maxrad;

      float angle = atan2(-fy, -fx) / CV_PI;
      float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);
      int k0 = (int)fk;
      int k1 = (k0 + 1) % colorwheel.size();
      float f = fk - k0;
      // f = 0; // uncomment to see original color wheel

      for (int b = 0; b < 3; b++) {
        float col0 = colorwheel[k0][b] / 255.0;
        float col1 = colorwheel[k1][b] / 255.0;
        float col = (1 - f) * col0 + f * col1;
        if (rad <= 1)
          col = 1 - rad * (1 - col); // increase saturation with radius
        else
          col *= .75; // out of range
        data[2 - b] = (int)(255.0 * col);
      }
    }
  }
}
