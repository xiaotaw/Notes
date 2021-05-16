#include "common/logging.h"
#include "dataset/dataset.h"
#include "motion_to_color.h"
#include <iostream>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void HelpInfo() {}

int main(int argc, char **argv) {
  // args
  std::string data_dir = "/media/xt/8T/DATASETS/KinectDkDataset/20200701/";
  std::string data_type = "azure_kinect";
  if (argc == 1) {
  } else if (argc == 3) {
    data_dir = argv[1];
    data_type = argv[2];
  } else {
    HelpInfo();
    exit(EXIT_FAILURE);
  }

  RgbdDataset::Ptr dataset = CreateDataset(data_type, data_dir);

  Mat prevgray, gray, hsv, flow, cflow, frame, frame_depth, motion2color, hsv_split[3];

  namedWindow("flow", 0);

  for (int num = 0; num < dataset->num(); num++) {
    double t = (double)getTickCount();

    dataset->FetchNextFrame(frame_depth, frame);

    cvtColor(frame, gray, CV_BGR2GRAY);
    cvtColor(frame, hsv, CV_BGR2HSV);
    split(hsv, hsv_split);
    imshow("original", frame);
    imshow("gray", gray);
    imshow("h", hsv_split[0]);
    imshow("s", hsv_split[1]);
    imshow("v", hsv_split[2]);

    if (prevgray.data) {
      calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
      MotionToColor(flow, motion2color);
      imshow("flow", motion2color);
    }
    // if (waitKey(10) >= 0)
    //   break;
    waitKey(0);

    std::swap(prevgray, gray);

    t = (double)getTickCount() - t;
    cout << "cost time: " << t / ((double)getTickFrequency() * 1000.) << endl;
  }
  return 0;
}