// https://blog.csdn.net/weixin_41558411/article/details/112427639

#include "common/logging.h"
#include "dataset/dataset.h"
#include "motion_to_color.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <time.h>
#include <vector>
using namespace cv;
using namespace std;
using namespace cv::cuda;

void convertFlowToImage(const Mat &flow, Mat &img_x, Mat &img_y,
                        double lowerBound, double higherBound) {
#define CAST(v, L, H)                                                          \
  ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255 * ((v) - (L)) / ((H) - (L))))
  for (int i = 0; i < img_x.rows; ++i) {
    for (int j = 0; j < img_x.cols; ++j) {
      // float x = flow_x.at<float>(i,j);
      // float y = flow_y.at<float>(i,j);
      img_x.at<uchar>(i, j) =
          CAST(flow.at<Point2f>(i, j).x, lowerBound, higherBound);
      img_y.at<uchar>(i, j) =
          CAST(flow.at<Point2f>(i, j).x, lowerBound, higherBound);
    }
  }
#undef CAST
}

void HelpInfo() {}

int main(int argc, char *argv[]) {
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

  vector<Mat> flow;
  GpuMat prev_gray, curr_gray, d_flow, cu_dst_y, cu_dst_x;
  Mat prev, curr, frame, frame_depth, motion2color, hsv, hsv_split[3];
  Ptr<cuda::OpticalFlowDual_TVL1> alg_tvl1 =
      cuda::OpticalFlowDual_TVL1::create();

  RgbdDataset::Ptr dataset = CreateDataset(data_type, data_dir);
  dataset->FetchNextFrame(frame_depth, frame);
  cv::cvtColor(frame, prev, CV_BGR2GRAY);
  cv::cvtColor(frame, hsv, CV_BGR2HSV);
  split(hsv, hsv_split);
  // prev = hsv_split[0].clone();

  cv::Size img_size = frame.size();
  prev_gray.create(img_size, CV_8UC1);
  curr_gray.create(img_size, CV_8UC1);
  cu_dst_x = cuda::GpuMat(img_size, CV_8UC1, Scalar(0));
  cu_dst_y = cuda::GpuMat(img_size, CV_8UC1, Scalar(0));

  clock_t begin, end;
  begin = clock();

  namedWindow("flow", 1);

  int num;
  for (num = 0; num < int(dataset->num() - 5); num++) {
    dataset->FetchNextFrame(frame_depth, frame);
    Mat cpu_flow;
    cv::cvtColor(frame, curr, CV_BGR2GRAY);

    cv::cvtColor(frame, hsv, CV_BGR2HSV);
    split(hsv, hsv_split);
    // curr = hsv_split[0].clone();
    imshow("h", hsv_split[0]);
    imshow("s", hsv_split[1]);
    imshow("v", hsv_split[2]);

    prev_gray.upload(prev);
    curr_gray.upload(curr);
    alg_tvl1->calc(prev_gray, curr_gray, d_flow);
    d_flow.download(cpu_flow);

    if (1) {
      MotionToColor(cpu_flow, motion2color);
      imshow("flow", motion2color);
    }
    // if (waitKey(10) >= 0)
    //   break;
    waitKey(0);

    flow.push_back(cpu_flow);
    prev = curr.clone();
    LOG(INFO) << "processing " << num;
  }
  //记得改路径
  char *save_dir = "/media/xt/8T/DATASETS/KinectDkDataset/20200701/flow";
  int save_dir_len = strlen(save_dir);
  for (int i = 0; i < flow.size(); i++) {
    Mat img_x(img_size, CV_8UC1);
    Mat img_y(img_size, CV_8UC1);
    convertFlowToImage(flow[i], img_x, img_y, -15, 15);
    char y_path[save_dir_len + 1 + 19];
    sprintf(y_path, "%s%s%s%06d%s", save_dir, "/", "y_frame_", i, ".jpg");
    char x_path[save_dir_len + 1 + 19];
    sprintf(x_path, "%s%s%s%06d%s", save_dir, "/", "x_frame_", i, ".jpg");
    imwrite(y_path, img_y);
    imwrite(x_path, img_x);
  }
  end = clock();
  cout << "total frames: " << num << endl;
  cout << "time used: " << (double)(end - begin) / CLOCKS_PER_SEC << endl;
  return 0;
}