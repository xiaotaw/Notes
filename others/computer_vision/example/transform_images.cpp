#include <k4a/k4a.hpp>

#include <iostream>
#include <vector>
#include <array>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#define MAX_DEPTH 5000.0f
#define MIN_DEPTH 30.0f
#define WAIT_TIME 5

// the code is just for test, the code comes from
// https://github.com/forestsen/KinectAzureDKProgramming

void DrawDepthImage(const cv::Mat &depth_img, std::string win_name = "depth image")
{
    double max_depth, min_depth;
    //cv::minMaxIdx(depth_img, &min_depth, &max_depth);
    //min_depth = (min_depth < MIN_DEPTH) ? (MIN_DEPTH) : (min_depth);
    //max_depth = (max_depth > MAX_DEPTH) ? (MAX_DEPTH) : (max_depth);
    //std::cout << max_depth << " ; " << min_depth << std::endl;
    //Visualize depth-image in opencv
    cv::Mat depth_scale;
    cv::convertScaleAbs(depth_img, depth_scale, 255 / MAX_DEPTH);
    cv::imshow(win_name, depth_scale);
    //cv::waitKey(0);
}

void SaveRawDepthImage(const cv::Mat &depth_img, const std::string &path)
{
    double max_depth, min_depth;
    cv::minMaxIdx(depth_img, &min_depth, &max_depth);
    //Visualize depth-image in opencv
    cv::Mat depth_scale;
    cv::convertScaleAbs(depth_img, depth_scale, 255 / max_depth);
    cv::imwrite(path, depth_scale);
}

void SaveDepthImage(const cv::Mat &depth_img, const std::string &path)
{
    double max_depth, min_depth;
    cv::minMaxIdx(depth_img, &min_depth, &max_depth);
    //Visualize depth-image in opencv
    cv::Mat depth_scale;
    cv::convertScaleAbs(depth_img, depth_scale, 255 / max_depth);
    cv::imwrite(path, depth_scale);
}

void print_calibration(const k4a::calibration calibration)
{
    std::cout << calibration.color_camera_calibration.resolution_width << " ; ";
    std::cout << calibration.color_camera_calibration.resolution_height << " ; ";
    std::cout << calibration.color_camera_calibration.intrinsics.parameters.param.cx << " ; ";
    std::cout << calibration.color_camera_calibration.intrinsics.parameters.param.cy << " ; ";
    std::cout << calibration.color_camera_calibration.intrinsics.parameters.param.fx << " ; ";
    std::cout << calibration.color_camera_calibration.intrinsics.parameters.param.fy << std::endl;
}

void downscale_calibration(const k4a::calibration calibration,
                           k4a::calibration &calibration_color_downscaled, float scale = 2.0)
{
    calibration_color_downscaled.color_camera_calibration.resolution_width /= scale;
    calibration_color_downscaled.color_camera_calibration.resolution_height /= scale;
    calibration_color_downscaled.color_camera_calibration.intrinsics.parameters.param.cx /= scale;
    calibration_color_downscaled.color_camera_calibration.intrinsics.parameters.param.cy /= scale;
    calibration_color_downscaled.color_camera_calibration.intrinsics.parameters.param.fx /= scale;
    calibration_color_downscaled.color_camera_calibration.intrinsics.parameters.param.fy /= scale;
}

int main(int argc, char **argv)
{
    const uint32_t deviceCount = k4a::device::get_installed_count();
    if (deviceCount == 0)
    {
        cout << "no azure kinect devices detected!" << endl;
    }
    // depth mode: K4A_DEPTH_MODE_NFOV_UNBINNED and K4A_DEPTH_MODE_WFOV_2X2BINNED is recommended
    // if depth mode == NFOV, rgb resolution = 4:3 is recommended.
    k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    config.camera_fps = K4A_FRAMES_PER_SECOND_30;
    config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    config.color_resolution = K4A_COLOR_RESOLUTION_1536P;
    config.synchronized_images_only = true;

    cout << "Started opening K4A device..." << endl;
    k4a::device device = k4a::device::open(K4A_DEVICE_DEFAULT);
    device.start_cameras(&config);
    cout << "Finished opening K4A device." << endl;

    k4a::calibration calibration = device.get_calibration(config.depth_mode,
                                                          config.color_resolution);
    k4a::calibration calibration_downscaled;
    memcpy(&calibration_downscaled, &calibration, sizeof(k4a::calibration));

    downscale_calibration(calibration, calibration_downscaled, 3.2);

    print_calibration(calibration);
    print_calibration(calibration_downscaled);

    k4a::transformation transformation(calibration_downscaled);
    //k4a::transformation transformation_downscaled;

    k4a::capture capture;

    k4a::image depthImage;
    k4a::image colorImage;
    k4a::image transformed_depth_image;

    cv::Mat depthFrame;
    cv::Mat colorFrame, resized_color_frame;
    cv::Mat transformed_depth_frame;

    int flag = 1;
    while (1)
    {
        if (flag)
        {
            if (device.get_capture(&capture, std::chrono::milliseconds(0)))
            {
                depthImage = capture.get_depth_image();
                colorImage = capture.get_color_image();
                transformed_depth_image = transformation.depth_image_to_color_camera(depthImage);

                depthFrame = cv::Mat(depthImage.get_height_pixels(),
                                     depthImage.get_width_pixels(), CV_16UC1, depthImage.get_buffer());
                colorFrame = cv::Mat(colorImage.get_height_pixels(),
                                     colorImage.get_width_pixels(), CV_8UC4, colorImage.get_buffer());
                cv::resize(colorFrame, resized_color_frame, cv::Size(640, 480), 0, 0, cv::INTER_AREA);
                //cv::resize(colorFrame, resized_color_frame, cv::Size(640, 480));
                transformed_depth_frame = cv::Mat(transformed_depth_image.get_height_pixels(),
                                                  transformed_depth_image.get_width_pixels(), CV_16UC1, transformed_depth_image.get_buffer());

                //cv::imshow("kinect depth map master", depthFrame);
                DrawDepthImage(transformed_depth_frame, "transformed depth frame");
                DrawDepthImage(depthFrame, "kinect depth map master");
                cv::imshow("kinect color frame master", resized_color_frame);
            }
        }
        if (waitKey(WAIT_TIME) == 27 || waitKey(WAIT_TIME) == 'q')
        {
            device.close();
            break;
        }
        else if (waitKey(WAIT_TIME) == 'p')
        {
            flag = 0;
        }
        else if (waitKey(WAIT_TIME) == 's')
        {
        }
        else if (waitKey(WAIT_TIME) == 'c')
        {
            flag = 1;
        }
    }
    return 0;
}
