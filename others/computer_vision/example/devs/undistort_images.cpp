/* code comes from https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/933 */
/* some bugs was fixed */

#include <iostream>
#include <iomanip>
#include <vector>
#include <k4a/k4a.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

template <typename T>
Mat create_mat_from_buffer(T *data, int width, int height, int channels = 1)
{
    Mat mat(height, width, CV_MAKETYPE(DataType<T>::type, channels));
    memcpy(mat.data, data, width * height * channels * sizeof(T));
    return mat;
}

int main(int argc, char **argv)
{
    cout << endl
         << "-------" << endl;
    cout << "Openning Kinect for Azure ..." << endl;
    cv::Mat imBGRA, imBGR, imD, im_transformed_color_image;
    cv::Mat im_transformed_depth_image;

    cv::Mat camera_matrix;
    cv::Mat new_camera_matrix;
    float factor{3.2}; // scaling factor
    cv::Mat cv_undistorted_color;
    cv::Mat cv_undistorted_depth;
    cv::Mat cv_depth_downscaled;
    cv::Mat cv_color_downscaled;
    cv::Mat map1;
    cv::Mat map2;

    int returnCode = 1;
    k4a_device_t device = NULL;
    const int32_t TIMEOUT_IN_MS = 1000;
    k4a_transformation_t transformation = NULL;
    int captureFrameCount;
    k4a_capture_t capture = NULL;
    k4a_image_t depth_image = NULL;
    k4a_image_t color_image = NULL;
    k4a_image_t ir_image = NULL;
    k4a_image_t transformed_color_image = NULL;
    k4a_image_t transformed_depth_image = NULL;

    uint32_t device_count = k4a_device_get_installed_count();
    bool continueornot = true;
    if (device_count == 0)
    {
        cout << "No K4A devices found" << endl;
        return 0;
    }

    if (K4A_RESULT_SUCCEEDED != k4a_device_open(K4A_DEVICE_DEFAULT, &device))
    {
        cout << "No K4A devices found" << endl;
        if (device != NULL)
        {
            k4a_device_close(device);
        }
    }
    k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    config.synchronized_images_only = true;
    config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    //config.color_resolution = K4A_COLOR_RESOLUTION_720P;
    config.color_resolution = K4A_COLOR_RESOLUTION_1536P;
    //config.depth_format = K4A_IMAGE_FORMAT_DEPTH16;
    config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    config.camera_fps = K4A_FRAMES_PER_SECOND_30;
    k4a_device_set_color_control(device, K4A_COLOR_CONTROL_CONTRAST, K4A_COLOR_CONTROL_MODE_MANUAL, 7);
    k4a_device_set_color_control(device, K4A_COLOR_CONTROL_SHARPNESS, K4A_COLOR_CONTROL_MODE_MANUAL, 4);

    k4a_calibration_t calibration;
    if (K4A_RESULT_SUCCEEDED !=
        k4a_device_get_calibration(device, config.depth_mode, config.color_resolution, &calibration))
    {
        cout << "Failed to get calibration" << endl;
        return 0;
    }
    const int width = calibration.color_camera_calibration.resolution_width;
    const int height = calibration.color_camera_calibration.resolution_height;
    auto calib = calibration.depth_camera_calibration;

    cout << "\n===== Device  =====\n";
    cout << "depth camera resolution width: " << calib.resolution_width << endl;
    cout << "depth camera resolution height: " << calib.resolution_height << endl;
    cout << "depth camera principal point x: " << calib.intrinsics.parameters.param.cx << endl;
    cout << "depth camera principal point y: " << calib.intrinsics.parameters.param.cy << endl;
    cout << "depth camera focal length x: " << calib.intrinsics.parameters.param.fx << endl;
    cout << "depth camera focal length y: " << calib.intrinsics.parameters.param.fy << endl;
    cout << "depth camera radial distortion coefficients:" << endl;
    cout << "depth camera k1: " << calib.intrinsics.parameters.param.k1 << endl;
    cout << "depth camera k2: " << calib.intrinsics.parameters.param.k2 << endl;
    cout << "depth camera k3: " << calib.intrinsics.parameters.param.k3 << endl;
    cout << "depth camera k4: " << calib.intrinsics.parameters.param.k4 << endl;
    cout << "depth camera k5: " << calib.intrinsics.parameters.param.k5 << endl;
    cout << "depth camera k6: " << calib.intrinsics.parameters.param.k6 << endl;
    cout << "depth camera center of distortion in Z=1 plane, x: " << calib.intrinsics.parameters.param.codx << endl;
    cout << "depth camera center of distortion in Z=1 plane, y: " << calib.intrinsics.parameters.param.cody << endl;
    cout << "depth camera tangential distortion coefficient x: " << calib.intrinsics.parameters.param.p1 << endl;
    cout << "depth camera tangential distortion coefficient y: " << calib.intrinsics.parameters.param.p2 << endl;
    cout << "depth camera metric radius: " << calib.intrinsics.parameters.param.metric_radius << endl;

    auto calib1 = calibration.color_camera_calibration;
    cout << "color camera resolution width: " << calib1.resolution_width << endl;
    cout << "color camera resolution height: " << calib1.resolution_height << endl;
    cout << "color camera principal point x: " << calib1.intrinsics.parameters.param.cx << endl;
    cout << "color camera principal point y: " << calib1.intrinsics.parameters.param.cy << endl;
    cout << "color camera focal length x: " << calib1.intrinsics.parameters.param.fx << endl;
    cout << "color camera focal length y: " << calib1.intrinsics.parameters.param.fy << endl;
    cout << "color camera radial distortion coefficients:" << endl;
    cout << "color camera k1: " << calib1.intrinsics.parameters.param.k1 << endl;
    cout << "color camera k2: " << calib1.intrinsics.parameters.param.k2 << endl;
    cout << "color camera k3: " << calib1.intrinsics.parameters.param.k3 << endl;
    cout << "color camera k4: " << calib1.intrinsics.parameters.param.k4 << endl;
    cout << "color camera k5: " << calib1.intrinsics.parameters.param.k5 << endl;
    cout << "color camera k6: " << calib1.intrinsics.parameters.param.k6 << endl;
    cout << "color camera center of distortion in Z=1 plane, x: " << calib1.intrinsics.parameters.param.codx << endl;
    cout << "color camera center of distortion in Z=1 plane, y: " << calib1.intrinsics.parameters.param.cody << endl;
    cout << "color camera tangential distortion coefficient x: " << calib1.intrinsics.parameters.param.p1 << endl;
    cout << "color camera tangential distortion coefficient y: " << calib1.intrinsics.parameters.param.p2 << endl;
    cout << "color camera metric radius: " << calib1.intrinsics.parameters.param.metric_radius << endl;

    //init undistortion map
    auto intrinsics = calibration.color_camera_calibration.intrinsics.parameters.param;
    cv_undistorted_color = cv::Mat::zeros(
        height / factor,
        width / factor,
        CV_8UC4);

    cv_undistorted_depth = cv::Mat::zeros(
        height / factor,
        width / factor,
        CV_16U);

    cv_depth_downscaled = cv::Mat::zeros(
        height / factor,
        width / factor,
        CV_16U);
    cv_color_downscaled = cv::Mat::zeros(
        height / factor,
        width / factor,
        CV_8UC4);

    std::vector<double> _camera_matrix = {
        intrinsics.fx / factor,
        0.f,
        intrinsics.cx / factor,
        0.f,
        intrinsics.fy / factor,
        intrinsics.cy / factor,
        0.f,
        0.f,
        1.f};

    // Create cv matrices
    camera_matrix = cv::Mat(3, 3, CV_64F, &_camera_matrix[0]);

    std::vector<double> _dist_coeffs = {intrinsics.k1, intrinsics.k2, intrinsics.p1,
                                        intrinsics.p2, intrinsics.k3, intrinsics.k4,
                                        intrinsics.k5, intrinsics.k6};

    cv::Mat dist_coeffs = cv::Mat(8, 1, CV_64F, &_dist_coeffs[0]);
    new_camera_matrix = cv::getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        cv_depth_downscaled.size(),
        0,
        cv_depth_downscaled.size());

    cv::Mat_<double> I = cv::Mat_<double>::eye(3, 3);

    map1 = cv::Mat::zeros(cv_depth_downscaled.size(), CV_16SC2);
    map2 = cv::Mat::zeros(cv_depth_downscaled.size(), CV_16UC1);
    initUndistortRectifyMap(camera_matrix, dist_coeffs, I, new_camera_matrix, cv::Size(width / factor, height / factor),
                            map1.type(), map1, map2);

    transformation = k4a_transformation_create(&calibration);

    if (K4A_RESULT_SUCCEEDED != k4a_device_start_cameras(device, &config))
    {
        cout << "Failed to start cameras" << endl;
        return 0;
    }

    //vector<int> compression_params;
    //compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    //compression_params.push_back(0);
    int count = 0;
    for (;;)
    {
        std::ostringstream cnt;
        //string path = "Frames/";
        cnt << std::setw(6) << std::setfill('0') << count;
        std::string file = "frame-" + cnt.str() + ".depth.png";
        std::string file1 = "frame-" + cnt.str() + ".color.png";
        std::string file2 = "frame-" + cnt.str() + ".depth-ori.png";
        std::string file3 = "frame-" + cnt.str() + ".color-ori.png";
        // Get a capture
        switch (k4a_device_get_capture(device, &capture, TIMEOUT_IN_MS))
        {
        case K4A_WAIT_RESULT_SUCCEEDED:
            break;
        case K4A_WAIT_RESULT_TIMEOUT:
            cout << "Timed out waiting for a capture" << endl;
            return 0;
        case K4A_WAIT_RESULT_FAILED:
            cout << "Failed to read a capture" << endl;
            return 0;
        }

        // Get a depth image
        depth_image = k4a_capture_get_depth_image(capture);
        if (depth_image == 0)
        {
            cout << "Failed to get depth image from capture" << endl;
        }

        int depth_image_width_pixels = k4a_image_get_width_pixels(depth_image);
        int depth_image_height_pixels = k4a_image_get_height_pixels(depth_image);
        imD = cv::Mat(depth_image_height_pixels, depth_image_width_pixels, CV_16UC1, (void *)k4a_image_get_buffer(depth_image));
        //ushort d = imD.ptr<ushort>(320)[320];
        //cout << "d = " << d << endl;
        // imshow("depth Image", imD);
        //waitKey(0);
        // Get a color image
        color_image = k4a_capture_get_color_image(capture);

        if (color_image == 0)
        {
            cout << "Failed to get color image from capture" << endl;
        }
        int color_image_width_pixels = k4a_image_get_width_pixels(color_image);
        int color_image_height_pixels = k4a_image_get_height_pixels(color_image);
        imBGRA = cv::Mat(color_image_height_pixels, color_image_width_pixels, CV_8UC4, (void *)k4a_image_get_buffer(color_image));

        cvtColor(imBGRA, imBGR, COLOR_BGRA2BGR);

        if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
                                                     color_image_width_pixels,
                                                     color_image_height_pixels,
                                                     color_image_width_pixels * (int)sizeof(uint16_t),
                                                     &transformed_depth_image))
        {
            cout << "Failed to create transformed depth image" << endl;
            return false;
        }
        if (K4A_RESULT_SUCCEEDED != k4a_transformation_depth_image_to_color_camera(transformation, depth_image, transformed_depth_image))
        {

            cout << "Failed to compute transformed depth image" << endl;
            return false;
        }

        cv::Mat frame;
        uint8_t *buffer = k4a_image_get_buffer(transformed_depth_image);
        uint16_t *depth_buffer = reinterpret_cast<uint16_t *>(buffer);
        create_mat_from_buffer<uint16_t>(depth_buffer, color_image_width_pixels, color_image_height_pixels).copyTo(frame);

        if (frame.empty())
            cout << "im_transformed_color_image is empty" << endl;

        cv::resize(
            frame,
            cv_depth_downscaled,
            cv_depth_downscaled.size(),
            cv::INTER_AREA);

        cv::resize(imBGRA,
                   cv_color_downscaled,
                   cv_color_downscaled.size(),
                   INTER_LINEAR); //CV_INTER_AREA

        /* By Jason Juang:  */
        /* Using INTER_LINEAR will cause antifacts at depth discontinuity. */
        /* Or do some depth discontinuity detection, */
        /* where at non depth discontinuity use INTER_LINEAR and depth discontinuity use INTER_NEAREST. */
        remap(cv_depth_downscaled, cv_undistorted_depth, map1, map2, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
        remap(cv_color_downscaled, cv_undistorted_color, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        //cv::cvtColor(cv_undistorted_color, cv_undistorted_color, cv::COLOR_BGRA2RGB);

        //imwrite(file, cv_undistorted_depth, compression_params);
        //imwrite(file1, cv_undistorted_color, compression_params);

        imwrite(file, cv_undistorted_depth);
        imwrite(file1, cv_undistorted_color);


        //imwrite(file2, cv_depth_downscaled);
        //imwrite(file3, cv_color_downscaled);

        k4a_image_release(depth_image);
        k4a_image_release(color_image);

        k4a_image_release(transformed_depth_image);
        k4a_capture_release(capture);
        ++count;
    }

    return 0;
}