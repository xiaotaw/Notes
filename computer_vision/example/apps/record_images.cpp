/**
 * Get depth and color images from Azure Kinect DK camera
 * Part of the code comes from https://blog.csdn.net/denkywu/article/details/103305714
 * @author: xiaotaw
 * @email: 
 * @date: 2020/06/22 14:16
 */

//#define DEBUG_STD_COUT
#define DEBUG_DISPLAY
#define CV_WAIT_TIME 5

#include <numeric> /* for std::accumulate */
#include <iostream>
#include <k4a/k4a.hpp>
#include <opencv2/opencv.hpp>

#include "common/path.hpp"
#include "common/time_val.h"
#include "common/static_image_properties.h" /* for colorize depth image */

/**
 * Visualize depth-image in opencv
 */
static void DrawDepthImage(const cv::Mat &depth_img)
{
	double max_depth, min_depth;
	cv::minMaxIdx(depth_img, &min_depth, &max_depth);
	std::cout << "depth min and max: " << min_depth << " : " << max_depth << std::endl;
	cv::Mat depth_scale;
	cv::convertScaleAbs(depth_img, depth_scale, 255 / max_depth);
	cv::imshow("depth image", depth_scale);
	cv::waitKey(0);
}

static void HelpInfo()
{
	std::cout << "Record depth_image, color_image, and imu data of Azure Kinect Camera." << std::endl;
	std::cout << "The depth images are mapped to color camera by k4a's api, and both " << std::endl;
	std::cout << "depth and color images are resized with 480*640, and undistorted." << std::endl;
	std::cout << "Usage 1: " << std::endl;
	std::cout << "    ./executable save_dir" << std::endl;
	std::cout << "Usage 2: " << std::endl;
	std::cout << "    ./executable save_dir save_colorized_depth" << std::endl;
	std::cout << "Usage 3: " << std::endl;
	std::cout << "    ./executable save_dir save_colorized_depth min_depth_for_colorization max_depth_for_colorization" << std::endl;
	std::cout << "Note1: " << std::endl;
	std::cout << "    save_dir -> string, save_colorized_depth -> {0, 1}, min/max_depth_for_colorization -> uint16_t" << std::endl;
	std::cout << "Note2: " << std::endl;
	std::cout << "    press 'q' or ctrl+C to exit, press 'p' to pause recording, press 'c'  or 'r' to continue recording" << std::endl;
}

static k4a_device_configuration_t GetDefaultConfig()
{
	k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
	config.camera_fps = K4A_FRAMES_PER_SECOND_30;
	config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
	//config.color_resolution = K4A_COLOR_RESOLUTION_720P;
	config.color_resolution = K4A_COLOR_RESOLUTION_1536P;
	config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
	// config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
	// ensures that depth and color images are both available in the capture
	config.synchronized_images_only = true;
	return config;
}

int main(int argc, char *argv[])
{
	// deal with args
	std::string save_dir;
    bool save_colorized_depth = false;
	uint16_t min_depth_for_colorize = 20, max_depth_for_colorize = 3000;
	if (argc == 2)
	{
		save_dir = argv[1];
	}
	else if (argc == 3)
	{
		save_dir = argv[1];
        save_colorized_depth = static_cast<bool>(std::stoi(argv[2]));
	}
	else if (argc == 5)
	{
		save_dir = argv[1];
        save_colorized_depth = static_cast<bool>(std::stoi(argv[2]));
		min_depth_for_colorize = std::stoi(argv[3]);
		max_depth_for_colorize = std::stoi(argv[4]);
	}
	else
	{
		HelpInfo();
		exit(EXIT_FAILURE);
	}

	// 发现已连接的设备
	const uint32_t device_count = k4a::device::get_installed_count();
	if (0 == device_count)
	{
		std::cout << "Error: no K4A devices found. " << std::endl;
		return EXIT_FAILURE;
	}
	std::cout << "Done: found " << device_count << " K4A device(s). " << std::endl;
	// 打开（默认）设备
	k4a::device device = k4a::device::open(K4A_DEVICE_DEFAULT);
	std::cout << "Done: open the default device. " << std::endl;
	// 获取一个简单的配置
	k4a_device_configuration_t config = GetDefaultConfig();
	// 启动设备
	device.start_cameras(&config);
	device.start_imu();
	std::cout << "Done: start camera." << std::endl;

	// transformation and undistort map
	k4a::calibration calibration = device.get_calibration(config.depth_mode, config.color_resolution);
	k4a::transformation transformation(calibration);

	auto intrinsics = calibration.color_camera_calibration.intrinsics.parameters.param;
	float factor = 3.2; /* downscale factor */
	std::vector<double> _camera_matrix = {
		intrinsics.fx / factor, 0.f, intrinsics.cx / factor,
		0.f, intrinsics.fy / factor, intrinsics.cy / factor,
		0.f, 0.f, 1.f};
	cv::Mat camera_matrix = cv::Mat(3, 3, CV_64F, &_camera_matrix[0]);
	std::vector<double> _dist_coeffs = {intrinsics.k1, intrinsics.k2, intrinsics.p1,
										intrinsics.p2, intrinsics.k3, intrinsics.k4,
										intrinsics.k5, intrinsics.k6};
	cv::Mat dist_coeffs = cv::Mat(8, 1, CV_64F, &_dist_coeffs[0]);
	cv::Size src_color_size = {calibration.color_camera_calibration.resolution_width,
							   calibration.color_camera_calibration.resolution_height};
	cv::Size tgt_color_size = {int(src_color_size.width / factor), int(src_color_size.height / factor)};
	cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(
		camera_matrix, dist_coeffs, tgt_color_size, 0, tgt_color_size);
#ifdef DEBUG_STD_COUT
	std::cout << "undistorted camera matrix: " << new_camera_matrix << std::endl;
#endif
	cv::Mat_<double> I = cv::Mat_<double>::eye(3, 3);
	cv::Mat map1 = cv::Mat::zeros(tgt_color_size, CV_16SC2);
	cv::Mat map2 = cv::Mat::zeros(tgt_color_size, CV_16UC1);
	cv::initUndistortRectifyMap(camera_matrix, dist_coeffs, I, new_camera_matrix,
								tgt_color_size, map1.type(), map1, map2);
	std::cout << "Done: get transformation from depth to color, get undistort map" << std::endl;

	// 从设备获取捕获
	k4a::capture capture;
	k4a::image color_image, depth_image, transformed_depth_image;
	k4a_imu_sample_t imu_sample;

	cv::Mat cv_rgba_image, cv_rgb_image, cv_scaled_rgb_image, cv_undistorted_rgb_image;
	cv::Mat cv_depth_image, cv_transformed_depth_image, cv_scaled_depth_image, cv_undistorted_depth_image, cv_colorized_depth_image;

	// 稳定设备，并计算device与system时间差
	std::vector<TimeVal> depth_image_time_offsets, color_image_time_offsets, imu_time_offsets;
	for (size_t i = 0; i <= 30;)
	{
		if (device.get_capture(&capture, std::chrono::milliseconds(1000)))
		{
			TimeVal sys_time = TimeVal::GetCurrentSysTime();
			// 计算depth image的device time和system time之间的差
			depth_image = capture.get_depth_image();
			TimeVal depth_offset = (sys_time - depth_image.get_device_timestamp()).Validate();
			depth_image_time_offsets.push_back(depth_offset);
			// 计算color image的device time和system time之间的差
			color_image = capture.get_color_image();
			TimeVal color_offset = (sys_time - color_image.get_device_timestamp()).Validate();
			color_image_time_offsets.push_back(color_offset);
			// 获取IMU数据，清出积累的imu数据，避免积累到下一帧
			uint64_t depth_timestamp = static_cast<uint64_t>(depth_image.get_device_timestamp().count());
			while (device.get_imu_sample(&imu_sample, std::chrono::milliseconds(0)))
			{
				uint64_t imu_timestamp = imu_sample.acc_timestamp_usec;
				if (imu_timestamp >= depth_timestamp)
				{
					break;
				}
			}
			i++;
#ifdef DEBUG_STD_COUT
			std::cout << std::setw(2) << i << " systime: " << sys_time << std::endl;
#endif
		}
		else
		{
			std::cout << "Stablize camera timeout" << std::endl;
		}
	}
	std::cout << "Done: Stablize camera" << std::endl;
	// time offset, mean and std
	TimeVal depth_image_time_offset_mean = TimeVal::Mean(depth_image_time_offsets);
	TimeVal color_image_time_offset_mean = TimeVal::Mean(color_image_time_offsets);
	TimeVal depth_image_time_offset_std = TimeVal::Std(depth_image_time_offsets, depth_image_time_offset_mean);
	TimeVal color_image_time_offset_std = TimeVal::Std(color_image_time_offsets, color_image_time_offset_mean);


	std::cout << "Start to record depth, color, imu data into: " << save_dir << std::endl;
	// make dirs and timestamp files
	std::ofstream color_timestamp_file, depth_timestamp_file, imu_file;
	if ((!save_dir.empty()))
	{
		CreateDirOrExit(save_dir);
		CreateDirOrExit(save_dir + "/color");
		CreateDirOrExit(save_dir + "/depth");
        if (save_colorized_depth){
    		CreateDirOrExit(save_dir + "/colored_depth");
        }
		// color image timestamp
		OpenFileOrExit(color_timestamp_file, save_dir + "/rgb.txt", std::ios_base::app);
		color_timestamp_file << "# color images" << std::endl;
		color_timestamp_file << "# timestamp filename" << std::endl;
		color_timestamp_file << "# color_sys_time = color_device_time + mean of color_image_time_offset" << std::endl;
		color_timestamp_file << "# time offset for color image, mean: " << color_image_time_offset_mean;
		color_timestamp_file << " std: " << color_image_time_offset_std << std::endl;
		// depth image timestamp
		OpenFileOrExit(depth_timestamp_file, save_dir + "/depth.txt", std::ios_base::app);
		depth_timestamp_file << "# depth images" << std::endl;
		depth_timestamp_file << "# timestamp filename" << std::endl;
		depth_timestamp_file << "# depth_sys_time = depth_device_time + mean of depth_image_time_offset" << std::endl;
		depth_timestamp_file << "# time offset for depth image, mean: " << depth_image_time_offset_mean;
		depth_timestamp_file << " std: " << depth_image_time_offset_std << std::endl;
		// imu data timestamp
		OpenFileOrExit(imu_file, save_dir + "/imu.txt", std::ios_base::app);
		imu_file << "# imu timestamp is aligned with depth image " << std::endl;
		imu_file << "# which means imu_sys_time = imu_device_time + mean of depth_image_time_offset" << std::endl;
		imu_file << "# imu data: system_timestamp, wx,wy,wz, ax,ay,az" << std::endl;
		// save camera intrinsic into file
		std::ofstream color_intrinsic_file;
		OpenFileOrExit(color_intrinsic_file, save_dir + "/color_intrinsic.txt", std::ios_base::out);
		color_intrinsic_file << new_camera_matrix << std::endl;
		color_intrinsic_file.close();
		std::ofstream depth_intrinsic_file;
		OpenFileOrExit(depth_intrinsic_file, save_dir + "/depth_intrinsic.txt", std::ios_base::out);
		depth_intrinsic_file << new_camera_matrix << std::endl;
		depth_intrinsic_file.close();
	}

#ifdef DEBUG_STD_COUT
	std::cout << "time offset for depth image, mean: " << depth_image_time_offset_mean;
	std::cout << " std: " << depth_image_time_offset_std << std::endl;
	std::cout << "time offset for color image, mean: " << color_image_time_offset_mean;
	std::cout << " std: " << color_image_time_offset_std << std::endl;
#endif

	enum
	{
		kStop = 0,
		kPause = 1,
		kRecord = 2
	} control_flag;
	for (control_flag = kRecord; control_flag != kStop;)
	{

		if (device.get_capture(&capture))
		{
			TimeVal sys_time = TimeVal::GetCurrentSysTime();
			// rgb
			// * Each pixel of BGRA32 data is four bytes. The first three bytes represent Blue, Green,
			// * and Red data. The fourth byte is the alpha channel and is unused in the Azure Kinect APIs.
			color_image = capture.get_color_image();
			cv_rgba_image = cv::Mat(src_color_size, CV_8UC4, (void *)color_image.get_buffer());
			cv::cvtColor(cv_rgba_image, cv_rgb_image, cv::COLOR_BGRA2BGR);

			// depth
			// * Each pixel of DEPTH16 data is two bytes of little endian unsigned depth data. The unit of the data is
			// * in millimeters from the origin of the camera.
			depth_image = capture.get_depth_image();
			//cv_depth_image = cv::Mat(depth_image.get_height_pixels(), depth_image.get_width_pixels(), CV_16UC1,
			//					 (void *)depth_image.get_buffer());//, static_cast<size_t>(depth_image.get_stride_bytes()));

			transformed_depth_image = transformation.depth_image_to_color_camera(depth_image);
			cv_transformed_depth_image = cv::Mat(src_color_size, CV_16U, (void *)transformed_depth_image.get_buffer(), static_cast<size_t>(transformed_depth_image.get_stride_bytes()));
			//DrawDepth(cv_transformed_depth_image);

			cv::resize(cv_rgb_image, cv_scaled_rgb_image, tgt_color_size, 0, 0, cv::INTER_AREA);
			cv::resize(cv_transformed_depth_image, cv_scaled_depth_image, tgt_color_size, 0, 0, cv::INTER_NEAREST);
			//DrawDepth(cv_scaled_depth_image);

			cv::remap(cv_scaled_rgb_image, cv_undistorted_rgb_image, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
			cv::remap(cv_scaled_depth_image, cv_undistorted_depth_image, map1, map2, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
			//DrawDepth(cv_undistorted_depth_image);

            if (save_colorized_depth){
    			std::vector<Pixel> pixel_buffer;
    			std::pair<uint16_t, uint16_t> expectedValueRange = {min_depth_for_colorize, max_depth_for_colorize};
    			ColorizeDepthImage(cv_undistorted_depth_image, DepthPixelColorizer::ColorizeBlueToRed, expectedValueRange, &pixel_buffer);
    			cv_colorized_depth_image = cv::Mat(tgt_color_size, CV_8UC4, pixel_buffer.data());
            }

			//std::cout << "start to get IMU sample" << std::endl;
			uint64_t depth_timestamp = static_cast<uint64_t>(depth_image.get_device_timestamp().count());
			while (device.get_imu_sample(&imu_sample, std::chrono::milliseconds(0)))
			{
				uint64_t imu_timestamp = imu_sample.acc_timestamp_usec;
				// imu data: timestamp, wx,wy,wz, ax,ay,az
				if (control_flag == kRecord && !save_dir.empty())
				{
					const auto &w = imu_sample.gyro_sample.xyz;
					const auto &a = imu_sample.acc_sample.xyz;
					TimeVal imu_systime = TimeVal(std::chrono::microseconds(imu_timestamp)) + depth_image_time_offset_mean;
					imu_file << imu_systime.Validate().GetTimeStampStr() << " ";
					imu_file << w.x << " " << w.y << " " << w.z << " ";
					imu_file << a.x << " " << a.y << " " << a.z << std::endl;
				}
				if (imu_timestamp >= depth_timestamp)
				{
					break;
				}
			}

#ifdef DEBUG_DISPLAY
			// show image
			cv::imshow("color", cv_undistorted_rgb_image);
            if (save_colorized_depth){
    			cv::imshow("depth", cv_colorized_depth_image);
            }
#endif
			if (control_flag == kRecord && !save_dir.empty())
			{
				TimeVal depth_systime = (TimeVal(depth_image.get_device_timestamp()) + depth_image_time_offset_mean).Validate();
				std::string depth_fn = "depth/" + depth_systime.GetTimeStampStr() + ".png";
				cv::imwrite(save_dir + "/" + depth_fn, cv_undistorted_depth_image);
                if (save_colorized_depth){
    				std::string colorized_depth_fn = "colored_depth/" + depth_systime.GetTimeStampStr() + ".png";
    				cv::imwrite(save_dir + "/" + colorized_depth_fn, cv_colorized_depth_image);
                }
				TimeVal color_systime = (TimeVal(color_image.get_device_timestamp()) + color_image_time_offset_mean).Validate();
				std::string color_fn = "color/" + color_systime.GetTimeStampStr() + ".png";
				cv::imwrite(save_dir + "/" + color_fn, cv_undistorted_rgb_image);

				color_timestamp_file << color_systime.GetTimeStampStr() << " " << color_fn << std::endl;
				depth_timestamp_file << depth_systime.GetTimeStampStr() << " " << depth_fn << std::endl;
			}
		}
		else
		{
			std::cout << "false: K4A_WAIT_RESULT_TIMEOUT." << std::endl;
		}
#ifdef DEBUG_DISPLAY
		switch (cv::waitKey(CV_WAIT_TIME))
		{
		case 'q':
			control_flag = kStop;
			std::cout << "Stop recoding and exit" << std::endl;
			break;
		case 'p':
			control_flag = kPause;
			std::cout << "Pause recording" << std::endl;
			break;
		case 'c':
		case 'r':
			control_flag = kRecord;
			std::cout << "Continue recording" << std::endl;
			break;
		}
#endif
	}

	if (!save_dir.empty())
	{
		color_timestamp_file.close();
		depth_timestamp_file.close();
		imu_file.close();
	}

	cv::destroyAllWindows();
	// 释放，关闭设备
	color_image.reset();
	depth_image.reset();
	transformed_depth_image.reset();
	capture.reset();
	device.stop_imu();
	device.close();

	return EXIT_SUCCESS;
}
