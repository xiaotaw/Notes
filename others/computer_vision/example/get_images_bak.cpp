#include <iostream>
#include <opencv2/opencv.hpp>
#include <k4a/k4a.hpp>

#define DEBUG_std_cout 1

// the code is just for test
// the code comes from https://blog.csdn.net/denkywu/article/details/103305714
// with some modification to compact the code

void print_screen(std::ostream &out, const k4a::image &image)
{
	out << "\n"
		<< "format: " << image.get_format() << "\n"
		<< "device_timestamp: " << std::setprecision(5) << image.get_device_timestamp().count() * 1e-6 << "\n"
		<< "system_timestamp: " << std::fixed << std::setprecision(5) << image.get_system_timestamp().count() * 1e-9 << "\n"
		<< "height*width: " << image.get_height_pixels() << ", " << image.get_width_pixels()
		<< std::endl;
}

int main(int argc, char *argv[])
{
	// 发现已连接的设备
	const uint32_t device_count = k4a::device::get_installed_count();
	if (0 == device_count)
	{
		std::cout << "Error: no K4A devices found. " << std::endl;
		return EXIT_FAILURE;
	}
	std::cout << "Done: found " << device_count << "K4A device(s). " << std::endl;
	// 打开（默认）设备
	k4a::device device = k4a::device::open(K4A_DEVICE_DEFAULT);
	std::cout << "Done: open the default device. " << std::endl;
	// 配置
	k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
	config.camera_fps = K4A_FRAMES_PER_SECOND_30;
	config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
	//config.color_resolution = K4A_COLOR_RESOLUTION_720P;
	config.color_resolution = K4A_COLOR_RESOLUTION_1080P;
	config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
	// config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
	config.synchronized_images_only = true; // ensures that depth and color images are both available in the capture
	// 启动设备
	device.start_cameras(&config);
	std::cout << "Done: start camera." << std::endl;

	// 从设备获取捕获
	k4a::capture capture;
	k4a::image rgbImage, depthImage, irImage;

	cv::Mat cv_rgbImage_with_alpha, cv_rgbImage_no_alpha;
	cv::Mat cv_depth, cv_depth_8U;
	cv::Mat cv_irImage, cv_irImage_8U;

	//while (true)
	for (size_t i = 0; i < 10000; i++)
	{
		//if (device.get_capture(&capture, std::chrono::milliseconds(0)))
		if (device.get_capture(&capture))
		{
			// rgb
			// * Each pixel of BGRA32 data is four bytes. The first three bytes represent Blue, Green,
			// * and Red data. The fourth byte is the alpha channel and is unused in the Azure Kinect APIs.
			rgbImage = capture.get_color_image();
#if DEBUG_std_cout == 1
			std::cout << "[rgb] ";
			print_screen(std::cout, rgbImage);
#endif
			cv_rgbImage_with_alpha = cv::Mat(rgbImage.get_height_pixels(),
											 rgbImage.get_width_pixels(),
											 CV_8UC4,
											 (void *)rgbImage.get_buffer());
			cv::cvtColor(cv_rgbImage_with_alpha, cv_rgbImage_no_alpha, cv::COLOR_BGRA2BGR);

			// depth
			// * Each pixel of DEPTH16 data is two bytes of little endian unsigned depth data. The unit of the data is 
			// * in millimeters from the origin of the camera.
			depthImage = capture.get_depth_image();
#if DEBUG_std_cout == 1
			std::cout << "[depth] ";
			print_screen(std::cout, depthImage);
#endif
			cv_depth = cv::Mat(depthImage.get_height_pixels(),
							   depthImage.get_width_pixels(),
							   CV_16U,
							   (void *)depthImage.get_buffer(),
							   static_cast<size_t>(depthImage.get_stride_bytes()));
			cv_depth.convertTo(cv_depth_8U, CV_8U, 1);

			// ir
			// * Each pixel of IR16 data is two bytes of little endian unsigned depth data. The value of the data 
			// * represents brightness.
			irImage = capture.get_ir_image();
#if DEBUG_std_cout == 1
			std::cout << "[ir] ";
			print_screen(std::cout, irImage);
#endif
			cv_irImage = cv::Mat(irImage.get_height_pixels(),
								 irImage.get_width_pixels(),
								 CV_16U,
								 (void *)irImage.get_buffer(),
								 static_cast<size_t>(irImage.get_stride_bytes()));
			cv_irImage.convertTo(cv_irImage_8U, CV_8U, 1);

			// show image
			cv::imshow("color", cv_rgbImage_no_alpha);
			cv::imshow("depth", cv_depth_8U);
			cv::imshow("ir", cv_irImage_8U);
			cv::waitKey(1);

			std::cout << "--- test ---" << std::endl;
		}
		else
		{
			std::cout << "false: K4A_WAIT_RESULT_TIMEOUT." << std::endl;
		}
	}

	cv::destroyAllWindows();
	// 释放，关闭设备
	rgbImage.reset();
	depthImage.reset();
	irImage.reset();
	capture.reset();
	device.close();

	// 等待输入，方便显示上述运行结果
	std::cout << "--------------------------------------------" << std::endl;
	std::cout << "Waiting for inputting an integer: ";
	int wd_wait;
	std::cin >> wd_wait;
	std::cout << "------------- closed -------------" << std::endl;
	
	return EXIT_SUCCESS;
}