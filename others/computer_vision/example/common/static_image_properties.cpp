/**
 * code comes from https://github.com/forestsen/KinectAzureDKProgramming
 */
#include <cmath>
#include <opencv2/core/core.hpp>
#include "static_image_properties.h"

void ColorizeDepthImage(const k4a::image &depthImage,
	DepthPixelVisualizationFunction visualizationFn,
	std::pair<uint16_t, uint16_t> expectedValueRange,
	std::vector<Pixel> *buffer)
{
	const k4a_image_format_t imageFormat = depthImage.get_format();
	if (imageFormat != K4A_IMAGE_FORMAT_DEPTH16 && imageFormat != K4A_IMAGE_FORMAT_IR16)

	{
		throw std::logic_error("Attempted to colorize a non-depth image!");
	}

	const int width = depthImage.get_width_pixels();
	const int height = depthImage.get_height_pixels();

	buffer->resize(static_cast<size_t>(width * height));

	const uint16_t *depthData = reinterpret_cast<const uint16_t *>(depthImage.get_buffer());
	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			const size_t currentPixel = static_cast<size_t>(h * width + w);
			(*buffer)[currentPixel] = visualizationFn(depthData[currentPixel],
				expectedValueRange.first,
				expectedValueRange.second);
		}
	}
}

void ColorizeDepthImage(const cv::Mat depthImage,
	DepthPixelVisualizationFunction visualizationFn,
	std::pair<uint16_t, uint16_t> expectedValueRange,
	std::vector<Pixel> *buffer)
{
	const cv::Size s = depthImage.size();
	buffer->resize(static_cast<size_t>(s.area()));

	const uint16_t * depthdata = reinterpret_cast<const uint16_t *>(depthImage.data);
	for (int h = 0; h < s.height; ++h)
	{
		for (int w = 0; w < s.width; ++w)
		{
			const size_t currentPixel = static_cast<size_t>(h * s.width + w);
			(*buffer)[currentPixel] = visualizationFn(depthdata[currentPixel],
				expectedValueRange.first,
				expectedValueRange.second);
		}
	}
}