/**
 * code comes from https://github.com/forestsen/KinectAzureDKProgramming
 */
#pragma once
#include <cmath>
#include <utility>
#include <algorithm>
#include <k4a/k4a.hpp>

// Helper structs/typedefs to cast buffers to
//
struct Pixel
{
	uint8_t Blue;
	uint8_t Green;
	uint8_t Red;
	uint8_t Alpha;
};
using DepthPixel = uint16_t;



inline void ColorConvertHSVtoRGB(float h, float s, float v, float &out_r, float &out_g, float &out_b)
{
	if (s == 0.0f)
	{
		// gray
		out_r = out_g = out_b = v;
		return;
	}

	h = fmodf(h, 1.0f) / (60.0f / 360.0f);
	int i = (int)h;
	float f = h - (float)i;
	float p = v * (1.0f - s);
	float q = v * (1.0f - s * f);
	float t = v * (1.0f - s * (1.0f - f));

	switch (i)
	{
	case 0:
		out_r = v;
		out_g = t;
		out_b = p;
		break;
	case 1:
		out_r = q;
		out_g = v;
		out_b = p;
		break;
	case 2:
		out_r = p;
		out_g = v;
		out_b = t;
		break;
	case 3:
		out_r = p;
		out_g = q;
		out_b = v;
		break;
	case 4:
		out_r = t;
		out_g = p;
		out_b = v;
		break;
	case 5:
	default:
		out_r = v;
		out_g = p;
		out_b = q;
		break;
	}
}

inline std::pair<uint16_t, uint16_t> GetDepthModeRange(const k4a_depth_mode_t depthMode)
{
	switch (depthMode)
	{
	case K4A_DEPTH_MODE_NFOV_2X2BINNED:
		return {(uint16_t)500, (uint16_t)5800};
	case K4A_DEPTH_MODE_NFOV_UNBINNED:
		return {(uint16_t)500, (uint16_t)4000};
	case K4A_DEPTH_MODE_WFOV_2X2BINNED:
		return {(uint16_t)250, (uint16_t)3000};
	case K4A_DEPTH_MODE_WFOV_UNBINNED:
		return {(uint16_t)250, (uint16_t)2500};

	case K4A_DEPTH_MODE_PASSIVE_IR:
	default:
		throw std::logic_error("Invalid depth mode!");
	}
}

inline std::pair<int, int> GetDepthDimensions(const k4a_depth_mode_t depthMode)
{
	switch (depthMode)
	{
	case K4A_DEPTH_MODE_NFOV_2X2BINNED:
		return {320, 288};
	case K4A_DEPTH_MODE_NFOV_UNBINNED:
		return {640, 576};
	case K4A_DEPTH_MODE_WFOV_2X2BINNED:
		return {512, 512};
	case K4A_DEPTH_MODE_WFOV_UNBINNED:
		return {1024, 1024};
	case K4A_DEPTH_MODE_PASSIVE_IR:
		return {1024, 1024};

	default:
		throw std::logic_error("Invalid depth dimensions value!");
	}
}

inline std::pair<int, int> GetColorDimensions(const k4a_color_resolution_t resolution)
{
	switch (resolution)
	{
	case K4A_COLOR_RESOLUTION_720P:
		return {1280, 720};
	case K4A_COLOR_RESOLUTION_2160P:
		return {3840, 2160};
	case K4A_COLOR_RESOLUTION_1440P:
		return {2560, 1440};
	case K4A_COLOR_RESOLUTION_1080P:
		return {1920, 1080};
	case K4A_COLOR_RESOLUTION_3072P:
		return {4096, 3072};
	case K4A_COLOR_RESOLUTION_1536P:
		return {2048, 1536};

	default:
		throw std::logic_error("Invalid color dimensions value!");
	}
}

inline std::pair<uint16_t, uint16_t> GetIrLevels(const k4a_depth_mode_t depthMode)
{
	switch (depthMode)
	{
	case K4A_DEPTH_MODE_PASSIVE_IR:
		return {(uint16_t)0, (uint16_t)100};

	case K4A_DEPTH_MODE_OFF:
		throw std::logic_error("Invalid depth mode!");

	default:
		return {(uint16_t)0, (uint16_t)1000};
	}
}

using DepthPixelVisualizationFunction = Pixel(const DepthPixel &value, const DepthPixel &min, const DepthPixel &max);

void ColorizeDepthImage(const k4a::image &depthImage,
						DepthPixelVisualizationFunction visualizationFn,
						std::pair<uint16_t, uint16_t> expectedValueRange,
						std::vector<Pixel> *buffer);

void ColorizeDepthImage(const cv::Mat depthImage,
						DepthPixelVisualizationFunction visualizationFn,
						std::pair<uint16_t, uint16_t> expectedValueRange,
						std::vector<Pixel> *buffer);




// Functions that provide ways to take depth images and turn them into color representations
// suitable for showing to humans.
//
class DepthPixelColorizer
{
public:
	// Computes a color representation of a depth pixel on the blue-red spectrum, using min
	// as the value for blue and max as the value for red.
	//
	static inline Pixel ColorizeBlueToRed(const DepthPixel &depthPixel,
										  const DepthPixel &min,
										  const DepthPixel &max)
	{
		constexpr uint8_t PixelMax = std::numeric_limits<uint8_t>::max();

		// Default to opaque black.
		//
		Pixel result = {uint8_t(0), uint8_t(0), uint8_t(0), PixelMax};

		// If the pixel is actual zero and not just below the min value, make it black
		//
		if (depthPixel == 0)
		{
			return result;
		}

		uint16_t clampedValue = depthPixel;
		clampedValue = std::min(clampedValue, max);
		clampedValue = std::max(clampedValue, min);

		// Normalize to [0, 1]
		//
		float hue = (clampedValue - min) / static_cast<float>(max - min);

		// The 'hue' coordinate in HSV is a polar coordinate, so it 'wraps'.
		// Purple starts after blue and is close enough to red to be a bit unclear,
		// so we want to go from blue to red.  Purple starts around .6666667,
		// so we want to normalize to [0, .6666667].
		//
		constexpr float range = 2.f / 3.f;
		hue *= range;

		// We want blue to be close and red to be far, so we need to reflect the
		// hue across the middle of the range.
		//
		hue = range - hue;

		float fRed = 0.f;
		float fGreen = 0.f;
		float fBlue = 0.f;
		ColorConvertHSVtoRGB(hue, 1.f, 1.f, fRed, fGreen, fBlue);

		result.Red = static_cast<uint8_t>(fRed * PixelMax);
		result.Green = static_cast<uint8_t>(fGreen * PixelMax);
		result.Blue = static_cast<uint8_t>(fBlue * PixelMax);

		return result;
	}

	// Computes a greyscale representation of a depth pixel.
	//
	static inline Pixel ColorizeGreyscale(const DepthPixel &value, const DepthPixel &min, const DepthPixel &max)
	{
		// Clamp to max
		//
		DepthPixel pixelValue = std::min(value, max);

		constexpr uint8_t PixelMax = std::numeric_limits<uint8_t>::max();
		const auto normalizedValue = static_cast<uint8_t>((pixelValue - min) * (double(PixelMax) / (max - min)));

		// All color channels are set the same (image is greyscale)
		//
		return Pixel{normalizedValue, normalizedValue, normalizedValue, PixelMax};
	}
};
