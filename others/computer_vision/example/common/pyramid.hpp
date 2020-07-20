/**
 * pyramid for depth and color image
 * @author: xiaotaw
 * @email: 
 * @date: 2020/07/09 19:04
 */
#pragma once
#include <opencv2/opencv.hpp>
#include "common/cuda_texture_surface.h"

template <typename T>
static inline int GetCvType()
{
    std::cout << "not implemented !" << std::endl;
    return -1;
}

template <>
inline int GetCvType<ushort>()
{
    return CV_16UC1;
}

template <>
inline int GetCvType<uchar4>()
{
    return CV_8UC4;
}

/**
 * Note: no check for number of cols, rows and levels
 *    make sure the #lavels is proper before use it
 */

template <typename T, unsigned layers>
class Pyramid
{
public:
    unsigned base_rows_, base_cols_;
    float scale_factor_;
    InterpolationMode interpolation_mode_;

    std::vector<typename CudaTextureSurface2D<T>::Ptr> pyd_;

    // ctor
    Pyramid(const unsigned rows, const unsigned cols, InterpolationMode interpolation_mode, const float scale_factor = 0.5) : base_rows_(rows), base_cols_(cols), scale_factor_(scale_factor), interpolation_mode_(interpolation_mode)
    {
        unsigned r = rows, c = cols;
        for (unsigned i = 0; i < layers; i++)
        {
            pyd_.push_back(std::make_shared<CudaTextureSurface2D<T>>(r, c));
            r = static_cast<unsigned>(r * scale_factor);
            c = static_cast<unsigned>(c * scale_factor);
        }
    }

    // upload the base image
    inline void UploadToPyramid(const PagelockMemory &pagelock_memory, cudaStream_t stream)
    {
        pagelock_memory.UploadToDevice(pyd_[0]->d_array_, stream);
    }

    // download the base image
    inline void DownloadFromPyramid(PagelockMemory &pagelock_memory, cudaStream_t stream)
    {
        pagelock_memory.DownloadFromDevice(pyd_[0]->d_array_, stream);
    }

    /** build pyramid by resize down image level by level
     * Note: UploadToPyramid should be called before this function
     */
    inline void BuildPyramid(cudaStream_t stream)
    {
        for (unsigned i = 0; i < layers - 1; i++)
        {
            pyd_[i]->ResizeDown(*pyd_[i + 1], stream, interpolation_mode_);
            CudaSafeCall(cudaStreamSynchronize(stream));
            CudaSafeCall(cudaGetLastError());
        }

        CudaSafeCall(cudaStreamSynchronize(stream));
        CudaSafeCall(cudaGetLastError());
    }

    /**
     * Concat all images by a Helix-like way.
     */
    inline cv::Mat DownloadHelix(cudaStream_t stream)
    {
        std::vector<PagelockMemory::Ptr> plm;
        // download from device
        for (unsigned i = 0; i < layers; i++)
        {
            plm.push_back(std::make_shared<PagelockMemory>(sizeof(T) * pyd_[i]->rows_ * pyd_[i]->cols_));
            plm[i]->DownloadFromDevice(pyd_[i]->d_array_, stream);
        }
        CudaSafeCall(cudaStreamSynchronize(stream));
        CudaSafeCall(cudaGetLastError());
        // copy to cv::Mat
        std::vector<cv::Mat> imgs;
        for (unsigned i = 0; i < layers; i++)
        {
            imgs.push_back(cv::Mat(pyd_[i]->rows_, pyd_[i]->cols_, GetCvType<T>()));
            plm[i]->HostCopyTo(imgs[i].data);
        }
        cv::Mat img;
        if (layers == 0)
        {
            ; // do nothing
        }
        else if (layers == 1)
        {
            img = cv::Mat::zeros(base_rows_, base_cols_, GetCvType<T>());
        }
        else
        {
            img = cv::Mat::zeros(base_rows_, base_cols_ + pyd_[1]->cols_, GetCvType<T>());
        }
        // concat images
        unsigned o_row = 0, o_col = 0;
        for (unsigned i = 0; i < layers; i++)
        {
            cv::Rect roi_rect = cv::Rect(o_col, o_row, imgs[i].cols, imgs[i].rows);
            //std::cout << roi_rect << std::endl;
            imgs[i].copyTo(img(roi_rect));
            if (i % 2 == 0)
            {
                o_col += imgs[i].cols;
            }
            else
            {
                o_row += imgs[i].rows;
            }
        }
        return img;
    }
};