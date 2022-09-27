#pragma once

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include "jpeglib.h"

namespace flysense
{
    namespace jetson
    {
        namespace camera
        {
            class Camera
            {
            public:
                virtual ~Camera();
                Camera(size_t width, size_t height, size_t fps, int idx);

                bool getNextImageRGB(cv::cuda::GpuMat &dst, uint64_t &timestamp);
                bool getNextImageBGR(cv::cuda::GpuMat &dst, uint64_t &timestamp);

            private:
                size_t mWidth;
                size_t mHeight;
                size_t mFps;
                int mIdx;

                void *mCam;
            };

            class Screen
            {
            public:
                virtual ~Screen();
                Screen(size_t width, size_t height, size_t fps);

                void render(cv::cuda::GpuMat &image);

            private:
                size_t mWidth;
                size_t mHeight;
                size_t mFps;

                void *mScreen;
            };

            class GPUJpgEncoder
            {
            public:
                virtual ~GPUJpgEncoder();
                GPUJpgEncoder();
                uchar *EncodeRGB(cv::cuda::GpuMat &image, unsigned long &outBufSize, int quality = 75, bool cudaColorI420 = false);

            private:
                struct jpeg_compress_struct cinfo;
                struct jpeg_error_mgr jerr;
            };

            // int EncodeJpg(cv::cuda::GpuMat &image, uint8_t *image_compressed, int *image_compressed_size, int quality);

            bool saveJPGRGBCPU(std::string path, cv::cuda::GpuMat &image, int quality = 95, bool sync = true);
            bool saveJPGBGRCPU(std::string path, cv::cuda::GpuMat &image, int quality = 95, bool sync = true);
        }
    }
}
