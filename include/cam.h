#pragma once

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

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
                Camera(cv::Size sensorSize, size_t fps, int idx, cv::Size downscale);

                bool getNextImageRGB(cv::cuda::GpuMat &dst, uint64_t &timestamp);
                bool getNextImageBGR(cv::cuda::GpuMat &dst, uint64_t &timestamp);

            private:
                size_t mWidth;
                size_t mHeight;
                size_t mFps;
                int mIdx;

                void *mCam;

                cv::cuda::GpuMat incoming;
                cv::Size downscale;
            };

            class Screen
            {
            public:
                virtual ~Screen();
                Screen(cv::Size size, size_t fps);

                void render(cv::cuda::GpuMat &image);

            private:
                size_t mWidth;
                size_t mHeight;
                size_t mFps;

                cv::cuda::GpuMat screenSized;
                cv::cuda::GpuMat screenSizedRGB;

                void *mScreen;
            };

            class GPUJpgEncoder
            {
            public:
                virtual ~GPUJpgEncoder();
                GPUJpgEncoder(int w, int h, int quality);
                uchar *EncodeRGB(cv::cuda::GpuMat &image, unsigned long &outBufSize, int quality = 75, bool cudaColorI420 = false);
                uchar *EncodeRGBnv(cv::cuda::GpuMat &image, unsigned long &outBufSize, int quality = 75, bool cudaColorI420 = false);

            private:
                void *cinfo;
                void *jerr;

                void *nppiEncoder;

                void *i420_out;
                char *yuv_data;

                int w, h;

                void *nvbuf;
            };

            void overlay(cv::cuda::GpuMat &in, cv::cuda::GpuMat &overlay);

            // int EncodeJpg(cv::cuda::GpuMat &image, uint8_t *image_compressed, int *image_compressed_size, int quality);

            bool saveJPGRGBCPU(std::string path, cv::cuda::GpuMat &image, int quality = 95, bool sync = true);
            bool saveJPGBGRCPU(std::string path, cv::cuda::GpuMat &image, int quality = 95, bool sync = true);
        }
    }
}
