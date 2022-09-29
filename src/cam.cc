#include "cam.h"

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>
#include <jetson-utils/cudaResize.h>
#include <jetson-utils/cudaOverlay.h>

#include <jetson-utils/cudaRGB.h>
#include <jetson-utils/cudaYUV.h>
#include <jetson-utils/imageIO.h>
#include <jetson-utils/imageFormat.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <chrono>
#include <sstream>

#include "jconfig.h"
#include "jpeglib.h"

#include "NvBuffer.h"
#include "NvUtils.h"
#include "NvJpegEncoder.h"

#include "NPPJpegCoder.h"

#include <chrono>
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

//#include <libgpujpeg/gpujpeg.h>

namespace flysense
{
    namespace jetson
    {
        namespace camera
        {
            Camera::Camera(cv::Size sensorSize, size_t fps, int idx, cv::Size downscale) : mWidth(sensorSize.width), mHeight(sensorSize.height), mFps(fps), mIdx(idx), downscale(downscale)
            {
                videoOptions cam_options;
                cam_options.width = mWidth;
                cam_options.height = mHeight;
                cam_options.frameRate = fps;
                cam_options.deviceType = videoOptions::DeviceType::DEVICE_CSI;
                cam_options.ioType = videoOptions::IoType::INPUT;
                cam_options.codec = videoOptions::Codec::CODEC_RAW;
                cam_options.flipMethod = videoOptions::FlipMethod::FLIP_HORIZONTAL;

                std::stringstream csiStream;
                csiStream << "csi://" << idx;

                videoSource *camera = videoSource::Create(csiStream.str().c_str(), cam_options);

                if (!camera)
                {
                    std::cerr << "could not init cam" << std::endl;
                    SAFE_DELETE(camera);
                    return;
                }

                if (!camera->Open())
                {
                    std::cerr << "could not open cam" << std::endl;
                    SAFE_DELETE(camera);
                    return;
                }

                mCam = (void *)camera;
            }

            bool Camera::getNextImageRGB(cv::cuda::GpuMat &dst, uint64_t &timestamp)
            {
                if (mCam == nullptr)
                {
                    return false;
                }

                videoSource *_cam = (videoSource *)mCam;

                uchar3 *image = NULL;
                bool captured = _cam->Capture(&image, 10000);
                if (!captured)
                {
                    return false;
                }

                timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

                if (!_cam->IsStreaming())
                {
                    return false;
                }

                cv::cuda::GpuMat gpu_frame(cv::Size(_cam->GetWidth(), _cam->GetHeight()), CV_8UC3, (void *)image);
                //  dst = gpu_frame;

                dst = cv::cuda::GpuMat(downscale, CV_8UC3);

                cv::cuda::resize(gpu_frame, dst, downscale);

                // uchar3 *resizedPtr = 0;
                // cudaAllocMapped((void **)&resizedPtr, downscale.width * downscale.height * sizeof(uchar3));
                // cudaResize(image, _cam->GetWidth(), _cam->GetHeight(), resizedPtr, downscale.width, downscale.height);

                // cv::cuda::GpuMat gpu_frame(cv::Size(downscale.width, downscale.height), CV_8UC3, (void *)resizedPtr);
                // gpu_frame.copyTo(dst);
                // CUDA_FREE_HOST(image);
                //  CUDA_FREE_HOST(resizedPtr);

                return true;
            }

            bool Camera::getNextImageBGR(cv::cuda::GpuMat &dst, uint64_t &timestamp)
            {
                cv::cuda::GpuMat tmp;
                bool ok = getNextImageRGB(tmp, timestamp);
                if (!ok)
                {
                    return false;
                }

                videoSource *_cam = (videoSource *)mCam;

                cv::cuda::GpuMat gpu_frame_bgr(cv::Size(_cam->GetWidth(), _cam->GetHeight()), CV_8UC3);
                cv::cuda::cvtColor(tmp, gpu_frame_bgr, cv::COLOR_RGB2BGR);

                dst = gpu_frame_bgr;
                return true;
            }

            Screen::Screen(cv::Size size, size_t fps) : mWidth(size.width), mHeight(size.height), mFps(fps)
            {
                cv::VideoWriter *writer = new cv::VideoWriter(
                    "appsrc ! videoconvert ! video/x-raw,format=RGBA ! nvvidconv ! nvoverlaysink sync=false",
                    cv::CAP_GSTREAMER,
                    0,   // fourcc
                    fps, // fps
                    cv::Size(mWidth, mHeight),
                    {
                        cv::VideoWriterProperties::VIDEOWRITER_PROP_IS_COLOR,
                        1,
                    });

                if (!writer->isOpened())
                {
                    std::cerr << "could not open screen" << std::endl;
                    return;
                }

                mScreen = (void *)writer;

                screenSized = cv::cuda::GpuMat(size, CV_8UC3);
                screenSizedRGB = cv::cuda::GpuMat(size, CV_8UC3);
            }

            void Screen::render(cv::cuda::GpuMat &image)
            {
                if (mScreen == nullptr)
                {
                    return;
                }

                cv::Mat out;
                cv::VideoWriter *_screen = (cv::VideoWriter *)mScreen;
                if (image.size().width != this->mWidth || image.size().height != this->mHeight)
                {
                    cv::cuda::resize(image, screenSized, cv::Size(mWidth, mHeight));
                    cv::cuda::cvtColor(screenSized, screenSizedRGB, cv::COLOR_RGB2BGR);
                }
                else
                {
                    cv::cuda::cvtColor(image, screenSizedRGB, cv::COLOR_RGB2BGR);
                }

                screenSizedRGB.download(out);
                _screen->write(out);
            }

            Screen::~Screen()
            {
                if (mScreen == nullptr)
                {
                    return;
                }

                cv::VideoWriter *_screen = (cv::VideoWriter *)mScreen;
                _screen->release();
                delete _screen;
                _screen = 0;
            }

            Camera::~Camera()
            {
                if (mCam == nullptr)
                {
                    return;
                }

                videoSource *_cam = (videoSource *)mCam;
                _cam->Close();
                delete _cam;
                _cam = 0;
            }

            void overlay(cv::cuda::GpuMat &in, cv::cuda::GpuMat &overlay)
            {
                uchar4 *rgbaPtr = 0;
                cudaAllocMapped((void **)&rgbaPtr, in.size().width * in.size().height * sizeof(uchar4));
                cudaRGB8ToRGBA8((uchar3 *)in.cudaPtr(), rgbaPtr, in.size().width, in.size().height);
                // CUDA_FREE_HOST(rgbaPtr);
                cudaOverlay((void *)overlay.cudaPtr(), overlay.size().width, overlay.size().height, (void *)rgbaPtr, in.size().width, in.size().height, imageFormat::IMAGE_RGBA8, 0, 0);

                uchar3 *rgbPtr = 0;
                cudaAllocMapped((void **)&rgbPtr, in.size().width * in.size().height * sizeof(uchar3));
                cudaRGBA8ToRGB8(rgbaPtr, rgbPtr, in.size().width, in.size().height);

                cv::cuda::GpuMat out(in.size(), CV_8UC3, rgbPtr);
                CUDA_FREE_HOST(rgbaPtr);
                out.copyTo(in);
                CUDA_FREE_HOST(rgbPtr);
            }

            /**
             * @brief Converts an OpenCV cuda GpuMat to a uchar3 jetson-utils image. uchar3 in RGB, GpuMat in BGR
             *
             * @param src
             * @param dst
             */
            void cvImg2uchar3(cv::cuda::GpuMat &src, uchar3 **dst)
            {
                *dst = (uchar3 *)src.data;
            }

            bool saveJPGRGBCPU(std::string path, cv::cuda::GpuMat &image, int quality, bool sync)
            {
                uchar3 *out = NULL;
                cvImg2uchar3(image, &out);

                if (out == nullptr)
                {
                    return false;
                }

                return saveImage(path.c_str(), out, image.size().width, image.size().height, quality, make_float2(0, 255), sync);
            }

            bool saveJPGBGRCPU(std::string path, cv::cuda::GpuMat &image, int quality, bool sync)
            {
                cv::cuda::GpuMat gpu_frame_rgb(image.size(), image.type());
                cv::cuda::cvtColor(image, gpu_frame_rgb, cv::COLOR_BGR2RGB);
                return saveJPGRGBCPU(path, gpu_frame_rgb, quality, sync);
            }

            GPUJpgEncoder::~GPUJpgEncoder()
            {
                jpeg_destroy_compress((struct jpeg_compress_struct *)cinfo);
                delete (struct jpeg_compress_struct *)cinfo;
                delete (struct jpeg_error_mgr *)jerr;
                ((npp::NPPJpegCoder *)nppiEncoder)->release();
                delete (npp::NPPJpegCoder *)nppiEncoder;
                CUDA_FREE_HOST(i420_out);
                delete yuv_data;
            }

            GPUJpgEncoder::GPUJpgEncoder(int w, int h, int quality) : w(w), h(h)
            {
                cinfo = new struct jpeg_compress_struct;
                jerr = new struct jpeg_error_mgr;
                memset(cinfo, 0, sizeof(struct jpeg_compress_struct));
                memset(jerr, 0, sizeof(struct jpeg_error_mgr));
                ((struct jpeg_compress_struct *)cinfo)->err = jpeg_std_error((struct jpeg_error_mgr *)jerr);

                jpeg_create_compress((struct jpeg_compress_struct *)cinfo);
                jpeg_suppress_tables((struct jpeg_compress_struct *)cinfo, TRUE);

                nppiEncoder = new npp::NPPJpegCoder();
                npp::NPPJpegCoder *encoder = (npp::NPPJpegCoder *)nppiEncoder;
                encoder->init(w, h, quality);

                size_t yuv_data_size = w * h * 3 / 2;
                cudaAllocMapped((void **)&i420_out, yuv_data_size);

                yuv_data = new char[yuv_data_size];
            }

            bool read_video_frame(const char *inpBuf, unsigned inpBufLen, NvBuffer &buffer)
            {
                uint32_t i, j;
                char *data;

                for (i = 0; i < buffer.n_planes; i++)
                {
                    NvBuffer::NvBufferPlane &plane = buffer.planes[i];
                    std::streamsize bytes_to_read = plane.fmt.bytesperpixel * plane.fmt.width;
                    data = (char *)plane.data;
                    plane.bytesused = 0;
                    for (j = 0; j < plane.fmt.height; j++)
                    {
                        unsigned numRead = std::min((unsigned)bytes_to_read, (unsigned)inpBufLen);

                        memcpy(data, inpBuf, numRead);

                        if (numRead < bytes_to_read)
                        {
                            return false;
                        }

                        inpBuf += numRead;
                        inpBufLen -= numRead;

                        data += plane.fmt.stride;
                    }
                    plane.bytesused = plane.fmt.stride * plane.fmt.height;
                }
                return true;
            }
            /*
                        int encodeJpg(cv::cuda::GpuMat &image, uint8_t *image_compressed, int *image_compressed_size, int quality)
                        {
                            struct gpujpeg_parameters param;
                            gpujpeg_set_default_parameters(&param);
                            param.quality = 80;
                            param.restart_interval = 16;
                            param.interleaved = 0;

                            struct gpujpeg_image_parameters param_image;
                            gpujpeg_image_set_default_parameters(&param_image);
                            param_image.width = image.size().width;
                            param_image.height = image.size().height;
                            param_image.comp_count = 3;
                            param_image.color_space = GPUJPEG_RGB;
                            param_image.pixel_format = GPUJPEG_444_U8_P012;

                            int device_id = 0;
                            int verbose_init = 0; // or GPUJPEG_VERBOSE

                            if (gpujpeg_init_device(device_id, verbose_init))
                            {
                                return -1;
                            }

                            struct gpujpeg_encoder *encoder = gpujpeg_encoder_create(0);
                            if (encoder == NULL)
                            {
                                return -1;
                            }

                            cv::Mat cpu_img_rgb;
                            image.download(cpu_img_rgb);

                            struct gpujpeg_encoder_input encoder_input;
                            encoder_input.type = gpujpeg_encoder_input_type::GPUJPEG_ENCODER_INPUT_GPU_IMAGE;
                            encoder_input.image = cpu_img_rgb.data;

                            if (gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input, &image_compressed,
                                                       image_compressed_size) != 0)
                            {
                                return -1;
                            }

                            return 0;
                        }
                        */

            uchar *GPUJpgEncoder::EncodeRGBnv(cv::cuda::GpuMat &image, unsigned long &outBufSize, int quality, bool cudaColorI420)
            {
                /*
                cv::Mat cpu;
                image.download(cpu);
                Npp8u *rgb_img_d = cpu.data();
                Npp8u *apDstImage[3];

                NPP_CHECK_NPP(nppiRGBToYUV420_8u_C3P3R(image.data, image.step, apDstImage, aDstImageStep,
                                                       osize));
                                                       */

                npp::NPPJpegCoder *encoder = (npp::NPPJpegCoder *)nppiEncoder;

                size_t jpegDataBufSize = image.size().width * image.size().height * 3;
                uchar *jpegData = new uchar[jpegDataBufSize];

                cv::cuda::Stream stream;
                auto start_getnext = high_resolution_clock::now();

                int ret = encoder->encode(image, jpegData, &outBufSize, jpegDataBufSize, stream);

                stream.waitForCompletion();
                if (ret != 0)
                {
                    std::cout << "error in encode" << std::endl;
                }
                auto end_getnext = high_resolution_clock::now();
                std::cout << "encoded with nppi " << duration_cast<milliseconds>(end_getnext - start_getnext).count() << "ms\n";

                return jpegData;
            }

            uchar *GPUJpgEncoder::EncodeRGB(cv::cuda::GpuMat &image, unsigned long &outBufSize, int quality, bool cudaColorI420)
            {
                // cv::Size downscale(1920, 1080);
                //  cv::cuda::GpuMat resized(image.size(), CV_8UC3);
                //  cv::cuda::resize(image, resized, cv::Size(1600, 900));
                //  uchar3 *resizedPtr = 0;
                //  cudaAllocMapped((void **)&resizedPtr, downscale.width * downscale.height * sizeof(uchar3));
                //  cudaResize((uchar3 *)image.cudaPtr(), image.size().width, image.size().height, resizedPtr, downscale.width, downscale.height);
                //  cv::cuda::GpuMat resized(downscale, CV_8UC3, (void *)resizedPtr);

                size_t yuv_data_size = w * h * 3 / 2;

                cv::Mat cpu_img_yuv(image.size(), CV_8UC1);
                auto start_encode = high_resolution_clock::now();
                if (cudaColorI420)
                {
                    if (CUDA_FAILED(cudaRGBToI420((uchar3 *)image.cudaPtr(), i420_out, w, h)))
                    {
                        std::cout << "failed color conv\n";
                        return 0;
                    }

                    yuv_data = (char *)i420_out;

                    // cudaMemcpy(yuv_data, i420_out, yuv_data_size, cudaMemcpyDeviceToHost);
                    //  memcpy(yuv_data, i420_out, yuv_data_size);
                }
                else
                {
                    auto start_encode = high_resolution_clock::now();
                    cv::Mat cpu_img_rgb;
                    image.download(cpu_img_rgb);
                    cv::cvtColor(cpu_img_rgb, cpu_img_yuv, cv::COLOR_RGB2YUV_I420);
                    yuv_data = (char *)cpu_img_yuv.ptr();
                    auto end_encode = high_resolution_clock::now();
                    std::cout << "cpu conv " << duration_cast<milliseconds>(end_encode - start_encode).count() << "ms\n\n";
                }

                NvBuffer nvbuf(V4L2_PIX_FMT_YUV420M, w, h, 0);
                nvbuf.allocateMemory();
                auto ret = read_video_frame(yuv_data, yuv_data_size, nvbuf);
                if (ret < 0)
                {
                    return 0;
                }

                outBufSize = yuv_data_size;
                uchar *outBufLocal = new uchar[outBufSize];

                std::unique_ptr<NvJPEGEncoder> jpgEnc(NvJPEGEncoder::createJPEGEncoder("jpegenc"));
                ret = jpgEnc->encodeFromBuffer(nvbuf, JCS_YCbCr, &outBufLocal, outBufSize, quality);

                return outBufLocal;
            }
        }
    }
}
