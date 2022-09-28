cmake_minimum_required(VERSION 3.10)
project(flysense-jetson-cam VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(PROJECT_SOURCE_DIR src)

find_package(OpenCV REQUIRED)

#include_directories( ${OpenCV_INCLUDE_DIRS}  include )
#add_compile_options(-Ofast)
include_directories(include)

file(GLOB SOURCES
    ${PROJECT_SOURCE_DIR}/*.h
    ${PROJECT_SOURCE_DIR}/*.cc
)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NVJPEG_LIBRARY nvjpeg ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NPPIG_LIBRARY nppig ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NPPC_LIBRARY nppc ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CULIBOS culibos ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

# setup CUDA
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
find_package(CUDA)
message("-- CUDA version: ${CUDA_VERSION}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_53 SM_62")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62)

    if(CUDA_VERSION_MAJOR GREATER 9)
        message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_72")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_72,code=sm_72)
    endif()

    if(CUDA_VERSION_MAJOR GREATER 10)
        message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_87")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_87,code=sm_87)
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
/usr/src/tegra_multimedia_api/include
/usr/local/include/jetson-utils /usr/local/cuda/targets/aarch64-linux/include /usr/include/gstreamer-1.0 /usr/include/glib-2.0 /usr/lib/aarch64-linux-gnu/glib-2.0/include)

set(LIBRARY_SRC
    src/cam.cpp
)

add_library(${PROJECT_NAME} SHARED
    ${PROJECT_SOURCE_DIR}/cam.cc
    ${PROJECT_SOURCE_DIR}/NvBuffer.cpp
    ${PROJECT_SOURCE_DIR}/NvUtils.cpp
    ${PROJECT_SOURCE_DIR}/NvJpegEncoder.cpp
    ${PROJECT_SOURCE_DIR}/NvLogging.cpp
    ${PROJECT_SOURCE_DIR}/NvElement.cpp
    ${PROJECT_SOURCE_DIR}/NvElementProfiler.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    PUBLIC_HEADER include/cam.h
)


include(GNUInstallDirs)

add_custom_target(uninstall
    xargs rm < install_manifest.txt
)

install(TARGETS ${PROJECT_NAME} 
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/flysense-jetson-cam
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} jetson-utils cuda cudart /usr/lib/aarch64-linux-gnu/tegra/libnvjpeg.so 
#gpujpeg
#${CUDART_LIBRARY} ${NVJPEG_LIBRARY} ${NPPIG_LIBRARY} ${NPPC_LIBRARY} ${CULIBOS}
)
