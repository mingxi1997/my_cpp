cmake_minimum_required(VERSION 3.1)
project(test)
set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED )
#list(APPEND PLUGINS "nvinfer")
#list(APPEND PLUGINS "nvonnxparser")
#list(APPEND PLUGINS "nvparsers")

#foreach(libName ${PLUGINS})
#    find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIXES lib)
#    list(APPEND PLUGIN_LIBS "${${libName}_lib}")
#endforeach()


set(TENSORRT_INCLUDE_PATH /home/xu/TensorRT-7.2.2.3/include)
set(COMMEN_PATH /home/xu/TensorRT-7.2.2.3/samples/common)
set(TENSORRT_LIB_PATH /home/xu/TensorRT-7.2.2.3/lib)

file(GLOB LIBS ${TENSORRT_LIB_PATH}/*.so)
file(GLOB COMMENCPPS ${COMMEN_PATH}/*.cpp)


include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${COMMEN_PATH})
include_directories(${TENSORRT_INCLUDE_PATH})
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(test trt.cpp ${COMMENCPPS})
target_link_libraries(test  ${LIBS} ${CUDA_LIBRARIES} ${OpenCV_LIBS})






