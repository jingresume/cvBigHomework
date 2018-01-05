cmake_minimum_required(VERSION 2.6)
project(cvbighomework)

set (CMAKE_CXX_FLAGS "-std=c++11 -O3")
#opencv
find_package( OpenCV  3.2  REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )



add_executable(cvbighomework main.cpp)
target_link_libraries ( cvbighomework ${OpenCV_LIBS} )

