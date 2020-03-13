cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(infoH503 CXX CUDA)


include(ExternalProject)

include(${CMAKE_CURRENT_SOURCE_DIR}/External-CImage.cmake)


add_executable(hello src/main.cu)

add_dependencies(hello CImg)


