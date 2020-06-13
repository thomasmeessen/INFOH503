#ifndef BENCHMARK_INCLUDE
#define BENCHMARK_INCLUDE

#include <opencv2/core.hpp>

#define BASE_SIZE  1000
#define MAX_FACTOR  7

cv::Mat generate_Image(int size_factor);

void run_integral_image_benchmark();

#endif