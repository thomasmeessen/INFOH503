#ifndef BENCHMARK_INCLUDE
#define BENCHMARK_INCLUDE

#include <opencv2/core.hpp>
#include "ocl_wrapper.h"

#define BASE_SIZE  1000
#define MAX_FACTOR  15

void cpu_integral_image(cv::Mat);

void run_integral_image_benchmark(Opencl_stuff ocl_stuff);

#endif