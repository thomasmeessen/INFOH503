#ifndef TOOLS_HEADER_FILE
#define TOOLS_HEADER_FILE

#include <string>
#include "ocl_wrapper.h"


void print_device_info(cl_device_id);
void show_build_error(const std::string*, cl_program*, cl_device_id);
void compile_source(const std::string*, cl_program*, cl_device_id, cl_context);
void image_padding(cv::Mat&, cv::Mat&, int);
Opencl_buffer guidedFilter(cv::Mat&, int, cl_context, cl_kernel, cl_kernel, cl_command_queue, struct Opencl_buffer, Opencl_stuff, const std::string*);
void image_difference(cv::Mat&, cv::Mat&, cv::Mat&, int, cl_context, cl_kernel, cl_command_queue, bool);
Opencl_buffer cost_range_layer(cv::Mat left_source_image, cv::Mat right_source_image, int disparity_range, cl_device_id device, cl_context context, cl_command_queue queue);
Opencl_buffer cost_range_layer(cv::Mat left_source_image, cv::Mat right_source_image, int disparity, Opencl_stuff ocl_stuff);
Opencl_buffer cost_selection(Opencl_buffer filtered_cost_buffer, int, cl_kernel, Opencl_stuff, const std::string*);

#endif