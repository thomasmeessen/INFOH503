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
enum Movement_direction {L_to_r, R_to_l};
Opencl_buffer cost_range_layer(cv::Mat, cv::Mat, int, cl_device_id, cl_context, cl_command_queue);
Opencl_buffer cost_range_layer(cv::Mat, cv::Mat, int, Opencl_stuff);
Opencl_buffer cost_selection(Opencl_buffer, int, cl_kernel, Opencl_stuff, const std::string*);
Opencl_buffer left_right_consistency(Opencl_buffer, Opencl_buffer, cl_kernel, Opencl_stuff);
void densification(Opencl_buffer , Opencl_buffer, cl_kernel, Opencl_stuff);
#endif