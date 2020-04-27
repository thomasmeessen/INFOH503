#ifndef TOOLS_HEADER_FILE
#define TOOLS_HEADER_FILE

#include <string>


struct opencl_buffer;
struct opencl_stuff;

void print_device_info(cl_device_id);
void show_build_error(const std::string*, cl_program*, cl_device_id);
void compile_source(const std::string*, cl_program*, cl_device_id, cl_context);
void image_padding(cv::Mat&, cv::Mat&, int);
cv::Mat guidedFilter(cv::Mat&, int, cl_context, cl_kernel, cl_kernel, cl_command_queue, struct opencl_buffer, opencl_stuff, const std::string*);
void image_difference(cv::Mat&, cv::Mat&, cv::Mat&, int, cl_context, cl_kernel, cl_command_queue, bool);
opencl_buffer cost_by_layer(cv::Mat, cv::Mat, int, cl_device_id, cl_context, cl_command_queue);
opencl_buffer cost_by_layer(cv::Mat, cv::Mat, int, opencl_stuff);
void cost_selection(cv::Mat, int, cl_kernel, opencl_stuff, const std::string*);

#endif