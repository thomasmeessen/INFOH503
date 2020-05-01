
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "tools.cpp"

using namespace std;


#define MAX_DISTANCE 30 // maximum differnenc in pixel



// A simple threshold kernel
const string greyscale_source_path = "greyscale.cl";
const string difference_image_source_path = "differenceImage.cl";
const string guidedFilter_source_path = "guidedFilterStart.cl";
const string guidedFilterEnd_source_path = "guidedFilterEnd.cl";
const string disparity_selection_source_path = "disparity_selection.cl";
const string left_image_path = "paper1.png";
const string right_image_path = "paper0.png";
//const string left_image_path = "classroom_l.png";
//const string right_image_path = "classroom_r.png";
const string cost_by_layer_source_path = "cost_volume_by_layer.cl";



int main(int argc, char** argv)
{
    // 1. Get a platform.
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    // 2. Find a gpu device.
    cl_device_id device;
    clGetDeviceIDs(platform,
        CL_DEVICE_TYPE_GPU,
        1,
        &device, NULL);
    print_device_info(device);

    // 3. Create a context and command queue on that device.
    cl_context context = clCreateContext(NULL,
        1,
        &device,
        NULL, NULL, NULL);

    cl_command_queue queue = clCreateCommandQueue(context,
        device,
        0, NULL);


    //--------------------------------------------
    // Layer cost computation
    Opencl_stuff ocl_stuff;
    ocl_stuff.device = device;
    ocl_stuff.context = context;
    ocl_stuff.queue = queue;



    Opencl_buffer cost_layer = cost_range_layer(left_image_path, right_image_path, MAX_DISTANCE, ocl_stuff);
    cost_layer.write_img("Cost_for_layer_normalized.png", ocl_stuff, true);
    cost_layer.write_img("Cost_for_layer.png", ocl_stuff, false);




    cl_program guidedFilterStart_program;
    compile_source(&guidedFilter_source_path, &guidedFilterStart_program, device, context);
    cl_kernel guidedFilter_kernel = clCreateKernel(guidedFilterStart_program, "memset", NULL );


    cl_program guidedFilterEnd_program;
    compile_source(&guidedFilterEnd_source_path, &guidedFilterEnd_program, device, context);
    cl_kernel guidedFilterEnd_kernel = clCreateKernel(guidedFilterEnd_program, "memset", NULL);

    cl_program disparity_selection_program;
    compile_source(&disparity_selection_source_path, &disparity_selection_program, device, context);
    cl_kernel disparity_selection_kernel = clCreateKernel(disparity_selection_program, "memset", NULL);

    cv::Mat left_source_image = cv::imread(left_image_path,cv::IMREAD_GRAYSCALE);
    Opencl_buffer filtered_cost = guidedFilter(left_source_image, MAX_DISTANCE, context, guidedFilter_kernel, guidedFilterEnd_kernel, queue, cost_layer, ocl_stuff, &left_image_path);
    filtered_cost.write_img("filtered_cost.png", ocl_stuff, true);
    auto depth_map = cost_selection(filtered_cost, MAX_DISTANCE, disparity_selection_kernel, ocl_stuff, &left_image_path);

    depth_map.write_img("depth_map_filtered_" + left_image_path, ocl_stuff, true);


    return 0;
}