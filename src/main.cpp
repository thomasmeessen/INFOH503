
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "tools.cpp"

using namespace std;


#define MAX_DISTANCE 16 // maximum differnenc in pixel

// A simple threshold kernel
const string greyscale_source_path = "greyscale.cl";
const string difference_image_source_path = "differenceImage.cl";
const string guidedFilter_source_path = "guidedFilter.cl";
const string left_image_path = "classroom_l.png";
const string right_image_path = "classroom_r.png";
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


    // 3. Create a context and command queue on that device.
    cl_context context = clCreateContext(NULL,
        1,
        &device,
        NULL, NULL, NULL);

    cl_command_queue queue = clCreateCommandQueue(context,
        device,
        0, NULL);


    cl_program cost_by_layer_program;
    compile_source(&cost_by_layer_source_path, &cost_by_layer_program, device, context);
    cl_kernel cost_by_layer_kernel = clCreateKernel (cost_by_layer_program, "memset", NULL);


    //--------------------------------------------
    // Layer cost computation
    opencl_stuff ocl_stuff;
    ocl_stuff.device = device;
    ocl_stuff.context = context;
    ocl_stuff.queue = queue;

    opencl_buffer cost_layer = cost_by_layer(left_image_path, right_image_path, 25, ocl_stuff);
    cost_layer.write_img("Cost_for_layer_20.png", ocl_stuff);

    return 0;
}