
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
const string guidedFilter_source_path = "guidedFilterStart.cl";
const string guidedFilterEnd_source_path = "guidedFilterEnd.cl";
const string left_image_path = "paper0.png";
const string right_image_path = "paper1.png";
// const string left_image_path = "classroom_l.png";
// const string right_image_path = "classroom_r.png";
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
    // 2.1 Check if the device tolerate images
    //cl_bool param_value_image_accepted;
    //clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), (void *)param_value_image_accepted, NULL);
    //cout <<( (param_value_image_accepted == CL_TRUE)? "Device accept image ": "Device do not accept image") << endl;

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
    cl_kernel cost_by_layer_kernel = clCreateKernel(cost_by_layer_program, "memset", NULL);


    //--------------------------------------------
    // Layer cost computation
    opencl_stuff ocl_stuff;
    ocl_stuff.device = device;
    ocl_stuff.context = context;
    ocl_stuff.queue = queue;


    opencl_buffer cost_layer = cost_by_layer(left_image_path, right_image_path, MAX_DISTANCE, ocl_stuff);
    cost_layer.write_img("Cost_for_layer.png", ocl_stuff, false);
    cost_layer.write_img("Cost_for_layer_normalized.png", ocl_stuff, true);




    cl_program greyscale_program;
    compile_source(&greyscale_source_path, &greyscale_program, device, context);
    cl_kernel greyscale_kernel = clCreateKernel(greyscale_program, "memset", NULL);

    cl_program guidedFilterStart_program;
    compile_source(&guidedFilter_source_path, &guidedFilterStart_program, device, context);
    cl_kernel guidedFilter_kernel = clCreateKernel(guidedFilterStart_program, "memset", NULL );


    cl_program guidedFilterEnd_program;
    compile_source(&guidedFilterEnd_source_path, &guidedFilterEnd_program, device, context);
    cl_kernel guidedFilterEnd_kernel = clCreateKernel(guidedFilterEnd_program, "memset", NULL);

    cv::Mat left_image;
    to_greyscale_plus_padding(&left_image_path ,left_image  ,MAX_DISTANCE ,context, greyscale_kernel, queue, true);
    guidedFilter(&left_image_path , MAX_DISTANCE, context, guidedFilter_kernel, guidedFilterEnd_kernel, queue, true, cost_layer, ocl_stuff);



    return 0;
}