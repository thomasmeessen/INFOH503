
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

    cl_program greyscale_program;
    compile_source(&greyscale_source_path, &greyscale_program, device, context);
    cl_kernel greyscale_kernel = clCreateKernel(greyscale_program, "memset", NULL);

    cl_program guidedFilter_program;
    compile_source(&guidedFilter_source_path, &guidedFilter_program, device, context);
    cl_kernel guidedFilter_kernel = clCreateKernel( guidedFilter_program, "memset", NULL );

    cv::Mat left_image;
    to_greyscale_plus_padding(&left_image_path ,left_image  ,MAX_DISTANCE ,context, greyscale_kernel, queue, true);
    cv::Mat right_image;
    to_greyscale_plus_padding(&right_image_path,right_image ,MAX_DISTANCE, context, greyscale_kernel, queue, false);
    guidedFilter(&left_image_path ,left_image, MAX_DISTANCE, context, guidedFilter_kernel, queue, true);


 //--------------------------------------------
 //--------Difference Image Kernel-------------
 //--------------------------------------------
     //now source image == the greysclae image

    cl_program difference_image_program;
    compile_source(&difference_image_source_path, &difference_image_program, device, context);

    cl_kernel difference_image_kernel = clCreateKernel(difference_image_program, "memset", NULL);

    cv::Mat output_image; // each image will be next to each other?
    image_difference(left_image, right_image, output_image, MAX_DISTANCE, context, difference_image_kernel, queue, false);

    left_image.release();
    right_image.release();
    output_image.release();
    return 0;
}