
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

    cl_program cost_by_layer_program;
    compile_source(&cost_by_layer_source_path, &cost_by_layer_program, device, context);
    cl_kernel cost_by_layer_kernel = clCreateKernel (cost_by_layer_program, "memset", NULL);

    cv::Mat left_image;
    to_greyscale_plus_padding(&left_image_path ,left_image  ,MAX_DISTANCE ,context, greyscale_kernel, queue, true);
    cv::Mat right_image;
    to_greyscale_plus_padding(&right_image_path,right_image ,MAX_DISTANCE, context, greyscale_kernel, queue, false);
    guidedFilter(&left_image_path ,left_image, MAX_DISTANCE, context, guidedFilter_kernel, queue, true);

    //--------------------------------------------
    // Layer cost computation
    int disparity_test =  1;
    int padding_size = MAX_DISTANCE;
    int number_rows_img = left_image.rows;
    int number_cols_img = left_image.cols;
    int channel_for_left = 0;
    int channel_for_right = 1;
    float alpha_weight = 0.5;

    // - Padding
    cv::Rect extract_zone (padding_size, padding_size, number_cols_img, number_rows_img);
    cv::Mat left_image_padded = cv::Mat::zeros(left_image.rows + 2*padding_size, left_image.cols + 2 * padding_size, CV_8UC1);
    left_image.copyTo(left_image_padded(extract_zone));
    cv::Mat right_image_padded  = cv::Mat::zeros(left_image.rows + 2*padding_size, left_image.cols + 2 * padding_size, CV_8UC1);
    right_image.copyTo(right_image_padded(extract_zone));
    cv::Mat output_layer_cost = cv::Mat::zeros(left_image.size(), CV_32FC1); // float

    // - Merging into a single matrix with entrelacement
    cv::Mat source_images_padded;
    cv::Mat temp_array[2]  = {left_image_padded, right_image_padded};
    cv::merge(temp_array, 2, source_images_padded);

    // - Allocating the buffers
    cl_mem cost_input_buffer = clCreateBuffer(context,
                                          CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                          source_images_padded.total() * source_images_padded.elemSize(),
                                          (void*)source_images_padded.data, NULL);
    cl_mem cost_output_buffer = clCreateBuffer(context,
                                        CL_MEM_WRITE_ONLY,
                                        output_layer_cost.total() * output_layer_cost.elemSize(),
                                        NULL, NULL);
    // - Passing arguments to the kernel
    clSetKernelArg(cost_by_layer_kernel, 0, sizeof(cost_input_buffer), (void*)&cost_input_buffer);
    clSetKernelArg(cost_by_layer_kernel, 1, sizeof(cost_output_buffer), (void*)&cost_output_buffer);
    clSetKernelArg(cost_by_layer_kernel, 2, sizeof(padding_size), (void*)&padding_size);
    clSetKernelArg(cost_by_layer_kernel, 3, sizeof(disparity_test), (void*)&disparity_test);
    clSetKernelArg(cost_by_layer_kernel, 4, sizeof(alpha_weight), (void*)&alpha_weight);
    // - Enqueuing kernel
    size_t global_work_size_cost_layer[] = {right_image.total() };
    clEnqueueNDRangeKernel(queue,
                           cost_by_layer_kernel,
                           2,
                           NULL,
                           global_work_size_cost_layer,
                           NULL,
                           0,
                           NULL, NULL);
    // - Waiting end execution
    clFinish(queue);
    cout<<"dinisf"<<endl;
//
//    //--------------------------------------------
//    //--------Difference Image Kernel-------------
//    //--------------------------------------------
//    //now source image == the greysclae image
//
//    cl_program difference_image_program;
//    compile_source(&difference_image_source_path, &difference_image_program, device, context);
//
//    cl_kernel difference_image_kernel = clCreateKernel(difference_image_program, "memset", NULL);
//
//    cv::Mat output_image; // each image will be next to each other?
//    image_difference(left_image, right_image, output_image, MAX_DISTANCE, context, difference_image_kernel, queue, false);

    left_image.release();
    right_image.release();
//    output_image.release();
    return 0;
}