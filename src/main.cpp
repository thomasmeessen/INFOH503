
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "tools.cpp"
#include "ocl_wrapper.h"

using namespace std;





// A simple threshold kernel
const string greyscale_source_path = "greyscale.cl";
const string difference_image_source_path = "differenceImage.cl";
const string guidedFilter_source_path = "guidedFilterStart.cl";
const string guidedFilterEnd_source_path = "guidedFilterEnd.cl";
const string disparity_selection_source_path = "disparity_selection.cl";
const string left_right_consistency_source_path = "left_right_consistency.cl";
const string densification_source_path = "densification.cl";
const string left_image_path = "paper1.png";
const string right_image_path = "paper0.png";
//const string left_image_path = "classroom_l.png";
//const string right_image_path = "classroom_r.png";
const string cost_by_layer_source_path = "cost_volume.cl";
cl_program cost_by_layer_program;
cl_program guidedFilterStart_program;
cl_program guidedFilterEnd_program;
cl_program disparity_selection_program;
cl_program left_right_consistency_program;
cl_program densification_program;
cl_kernel cost_volume_kernel;
cl_kernel guidedFilter_kernel;
cl_kernel guidedFilterEnd_kernel;
cl_kernel disparity_selection_kernel;
cl_kernel left_right_consistency_kernel;
cl_kernel densification_kernel;
Opencl_stuff ocl_stuff;

void set_up(){
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

    ocl_stuff.device = device;
    ocl_stuff.context = context;
    ocl_stuff.queue = queue;

}
void compile_sources(){
    cl_int error;
    // Cost kernel
    compile_source(&cost_by_layer_source_path, &cost_by_layer_program, ocl_stuff.device, ocl_stuff.context);
    cost_volume_kernel = clCreateKernel(cost_by_layer_program, "cost_volume_in_range", &error);
    if(error != CL_SUCCESS)
        printf("error with cost_volume with error code : %i. list of error msgs : https://streamhpc.com/blog/2013-04-28/opencl-error-codes/\n", error);
    //guidedFilter Start
    compile_source(&guidedFilter_source_path, &guidedFilterStart_program, ocl_stuff.device, ocl_stuff.context);
    guidedFilter_kernel = clCreateKernel(guidedFilterStart_program, "memset", &error );
    if(error != CL_SUCCESS)
        printf("error with guidedFIlterStart with error code : %i. list of error msgs : https://streamhpc.com/blog/2013-04-28/opencl-error-codes/\n", error);

    compile_source(&guidedFilterEnd_source_path, &guidedFilterEnd_program, ocl_stuff.device, ocl_stuff.context);
    guidedFilterEnd_kernel = clCreateKernel(guidedFilterEnd_program, "memset", &error);
    if(error != CL_SUCCESS)
        printf("error with guidedFilterEndwith error code : %i. list of error msgs : https://streamhpc.com/blog/2013-04-28/opencl-error-codes/\n", error);

    compile_source(&disparity_selection_source_path, &disparity_selection_program, ocl_stuff.device, ocl_stuff.context);
    disparity_selection_kernel = clCreateKernel(disparity_selection_program, "memset", &error);
    if(error != CL_SUCCESS)
        printf("error with DIsparity with error code : %i. list of error msgs : https://streamhpc.com/blog/2013-04-28/opencl-error-codes/\n", error);

    compile_source(&left_right_consistency_source_path, &left_right_consistency_program, ocl_stuff.device, ocl_stuff.context);
    left_right_consistency_kernel = clCreateKernel(left_right_consistency_program, "consistency", &error);
    if(error != CL_SUCCESS)
        printf("error with left right consistency with error code : %i. list of error msgs : https://streamhpc.com/blog/2013-04-28/opencl-error-codes/\n", error);

    compile_source(&densification_source_path, &densification_program, ocl_stuff.device, ocl_stuff.context);
    densification_kernel = clCreateKernel(densification_program, "densification", &error);
    if(error != CL_SUCCESS)
        printf("error with left right consistency with error code : %i. list of error msgs : https://streamhpc.com/blog/2013-04-28/opencl-error-codes/\n", error);

}

Opencl_buffer compute_depth_map(const string &base_image_path, const string &compared_image_path, int disparity){
    string indicator = (disparity < 0) ? "R" : "L";

    int disparity_sign = disparity/abs(disparity);
    cv::Mat base_source_image = cv::imread(base_image_path, cv::IMREAD_GRAYSCALE);

    Opencl_buffer cost_layer = cost_range_layer(base_image_path, compared_image_path, MAX_DISTANCE, disparity_sign, cost_volume_kernel, ocl_stuff);
    cost_layer.write_img("Cost_for_layer_normalized_" + indicator + "_.png", ocl_stuff, true);
    
    printf("Cost %s done\n", indicator.c_str());
    Opencl_buffer filtered_cost = guidedFilter(base_source_image, MAX_DISTANCE, ocl_stuff.context, guidedFilter_kernel, guidedFilterEnd_kernel, ocl_stuff.queue, cost_layer, ocl_stuff, &base_image_path);
    filtered_cost.write_img("filtered_cost_" + indicator + "_.png" , ocl_stuff, true);
    printf("Filtering %s done\n", indicator.c_str());
    
    Opencl_buffer depth_map = cost_selection(filtered_cost, MAX_DISTANCE, disparity_selection_kernel, ocl_stuff, &base_image_path);
    depth_map.write_img("depth_map_" + indicator + "_" + base_image_path, ocl_stuff, true);
    printf("Depth map %s done \n", indicator.c_str());
    
    return depth_map;
}

int main(int argc, char** argv)
{
    set_up();
    compile_sources();
    Opencl_buffer left_depth_map = compute_depth_map(left_image_path, right_image_path, MAX_DISTANCE);  // must return depth map
    Opencl_buffer right_depth_map =  compute_depth_map(right_image_path, left_image_path, -MAX_DISTANCE);

    Opencl_buffer consistent_depth_map = left_right_consistency(left_depth_map, right_depth_map, left_right_consistency_kernel, ocl_stuff);
    printf("Left Right consistency done\n");
    consistent_depth_map.write_img((string)"consistentcy_output.png", ocl_stuff, true);
    
    densification(left_depth_map, consistent_depth_map, densification_kernel, ocl_stuff);
    printf("densification/filling done\n");
    left_depth_map.write_img((string)"densification_output.png", ocl_stuff, true);

    return 0;
}