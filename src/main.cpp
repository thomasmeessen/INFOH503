
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "tools.cpp"
#include "ocl_wrapper.h"

using namespace std;


#define MAX_DISTANCE 16 // maximum differnenc in pixel



// A simple threshold kernel
const string greyscale_source_path = "greyscale.cl";
const string difference_image_source_path = "differenceImage.cl";
const string guidedFilter_source_path = "guidedFilterStart.cl";
const string guidedFilterEnd_source_path = "guidedFilterEnd.cl";
const string disparity_selection_source_path = "disparity_selection.cl";
const string left_right_consistency_source_path = "left_right_consistency.cl";
const string densification_source_path = "densification.cl";
const string left_image_path = "paper0.png";
const string right_image_path = "paper1.png";
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
    clGetPlatformIDs(1, &platform, nullptr);

    // 2. Find a gpu device.
    cl_device_id device;
    clGetDeviceIDs(platform,
        CL_DEVICE_TYPE_GPU,
        1,
        &device, nullptr);
    print_device_info(device);

    // 3. Create a context and command queue on that device.
    cl_context context = clCreateContext(nullptr,
        1,
        &device,
        nullptr, nullptr, nullptr);

    cl_command_queue queue = clCreateCommandQueueWithProperties(context,
        device,
        nullptr, nullptr);

    ocl_stuff.device = device;
    ocl_stuff.context = context;
    ocl_stuff.queue = queue;
    // Used for throwing error when allocating more than available
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ocl_stuff.memory_available), &ocl_stuff.memory_available, nullptr);
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


Opencl_buffer compute_depth_map(const string &start_image_path, const string &end_image_path, int disparity_range, Movement_direction dir ){
    string indicator = Movement_direction::L_to_r == dir ? "L" : "R";
    int disparity_sign = Movement_direction::L_to_r == dir ? -1 : 1;

    Opencl_buffer cost_volume = cost_range_layer(start_image_path, end_image_path, disparity_range, cost_volume_kernel,
                                                 ocl_stuff, disparity_sign);
    cost_volume.write_img("Cost_for_layer_normalized_" + indicator + "_.png", true);
    
    printf("Cost %s done\n", indicator.c_str());
    Opencl_buffer filtered_cost = guidedFilter(start_image_path, disparity_range, guidedFilter_kernel,
                                               guidedFilterEnd_kernel, cost_volume, ocl_stuff);
    filtered_cost.write_img("filtered_cost_" + indicator + "_.png", true);
    printf("Filtering %s done\n", indicator.c_str());
    cost_volume.free();
    
    Opencl_buffer depth_map = cost_selection(filtered_cost, disparity_range, disparity_selection_kernel, ocl_stuff);
    depth_map.write_img("depth_map_" + indicator + "_" + start_image_path, true);
    printf("Depth map %s done \n", indicator.c_str());
    filtered_cost.free();
    
    return depth_map;
}

int main(int argc, char** argv)
{


    set_up();
    compile_sources();



    Opencl_buffer left_depth_map = compute_depth_map(left_image_path, right_image_path, MAX_DISTANCE, Movement_direction::L_to_r);  // must return depth map
    Opencl_buffer right_depth_map =  compute_depth_map(right_image_path, left_image_path, MAX_DISTANCE, Movement_direction::R_to_l);

    Opencl_buffer consistent_depth_map = left_right_consistency(left_depth_map, right_depth_map, left_right_consistency_kernel, ocl_stuff);
    printf("Left Right consistency done\n");
    consistent_depth_map.write_img((string) "consistentcy_output.png", true);
    
    densification(left_depth_map, consistent_depth_map, densification_kernel, ocl_stuff);
    printf("densification/filling done\n");
    left_depth_map.write_img((string) "densification_output.png", true);
    return 0;

}