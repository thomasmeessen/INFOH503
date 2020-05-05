
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
const string cost_by_layer_source_path = "cost_volume.cl";
cl_program cost_by_layer_program;
cl_program guidedFilterStart_program;
cl_program guidedFilterEnd_program;
cl_program disparity_selection_program;
cl_kernel cost_volume_kernel;
cl_kernel guidedFilter_kernel;
cl_kernel guidedFilterEnd_kernel;
cl_kernel disparity_selection_kernel;
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
    // Cost kernel
    compile_source(&cost_by_layer_source_path, &cost_by_layer_program, ocl_stuff.device, ocl_stuff.context);
    cost_volume_kernel = clCreateKernel(cost_by_layer_program, "cost_volume_in_range", NULL);
    //guidedFilter Start
    compile_source(&guidedFilter_source_path, &guidedFilterStart_program, ocl_stuff.device, ocl_stuff.context);
    guidedFilter_kernel = clCreateKernel(guidedFilterStart_program, "memset", NULL );
    
    compile_source(&guidedFilterEnd_source_path, &guidedFilterEnd_program, ocl_stuff.device, ocl_stuff.context);
    guidedFilterEnd_kernel = clCreateKernel(guidedFilterEnd_program, "memset", NULL);
    
    compile_source(&disparity_selection_source_path, &disparity_selection_program, ocl_stuff.device, ocl_stuff.context);
    disparity_selection_kernel = clCreateKernel(disparity_selection_program, "memset", NULL);


}

Opencl_buffer compute_depth_map(const string &base_image_path, const string &compared_image_path, int disparity){
    string indicator = "_L_"; // Left image
    int disparity_sign = disparity/abs(disparity);
    cv::Mat base_source_image = cv::imread(base_image_path, cv::IMREAD_GRAYSCALE);

    Opencl_buffer cost_layer = cost_range_layer(base_image_path, compared_image_path, MAX_DISTANCE, disparity_sign, cost_volume_kernel, ocl_stuff);
    Opencl_buffer filtered_cost = guidedFilter(base_source_image, MAX_DISTANCE, ocl_stuff.context, guidedFilter_kernel, guidedFilterEnd_kernel, ocl_stuff.queue, cost_layer, ocl_stuff, &left_image_path);
    Opencl_buffer depth_map = cost_selection(filtered_cost, MAX_DISTANCE, disparity_selection_kernel, ocl_stuff, &left_image_path);
    
    if(disparity < 0)
        indicator = "_R_";
    cost_layer.write_img("Cost_for_layer_normalized" + indicator + ".png", ocl_stuff, true);
    filtered_cost.write_img("filtered_cost" + indicator + ".png" , ocl_stuff, true);
    depth_map.write_img("depth_map_" + indicator + left_image_path, ocl_stuff, true);
    return cost_layer;
}

int main(int argc, char** argv)
{
    set_up();
    compile_sources();
    Opencl_buffer left_depth_map = compute_depth_map(left_image_path, right_image_path, MAX_DISTANCE);  // must return depth map
    Opencl_buffer right_depth_map =  compute_depth_map(right_image_path, left_image_path, -MAX_DISTANCE);

    opencl_buffer consistent_depht_map = left_right_consistency(left_depth_map, right_depth_map);
    //densification(consistent_depht_map);
    

    return 0;
}