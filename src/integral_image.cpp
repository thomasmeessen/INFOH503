#include "integral_image.h"
#include "ocl_wrapper.h"
#include <iostream>
#include <CL/cl.h>

using namespace std;

ScanParameters::ScanParameters(const Opencl_buffer &image, const Opencl_stuff &ocl_stuff){
    int number_pixels = image.rows * image.cols;
    size_t max_workgroup_dimensions[3];
    clGetDeviceInfo(ocl_stuff.device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_workgroup_dimensions), &max_workgroup_dimensions, NULL);
    size_t max_work_group_size;
    clGetDeviceInfo(ocl_stuff.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    int max_pixel_per_bloc = min(max_work_group_size, max_workgroup_dimensions[0]);

    this->number_blocs = number_pixels / max_pixel_per_bloc;
    cl_uint max_number_blocs;
    clGetDeviceInfo(ocl_stuff.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_number_blocs), &max_number_blocs, NULL);


    cout << "Number of pixels: " << number_pixels <<endl;
    cout << "Maximum number of work item in a work group " << (int) max_work_group_size <<endl;
    cout << "Maximum dimensions of work item in a work group x = " << (int) max_workgroup_dimensions[0] <<endl;
    cout << "Max number of blocs = " << (int) max_number_blocs << endl;
    cout << "Number of blocs = " << (float)number_pixels / (float)max_pixel_per_bloc<<endl;
}

void compute_integral_image(const Opencl_buffer &image, const Opencl_stuff &ocl_stuff) {
    ScanParameters scan_parameter(image, ocl_stuff);
}
