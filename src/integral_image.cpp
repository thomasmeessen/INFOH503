#include "integral_image.h"
#include "ocl_wrapper.h"
#include "tools.h"
#include <iostream>
#include <CL/cl.h>

using namespace std;
const string scan_kernel_path = "scan.cl";
const string transpose_kernel_path = "transpose.cl";

ScanParameters::ScanParameters(const Opencl_buffer &image, const Opencl_stuff &ocl_stuff){
    int number_pixels = image.rows * image.cols;
    size_t max_workgroup_dimensions[3];
    clGetDeviceInfo(ocl_stuff.device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_workgroup_dimensions), &max_workgroup_dimensions, NULL);
    this->local_size = max_workgroup_dimensions[0];
    size_t max_work_group_size;
    clGetDeviceInfo(ocl_stuff.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    int max_pixel_per_bloc = min(max_work_group_size, max_workgroup_dimensions[0]);

    this->number_blocs = (number_pixels / 2) / max_pixel_per_bloc;
    cl_uint max_number_blocs;
    clGetDeviceInfo(ocl_stuff.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_number_blocs), &max_number_blocs, NULL);


    cout << "Number of pixels: " << number_pixels <<endl;
    cout << "Maximum number of work item in a work group " << (int) max_work_group_size <<endl;
    cout << "Maximum dimensions of work item in a work group x = " << (int) max_workgroup_dimensions[0] <<endl;
    cout << "Max number of blocs = " << (int) max_number_blocs << endl;
    cout << "Number of blocs = " << this -> number_blocs <<endl;
}

void compute_integral_image(const Opencl_buffer &image, const Opencl_stuff &ocl_stuff) {
    // - Compile source
    cl_int error1, error2;
    cl_program  scan_program;
    compile_source(&scan_kernel_path, &scan_program, ocl_stuff.device, ocl_stuff.context);
    cl_kernel scan_kernel = clCreateKernel(scan_program, "scan", &error1);
    cl_program transpose_program;
    compile_source(&transpose_kernel_path, &transpose_program, ocl_stuff.device, ocl_stuff.context);
    cl_kernel transpose_kernel = clCreateKernel(scan_program, "scan", &error2);
    if ( (error1 != CL_SUCCESS) or (error2 != CL_SUCCESS)){
        throw "Error during kernel compilation for scan algorithm";
    }

    // - Choose bloc size
    ScanParameters scan_parameter(image, ocl_stuff);
    // -- Reserve buffer for the intermediate result
    Opencl_buffer intermediate_result(image.rows, image.cols, ocl_stuff);

    // - Apply Scan on the rows of the image
    clSetKernelArg(scan_kernel, 0, sizeof(image.buffer), (void*)&image.buffer);
    clSetKernelArg(scan_kernel, 1, sizeof(intermediate_result.buffer), (void*)&intermediate_result.buffer);
    clSetKernelArg(scan_kernel, 2, scan_parameter.local_size * sizeof(float), (void*)nullptr);
    // -- The number of thread is half the number of pixel
    size_t global_work_size_image[] = {(size_t) (image.cols * image.rows) /2};

    clEnqueueNDRangeKernel(ocl_stuff.queue,
                           scan_kernel,
                           1,
                           NULL,
                           global_work_size_image,
                           NULL,
                           0,
                           NULL, NULL);

    clFinish(ocl_stuff.queue);
    // -- Apply the kernel that compute the integral for a bloc

    // -- Apply the kernel that integrate all the blocs

    // - Transpose the intermediate result

    // -- Apply Scan on the row of the transposed intermediate result

    // -- Apply the kernel that integrate all blocs

    // - Compute the window average

    // - Compute the window variance
}
