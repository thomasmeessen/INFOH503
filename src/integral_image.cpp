#include "integral_image.h"
#include "ocl_wrapper.h"
#include "tools.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <CL/cl.h>
#include <math.h>

using namespace std;
const string scan_kernel_path = "scan.cl";
const string scan_integration_kernel_path = "scan_integration.cl";
const string transpose_kernel_path = "transpose.cl";


ScanParameters::ScanParameters(const Opencl_buffer &image, const Opencl_stuff &ocl_stuff){
    int number_pixels = image.rows * image.cols;
    size_t max_workgroup_dimensions[3];
    clGetDeviceInfo(ocl_stuff.device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_workgroup_dimensions), &max_workgroup_dimensions, NULL);

    size_t max_work_group_size;
    clGetDeviceInfo(ocl_stuff.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    int max_pixel_per_bloc = min(max_work_group_size, max_workgroup_dimensions[0]);

    this->number_blocs = max((int)ceil(((double)number_pixels/2.0) / (double)max_pixel_per_bloc) , 1); //why divided by 2.0 ?
    this->global_size = this->number_blocs * max_workgroup_dimensions[0];
    this->local_size = max_workgroup_dimensions[0] ;
    cl_uint max_number_blocs;
    clGetDeviceInfo(ocl_stuff.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_number_blocs), &max_number_blocs, NULL);

    /**
    cout << "Number of pixels: " << number_pixels <<endl;
    cout << "Number of blocs = " << this -> number_blocs <<endl;
    cout << "global size " << this->global_size<<endl;
    cout << "local size " <<  this->local_size <<endl;
    **/
}


// static int cpt = 0;

void apply_scan(Opencl_buffer &array_to_process,const Opencl_stuff &ocl_stuff, cl_kernel kernel_bloc, cl_kernel kernel_integration){
    assert(array_to_process.type == CV_32FC1);
    // - Set bloc size
    ScanParameters scan_parameter(array_to_process, ocl_stuff);
    // - Compute integral image of each bloc
    Opencl_buffer blocs_sums(1, scan_parameter.number_blocs, ocl_stuff,CV_32FC1);
    clSetKernelArg(kernel_bloc, 0, sizeof(array_to_process.buffer), (void*)&array_to_process.buffer);
    clSetKernelArg(kernel_bloc, 1, scan_parameter.local_size * 2 * sizeof(float), (void*)nullptr);
    clSetKernelArg(kernel_bloc, 2, sizeof(blocs_sums.buffer), (void*) &blocs_sums.buffer);
    // -- The number of thread is half the number of pixel
    size_t global_work_size_image[] = {(size_t) scan_parameter.global_size};
    size_t local_work_size_image[] = {(size_t) scan_parameter.local_size};  //if it's a half then why not divided?
    cout << scan_parameter.global_size << endl;
    cout << scan_parameter.local_size << endl;
    clEnqueueNDRangeKernel(ocl_stuff.queue,
                           kernel_bloc,
                           1,
                           NULL,
                           global_work_size_image,
                           local_work_size_image,
                           0,
                           NULL, NULL);

    clFinish(ocl_stuff.queue);
    // - Recursive call
    if(scan_parameter.number_blocs >1){
        cout << "-- Launching Intermediate step" << endl;
        // The last integral can be computed in one work group
        apply_scan(blocs_sums, ocl_stuff, kernel_bloc, kernel_integration);
        cout << "-- End Intermediate step" << endl;
    }

    if(scan_parameter.number_blocs >1){
        // - Not the deepest case when the integral image can be computed in one work group
        // - Join each bloc's integral image.
        clSetKernelArg(kernel_integration, 0, sizeof(array_to_process.buffer), (void*)&array_to_process.buffer);
        clSetKernelArg(kernel_integration, 1, sizeof(blocs_sums.buffer), (void*) &blocs_sums.buffer);
        // -- The number of work item must be a multiple of the local workspace or the kernel do not launch.
        // --- Not sure yet what the influence of dangling work item has in the result
        size_t global_work_size_image2[] = {(size_t) ceil((int)(array_to_process.cols * array_to_process.rows) / scan_parameter.local_size) * scan_parameter.local_size };
        clEnqueueNDRangeKernel(ocl_stuff.queue,
                               kernel_integration,
                               1,
                               NULL,
                               global_work_size_image2,
                               local_work_size_image,
                               0,
                               NULL, NULL);
        clFinish(ocl_stuff.queue);
    }
    /**
    // use a static int to differentiate between recursive call, hack for debugging
    cout << "cpt " << cpt << " " << array_to_process.rows << " "<< array_to_process.cols << " " <<array_to_process.buffer_size <<endl;
    array_to_process.write_img(to_string(cpt) + "Yipikai.png", ocl_stuff, true);
    cpt ++;
     **/
}

Opencl_buffer transpose(string image_path, int max_distance, Opencl_stuff ocl_stuff) {
    cl_int error;
    cl_program  transpose_program;
    compile_source(&transpose_kernel_path, &transpose_program, ocl_stuff.device, ocl_stuff.context);
    cl_kernel transpose_kernel = clCreateKernel(transpose_program, "transpose", &error);
    assert(error == CL_SUCCESS);
    int block_size = 16;
    Opencl_buffer image_buffer(image_path, ocl_stuff, max_distance);
    Opencl_buffer output_buffer(image_buffer.rows, image_buffer.cols, ocl_stuff);
    Opencl_buffer block_buffer(block_size, block_size, ocl_stuff);

    int width = image_buffer.cols; // because the image is padded
    int height = image_buffer.rows;
    clSetKernelArg(transpose_kernel, 0, sizeof(image_buffer.buffer), (void*)&image_buffer.buffer);
    clSetKernelArg(transpose_kernel, 1, sizeof(output_buffer.buffer), (void*)&output_buffer.buffer);
    clSetKernelArg(transpose_kernel, 2, sizeof(width), &width);
    clSetKernelArg(transpose_kernel, 3, sizeof(height), &height);
    clSetKernelArg(transpose_kernel, 4, sizeof(block_buffer.buffer), (void*)nullptr);

    size_t global_work_size_image[] = { (size_t)width, (size_t)height };
    size_t local_work_size_image[] = { (size_t)block_size, (size_t)block_size };


    clEnqueueNDRangeKernel(ocl_stuff.queue,
        transpose_kernel,
        2,
        NULL,
        global_work_size_image,
        local_work_size_image,
        0,
        NULL, NULL);

    clFinish(ocl_stuff.queue); // syncing

    return output_buffer;

}



void compute_integral_image(Opencl_buffer &image, const Opencl_stuff &ocl_stuff) {
    assert(image.type == CV_32FC1);
    // - Compile source
    cl_int error1, error2, error3;
    cl_program  scan_program;
    compile_source(&scan_kernel_path, &scan_program, ocl_stuff.device, ocl_stuff.context);
    cl_kernel scan_kernel = clCreateKernel(scan_program, "scan", &error1);
    cl_program  scan_integration_program;
    compile_source(&scan_integration_kernel_path, &scan_integration_program, ocl_stuff.device, ocl_stuff.context);
    cl_kernel scan_integration_kernel = clCreateKernel(scan_integration_program, "scan_integration", &error3);
    cl_program transpose_program;
    compile_source(&transpose_kernel_path, &transpose_program, ocl_stuff.device, ocl_stuff.context);
    cl_kernel transpose_kernel = clCreateKernel(transpose_program, "transpose", &error2);
   // assert( (error1 == CL_SUCCESS) and (error2 == CL_SUCCESS) and (error3 == CL_SUCCESS));
    assert(error1 == CL_SUCCESS);
    assert(error2 == CL_SUCCESS);
    assert(error3 == CL_SUCCESS);

    // - Apply Scan horizontally
    apply_scan(image, ocl_stuff, scan_kernel, scan_integration_kernel);
    image.write_img("HorizontalIntegralImage.png", ocl_stuff, true);

    // - Transpose the intermediate result

    // -- Apply Scan on the row of the transposed intermediate result


    // - Compute the window average

    // - Compute the window variance
}
