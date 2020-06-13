#include "integral_image.h"
#include "ocl_wrapper.h"
#include "tools.h"
#include <iostream>
#include <thread>
#include <opencv2/core.hpp>
#include <CL/cl.h>
#include <math.h>

using namespace std;
const string scan_kernel_path = "scan.cl";
const string scan_integration_kernel_path = "scan_integration.cl";
const string transpose_kernel_path = "transpose.cl";


ScanParameters::ScanParameters(const Opencl_buffer &image, const Opencl_stuff &ocl_stuff){
    int number_rows = image.rows;
    this->pixels_per_row = image.cols;
    size_t max_workgroup_dimensions[3];
    clGetDeviceInfo(ocl_stuff.device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_workgroup_dimensions), &max_workgroup_dimensions, NULL);
    size_t max_work_group_size;
    clGetDeviceInfo(ocl_stuff.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    int max_pixel_per_bloc = min(max_work_group_size, max_workgroup_dimensions[0]);
    // Each bloc affect twice it size of pixels
    this->number_of_bloc_per_row = max((int)ceil(((double)pixels_per_row/2.0) / (double)max_pixel_per_bloc), 1);
    this->number_blocs = number_of_bloc_per_row * number_rows;
    this->global_size = this->number_blocs * max_workgroup_dimensions[0];
    this->local_size = max_workgroup_dimensions[0] ;
    cl_uint max_number_blocs;
    clGetDeviceInfo(ocl_stuff.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_number_blocs), &max_number_blocs, NULL);

//
//    cout << "Number of pixels: " << number_pixels <<endl;
//    cout << "Number of blocs = " << this -> number_blocs <<endl;
//    cout << "global size " << this->global_size<<endl;
//    cout << "local size " <<  this->local_size <<endl;

}


void apply_scan_on_row(Opencl_buffer &array_to_process, const Opencl_stuff &ocl_stuff, cl_kernel kernel_bloc,
                cl_kernel kernel_integration) {
    assert(array_to_process.type == CV_32FC1);
    // - Set bloc size
    ScanParameters scan_parameter(array_to_process, ocl_stuff);
    // - Compute integral image of each bloc
    int actual_size = scan_parameter.pixels_per_row;
    int number_bloc_per_row = scan_parameter.number_of_bloc_per_row;
   // cout << actual_size << "  " << number_bloc_per_row << endl;
//    cout << scan_parameter.global_size <<endl;
    Opencl_buffer blocs_sums(array_to_process.rows, scan_parameter.number_of_bloc_per_row , ocl_stuff,CV_32FC1);
    clSetKernelArg(kernel_bloc, 0, sizeof(array_to_process.buffer), (void*)&array_to_process.buffer);
    clSetKernelArg(kernel_bloc, 1, scan_parameter.local_size * 2 * sizeof(float), (void*)nullptr);
    clSetKernelArg(kernel_bloc, 2, sizeof(blocs_sums.buffer), (void*) &blocs_sums.buffer);
    clSetKernelArg(kernel_bloc, 3, sizeof(actual_size), &actual_size);
    clSetKernelArg(kernel_bloc, 4, sizeof(number_bloc_per_row), &number_bloc_per_row);
    // -- The number of thread is half the number of pixel
    size_t global_work_size_image[] = {(size_t) scan_parameter.global_size};
    size_t local_work_size_image[] = {(size_t) scan_parameter.local_size};  //if it's a half then why not divided?
    clEnqueueNDRangeKernel(ocl_stuff.queue,
                           kernel_bloc,
                           1,
                           nullptr,
                           global_work_size_image,
                           local_work_size_image,
                           0,
                           NULL, NULL);

    clFinish(ocl_stuff.queue);
    // - Recursive call
    if(scan_parameter.number_of_bloc_per_row > 1) {
        //cout << " -- Launching Intermediate step" << endl;
        // The last integral can be computed in one work group
        apply_scan_on_row(blocs_sums, ocl_stuff, kernel_bloc, kernel_integration);
        //cout << " -- End Intermediate step" << endl;

        // - Not the deepest case when the integral image can be computed in one work group
        // - Join each bloc's integral image.
        clSetKernelArg(kernel_integration, 0, sizeof(array_to_process.buffer), (void*)&array_to_process.buffer);
        clSetKernelArg(kernel_integration, 1, sizeof(blocs_sums.buffer), (void*) &blocs_sums.buffer);
        clSetKernelArg(kernel_integration, 2, sizeof(actual_size), &actual_size);
        // --- Multiplied by two because there is 2 times more threads.
        number_bloc_per_row *= 2;
        clSetKernelArg(kernel_integration, 3, sizeof(number_bloc_per_row), &number_bloc_per_row);
        // -- The number of work item must be a multiple of the local workspace or the kernel do not launch.
        // --- There is 2 times more work items than needed for scan
        size_t global_work_size_image2[] = {(size_t) scan_parameter.global_size *2 };
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

void apply_scan(Opencl_buffer &array_to_process, const Opencl_stuff &ocl_stuff, cl_kernel kernel_bloc,
                cl_kernel kernel_integration){

    int row_number = array_to_process.rows;
    //auto thread_array = new thread[row_number];
    apply_scan_on_row(array_to_process, ocl_stuff, kernel_bloc, kernel_integration);
//    for(int i = 0; i < row_number; i++){
////        thread th (apply_scan_on_row,ref(array_to_process), ref(ocl_stuff), kernel_bloc, kernel_integration, i);
////        th.join();
//        thread_array[i] = thread(apply_scan_on_row,ref(array_to_process), ref(ocl_stuff), kernel_bloc, kernel_integration, i);
//        thread_array[i].join();
//    }
    // Curieusement le multithreading ne fonctionne pas qui aurait cru?
//    for(int i = 0; i < row_number; i++) {
//        thread_array[i].join();
//    }

}

Opencl_buffer transpose(Opencl_buffer image_buffer, Opencl_stuff ocl_stuff) {
    cl_int error;
    cl_program  transpose_program;
    compile_source(&transpose_kernel_path, &transpose_program, ocl_stuff.device, ocl_stuff.context);
    cl_kernel transpose_kernel = clCreateKernel(transpose_program, "transpose", &error);
    assert(error == CL_SUCCESS);
    int block_size = 8;

    Opencl_buffer output_buffer(image_buffer.cols, image_buffer.rows, ocl_stuff,CV_32FC1);
    Opencl_buffer block_buffer(block_size, block_size, ocl_stuff);

    int width = image_buffer.cols; // because the image is padded
    int height = image_buffer.rows;

    //cout <<" - Transpose "<< endl;
    clSetKernelArg(transpose_kernel, 0, sizeof(image_buffer.buffer), (void*)&image_buffer.buffer);
    clSetKernelArg(transpose_kernel, 1, sizeof(output_buffer.buffer), (void*)&output_buffer.buffer);
    clSetKernelArg(transpose_kernel, 2, sizeof(width), &width);
    clSetKernelArg(transpose_kernel, 3, sizeof(height), &height);
    clSetKernelArg(transpose_kernel, 4, sizeof(float)*block_size * block_size, (void*)nullptr);

    size_t global_work_size_image[] = { (size_t)width, (size_t)height };
    size_t local_work_size_image[] = { (size_t)block_size, (size_t)block_size };


    clEnqueueNDRangeKernel(ocl_stuff.queue,
        transpose_kernel,
        2,
        nullptr,
        global_work_size_image,
        local_work_size_image,
        0,
        NULL, NULL);

    clFinish(ocl_stuff.queue); // syncing
    image_buffer.free();
    return output_buffer;

}

Opencl_buffer transpose(string image_path, int max_distance, Opencl_stuff ocl_stuff){
    Opencl_buffer image_buffer(image_path, ocl_stuff, 0, CV_32FC1);
    image_buffer.write_img("transpose_ref.jpg", false);
    return transpose(image_buffer, ocl_stuff);
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
    assert(error1 == CL_SUCCESS);
    assert(error2 == CL_SUCCESS);
    assert(error3 == CL_SUCCESS);

    // - Apply Scan horizontally
   // cout <<" - Scan "<< endl;
    apply_scan(image, ocl_stuff, scan_kernel, scan_integration_kernel);
    image.write_img("HorizontalIntegralImage.png", true);
    // - Transpose the intermediate result
    image = transpose(image,ocl_stuff);

    // -- Apply Scan on the row of the transposed intermediate result
    //cout <<" - Scan "<< endl;
    apply_scan(image, ocl_stuff, scan_kernel, scan_integration_kernel);
    image.write_img("VerticalIntegralImage.png", true);

    // - Transpose to obtain the integral image
    image = transpose(image,ocl_stuff);
   // image.write_img("IntegralImage.png", true);

    // - Compute the window average

    // - Compute the window variance
}
