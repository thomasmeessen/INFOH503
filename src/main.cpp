
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;


#define MAX_DISTANCE 16 // maximum differnenc in pixel

// A simple threshold kernel
const string greyscale_source_path = "greyscale.cl";
const string difference_image_source_path = "differenceImage.cl";
const string left_image_path  = "paper0.png";
const string right_image_path = "paper1.png";

void compile_source(const string *source_path, cl_program *program, cl_device_id device, cl_context context){
    // 4. Perform runtime source compilation, and obtain kernel entry point.
    std::ifstream source_file(*source_path);
    std::string source_code(std::istreambuf_iterator<char>(source_file), (std::istreambuf_iterator<char>()));
    const char* c_string_code = &source_code[0];
    *program = clCreateProgramWithSource( context,
                                                    1,
                                                    (const char **) &c_string_code,
                                                    NULL, NULL );

    clBuildProgram( *program, 1, &device, NULL, NULL, NULL );
}

void to_greyscale(const string *image, cv::Mat &source_image, cl_context context, cl_kernel kernel, cl_command_queue queue){
    source_image = cv::imread(*image, cv::IMREAD_COLOR);
    source_image.convertTo(source_image, CV_8U);  // for greyscale

    int image_1D_size = source_image.cols * source_image.rows * sizeof(char)*3;
    cl_mem buffer = clCreateBuffer( context,
                                    CL_MEM_COPY_HOST_PTR,
                                    image_1D_size,
                                    (void*)source_image.data, NULL );

    // 6. Launch the kernel. Let OpenCL pick the local work size.
    clSetKernelArg(kernel, 0, sizeof(buffer), (void*) &buffer);

    size_t global_work_size_image[] = {(size_t) source_image.cols, (size_t) source_image.rows};
    clEnqueueNDRangeKernel( queue,
                            kernel,
                            2,
                            NULL,
                            global_work_size_image,
                            NULL,
                            0,
                            NULL, NULL);

    clFinish( queue ); // syncing

    // 7. Look at the results via synchronous buffer map.

    clEnqueueReadBuffer(queue,
                      buffer,
                      CL_TRUE,
                      NULL,
                      image_1D_size,
                     (void*)source_image.data, NULL, NULL, NULL );


}

int main(int argc, char ** argv)
{
    // 1. Get a platform.
    cl_platform_id platform;
    clGetPlatformIDs( 1, &platform, NULL );

    // 2. Find a gpu device.
    cl_device_id device;
    clGetDeviceIDs( platform,
                    CL_DEVICE_TYPE_GPU,
                    1,
                    &device, NULL);

    // 3. Create a context and command queue on that device.
    cl_context context = clCreateContext( NULL,
                                          1,
                                          &device,
                                          NULL, NULL, NULL);

    cl_command_queue queue = clCreateCommandQueue( context,
                                                   device,
                                                   0, NULL );

 
    cl_program greyscale_program;
    compile_source(&greyscale_source_path, &greyscale_program, device, context);

    cl_kernel greyscale_kernel = clCreateKernel( greyscale_program, "memset", NULL );
   
   
    cv::Mat left_image;
    to_greyscale(&left_image_path , left_image , context, greyscale_kernel, queue);
    cv::Mat right_image;
    to_greyscale(&right_image_path, right_image, context, greyscale_kernel, queue);
    // cv::imwrite("gauche.png", left_image);
    // cv::imwrite("droit.png", right_image);

//--------------------------------------------
//--------Difference Image Kernel-------------
//--------------------------------------------
    //now source image = the greysclae image

    // cl_program difference_image_program;
    // compile_source(&difference_image_source_path, &difference_image_program, device, context);

    // cl_kernel difference_image_kernel = clCreateKernel( difference_image_program, "memset", NULL );

    // // 5. Load an image into a buffer
    // long int output_size = image_1D_size*MAX_DISTANCE; // we will make an image MAX_DISTANCE time bigger than the source image as it will be stored in one array
    // unsigned char* output_image = malloc(output_size); 
    
    // cl_mem buffer1 = clCreateBuffer( context,
    //                                 CL_MEM_COPY_HOST_PTR,
    //                                 image_1D_size,
    //                                 (void*)source_image.data, NULL );
    // clSetKernelArg(difference_image_kernel, 0, sizeof(buffer1), (void*) &buffer1);//https://stackoverflow.com/a/22101104

    // cl_mem buffer2 = clCreateBuffer( context,
    //                                 CL_MEM_COPY_HOST_PTR,
    //                                 output_size,
    //                                 (void*)output_image.data, NULL );

    // clSetKernelArg(difference_image_kernel, 0, sizeof(buffer1), (void*) &buffer2);//https://stackoverflow.com/a/22101104

    // // 6. Launch the kernel. Let OpenCL pick the local work size.

    // size_t global_work_size_image[] = {(size_t) source_image.cols, (size_t) source_image.rows};
    // clEnqueueNDRangeKernel( queue,
    //                         difference_image_kernel,
    //                         2,
    //                         NULL,
    //                         global_work_size_image,
    //                         NULL,
    //                         0,
    //                         NULL, NULL);

    // clFinish( queue ); // syncing

    // // 7. Look at the results via synchronous buffer map.

    // clEnqueueReadBuffer(queue,
    //                   buffer,
    //                   CL_TRUE,
    //                   NULL,
    //                   image_1D_size,
    //                  (void*)source_image.data, NULL, NULL, NULL );




    // free(output_image);
    return 0;
}