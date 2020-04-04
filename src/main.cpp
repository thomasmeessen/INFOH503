
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;



// A simple threshold kernel
const string source_path = "greyscale.cl";
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
    compile_source(&source_path, &greyscale_program, device, context);

    cl_kernel greyscale_kernel = clCreateKernel( greyscale_program, "memset", NULL );

    // 5. Load an image into a buffer
    cv::Mat source_image = cv::imread("paper0.png", cv::IMREAD_COLOR);
    source_image.convertTo(source_image, CV_8U);  // for greyscale

    cv::imwrite("pre.png", source_image);
    int image_1D_size = source_image.cols * source_image.rows * sizeof(char)*3;
    cl_mem buffer = clCreateBuffer( context,
                                    CL_MEM_COPY_HOST_PTR,
                                    image_1D_size,
                                    (void*)source_image.data, NULL );

    // 6. Launch the kernel. Let OpenCL pick the local work size.
    clSetKernelArg(greyscale_kernel, 0, sizeof(buffer), (void*) &buffer);

    size_t global_work_size_image[] = {(size_t) source_image.cols, (size_t) source_image.rows};
    clEnqueueNDRangeKernel( queue,
                            greyscale_kernel,
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


    cv::imwrite("post_greyscale.png", source_image);


    return 0;
}