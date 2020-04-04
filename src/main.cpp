
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

    // 4. Perform runtime source compilation, and obtain kernel entry point.
    std::ifstream source_file(source_path);
    std::string source_code(std::istreambuf_iterator<char>(source_file), (std::istreambuf_iterator<char>()));
    const char* c_string_code = &source_code[0];
    cl_program program = clCreateProgramWithSource( context,
                                                    1,
                                                    (const char **) &c_string_code,
                                                    NULL, NULL );

    clBuildProgram( program, 1, &device, NULL, NULL, NULL );

    cl_kernel kernel = clCreateKernel( program, "memset", NULL );

    // 5. Load an image into a buffer
    cv::Mat source_image = cv::imread("paper0.png", cv::IMREAD_COLOR);
    source_image.convertTo(source_image, CV_8U); 

    cv::imwrite("pre.png", source_image);
    int image_1D_size = source_image.cols * source_image.rows * sizeof(int);
    cl_mem buffer = clCreateBuffer( context,
                                    CL_MEM_COPY_HOST_PTR,
                                    image_1D_size,
                                    (void*)source_image.data, NULL );

    // 6. Launch the kernel. Let OpenCL pick the local work size.
    clSetKernelArg(kernel, 0, sizeof(buffer), (void*) &buffer);
    size_t nb_pixels = source_image.cols * source_image.rows;
    size_t global_work_size_image[] = {(size_t) source_image.cols, (size_t) source_image.rows};
    clEnqueueNDRangeKernel( queue,
                            kernel,
                            2,
                            NULL,
                            global_work_size_image,
                            NULL,
                            0,
                            NULL, NULL);

    clFinish( queue );

    // 7. Look at the results via synchronous buffer map.

    clEnqueueReadBuffer(queue,
                      buffer,
                      CL_TRUE,
                      NULL,
                      image_1D_size,
                     (void*)source_image.data, NULL, NULL, NULL );


    cv::imwrite("post.png", source_image);
//    int i;
//
//    for(i=0; i < nb_pixels; i++)
//        printf("%d %d\n", i, ptr[i]);

    return 0;
}