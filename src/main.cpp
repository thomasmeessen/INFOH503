
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

    // 2.1 Check if the device tolerate images
    //cl_bool param_value_image_accepted;
    //clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), (void *)param_value_image_accepted, NULL);
    //cout <<( (param_value_image_accepted == CL_TRUE)? "Device accept image ": "Device do not accept image") << endl;

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
    int image_1D_size = left_image.cols * left_image.rows * sizeof(char)*3;
    int original_width = left_image.cols;
    int original_height = left_image.rows;
    int max_dist = MAX_DISTANCE;

    cl_program difference_image_program;
    compile_source(&difference_image_source_path, &difference_image_program, device, context);

    cl_kernel difference_image_kernel = clCreateKernel( difference_image_program, "memset", NULL );

    // 5. Load an image into a buffer
    // unsigned char* output_image = (unsigned char*) malloc(output_size);  
    cv::Mat output_image(left_image.rows, left_image.cols*MAX_DISTANCE, CV_8U) ;   // each image will be next to each other?
    int output_size = output_image.total() * output_image.elemSize(); //https://answers.opencv.org/question/21296/cvmat-data-size/

    cl_mem left_image_buffer = clCreateBuffer( context,
                                    CL_MEM_COPY_HOST_PTR,
                                    image_1D_size,
                                    (void*)left_image.data, NULL );
    clSetKernelArg(difference_image_kernel, 0, sizeof(left_image_buffer), (void*) &left_image_buffer);//https://stackoverflow.com/a/22101104

    cl_mem right_image_buffer = clCreateBuffer( context,
                                    CL_MEM_COPY_HOST_PTR,
                                    image_1D_size,
                                    (void*)right_image.data, NULL );

    clSetKernelArg(difference_image_kernel, 1, sizeof(right_image_buffer), (void*) &right_image_buffer);//https://stackoverflow.com/a/22101104
        
    cl_mem destination_buffer = clCreateBuffer( context,
                                    CL_MEM_COPY_HOST_PTR,
                                    output_size,
                                    (void*)output_image.data, NULL );

    clSetKernelArg(difference_image_kernel, 2, sizeof(destination_buffer), (void*) &destination_buffer);
    clSetKernelArg(difference_image_kernel, 3, sizeof(original_height), &original_height);//set width value
    
    clSetKernelArg(difference_image_kernel, 4, sizeof(original_width), &original_width);//set width value
    clSetKernelArg(difference_image_kernel, 5, sizeof(max_dist), &max_dist);//set maxDistance value

    // 6. Launch the kernel. Let OpenCL pick the local work size.

    size_t global_work_size_image[] = {(size_t) left_image.cols, (size_t) left_image.rows};
    clEnqueueNDRangeKernel( queue,
                            difference_image_kernel,
                            2,
                            NULL,
                            global_work_size_image,
                            NULL,
                            0,
                            NULL, NULL);

    clFinish( queue ); // syncing

    // 7. Look at the results via synchronous buffer map.
    clEnqueueReadBuffer(queue,
                      destination_buffer,
                      CL_TRUE,
                      NULL,
                      output_size,
                     (void*)output_image.data, NULL, NULL, NULL );
    cv::imwrite("test.png", output_image);
    return 0;
}