#include <string>

using namespace std;

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

cv::Mat to_greyscale(const string image_path, cl_context context, cl_device_id device, const string greyscale_source_path, cl_command_queue queue, bool write_to_png){

    cl_program greyscale_program;
    compile_source(&greyscale_source_path, &greyscale_program, device, context);

    cl_kernel greyscale_kernel = clCreateKernel(greyscale_program, "memset", NULL);

    cv::Mat source_image = cv::imread(image_path, cv::IMREAD_COLOR);
    source_image.convertTo(source_image, CV_8U);  // for greyscale

    cv::Mat output_image = cv::Mat(source_image.rows, source_image.cols, CV_8U);

    int image_1D_size = source_image.cols * source_image.rows * sizeof(char)*3;
    cl_mem buffer = clCreateBuffer( context,
                                    CL_MEM_COPY_HOST_PTR,
                                    image_1D_size,
                                    (void*)source_image.data, NULL );

    cl_mem destination_buffer = clCreateBuffer(context,
        CL_MEM_COPY_HOST_PTR,
        image_1D_size/3,
        (void*)output_image.data, NULL);

    // 6. Launch the kernel. Let OpenCL pick the local work size.
    clSetKernelArg(greyscale_kernel, 0, sizeof(buffer), (void*) &buffer);
    clSetKernelArg(greyscale_kernel, 1, sizeof(destination_buffer), (void*) &destination_buffer);

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

    /*clEnqueueReadBuffer(queue,
                      buffer,
                      CL_TRUE,
                      NULL,
                      image_1D_size,
                     (void*)source_image.data, NULL, NULL, NULL );*/

    clEnqueueReadBuffer(queue,
        destination_buffer,
        CL_TRUE,
        NULL,
        image_1D_size/3,
        (void*)output_image.data, NULL, NULL, NULL);

    
    if(write_to_png){
        string output = "grey_" + image_path;
        std::cout << output;
        cv::imwrite(output, output_image);
    }

    return output_image;

}


void guidedFilter(cv::Mat& image, cl_context context, cl_device_id device, const string guidedFilter_source_path, cl_command_queue queue) {

    cl_program guidedFilter_program;
    compile_source(&guidedFilter_source_path, &guidedFilter_program, device, context);

    cl_kernel guidedFilter_kernel = clCreateKernel(guidedFilter_program, "memset", NULL);

    int image_1D_size = image.cols * image.rows * sizeof(char);
    cl_mem buffer = clCreateBuffer(context,
        CL_MEM_COPY_HOST_PTR,
        image_1D_size,
        (void*)image.data, NULL);

    cv::Mat output = cv::Mat(image.rows, image.cols, CV_8U);

    cv::Mat cost = cv::Mat(image.rows, image.cols, CV_8U);


    cl_mem output_buffer = clCreateBuffer(context,
        CL_MEM_COPY_HOST_PTR,
        image_1D_size,
        (void*)image.data, NULL);

    cl_mem cost_buffer = clCreateBuffer(context,
        CL_MEM_COPY_HOST_PTR,
        image_1D_size,
        (void*)image.data, NULL);

    // 6. Launch the kernel. Let OpenCL pick the local work size.
    clSetKernelArg(guidedFilter_kernel, 0, sizeof(buffer), (void*)&buffer);
    clSetKernelArg(guidedFilter_kernel, 1, sizeof(output_buffer), (void*)&output_buffer);
    clSetKernelArg(guidedFilter_kernel, 2, sizeof(cost_buffer), (void*)&cost_buffer);

    size_t global_work_size_image[] = { (size_t)image.cols, (size_t)image.rows };
    clEnqueueNDRangeKernel(queue,
        guidedFilter_kernel,
        2,
        NULL,
        global_work_size_image,
        NULL,
        0,
        NULL, NULL);

    clFinish(queue); // syncing

    // 7. Look at the results via synchronous buffer map.

    /*clEnqueueReadBuffer(queue,
                      buffer,
                      CL_TRUE,
                      NULL,
                      image_1D_size,
                     (void*)source_image.data, NULL, NULL, NULL );*/

    clEnqueueReadBuffer(queue,
        buffer,
        CL_TRUE,
        NULL,
        image_1D_size,
        (void*)image.data, NULL, NULL, NULL);

}










void image_difference(cv::Mat &left_image, cv::Mat &right_image, cv::Mat &output_image,int max_distance, cl_context context, cl_kernel kernel, cl_command_queue queue, bool write_to_png){
    int image_1D_size = left_image.cols * left_image.rows * sizeof(char)*3;
    int original_width = left_image.cols;
    int original_height = left_image.rows;

    output_image = cv::Mat(left_image.rows, left_image.cols*max_distance, CV_8U) ;   // each image will be next to each other?
    
    int output_size = output_image.total() * output_image.elemSize(); //https://answers.opencv.org/question/21296/cvmat-data-size/

    cl_mem left_image_buffer = clCreateBuffer( context,
                                    CL_MEM_COPY_HOST_PTR,
                                    image_1D_size,
                                    (void*)left_image.data, NULL );
    clSetKernelArg(kernel, 0, sizeof(left_image_buffer), (void*) &left_image_buffer);//https://stackoverflow.com/a/22101104

    cl_mem right_image_buffer = clCreateBuffer( context,
                                    CL_MEM_COPY_HOST_PTR,
                                    image_1D_size,
                                    (void*)right_image.data, NULL );

    clSetKernelArg(kernel, 1, sizeof(right_image_buffer), (void*) &right_image_buffer);//https://stackoverflow.com/a/22101104
        
    cl_mem destination_buffer = clCreateBuffer( context,
                                    CL_MEM_COPY_HOST_PTR,
                                    output_size,
                                    (void*)output_image.data, NULL );

    clSetKernelArg(kernel, 2, sizeof(destination_buffer), (void*) &destination_buffer);
    clSetKernelArg(kernel, 3, sizeof(original_height), &original_height);//set width value
    
    clSetKernelArg(kernel, 4, sizeof(original_width), &original_width);//set width value
    clSetKernelArg(kernel, 5, sizeof(max_distance), &max_distance);//set maxDistance value

    // 6. Launch the kernel. Let OpenCL pick the local work size.

    size_t global_work_size_image[] = {(size_t) left_image.cols, (size_t) left_image.rows};
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
                      destination_buffer,
                      CL_TRUE,
                      NULL,
                      output_size,
                     (void*)output_image.data, NULL, NULL, NULL );
    if(write_to_png){
        cv::imwrite("img_diff.png", output_image);
    }                 

}