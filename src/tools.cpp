#include <string>
#include <iostream>

using namespace std;

void show_build_error(const string *source_path, cl_program *program, cl_device_id device){
    char *buff_error;
    cl_int errcode;
    size_t build_log_len;
    errcode = clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
    if (errcode) {
        printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
        exit(-1);
    }
        buff_error = (char*) malloc(build_log_len);
    if (!buff_error) {
        printf("malloc failed at line %d\n", __LINE__);
        exit(-2);
    }

    errcode = clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, build_log_len, buff_error, NULL);
    if (errcode) {
        printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
        exit(-3);
    }

    fprintf(stderr,"Build log: \n%s\n", buff_error); //Be careful with  the fprint
    free(buff_error);
    fprintf(stderr,"clBuildProgram failed\n");
    exit(EXIT_FAILURE);
}

void compile_source(const string *source_path, cl_program *program, cl_device_id device, cl_context context){
    // 4. Perform runtime source compilation, and obtain kernel entry point.
    std::ifstream source_file(*source_path);
    std::string source_code(std::istreambuf_iterator<char>(source_file), (std::istreambuf_iterator<char>()));
    const char* c_string_code = &source_code[0];
    *program = clCreateProgramWithSource( context,
                                                    1,
                                                    (const char **) &c_string_code,
                                                    NULL, NULL );

    cl_int result = clBuildProgram( *program, 1, &device, NULL, NULL, NULL );
    if(result!= CL_SUCCESS){ //https://stackoverflow.com/a/29813956
        //  if error while building the kernel we print it
        cout << "Error while building \""<<*source_path << "\""  << endl;
    show_build_error(source_path, program, device);
    }
}

cv::Mat to_greyscale_plus_padding(const string image_path, cv::Mat &source_image, int max_distance, cl_context context, cl_kernel kernel, cl_command_queue queue, bool write_to_png){
    //put a picture to greyscale and add padding around it.
    source_image = cv::imread(image_path, cv::IMREAD_COLOR);
    source_image.convertTo(source_image, CV_8U);  // for greyscale
    // + 2* max_distance because will had max_distance column/row on the left/top and max_distance column/row to the right/bottom.
    cv::Mat output_image = cv::Mat(source_image.rows+ 2*max_distance, source_image.cols + 2*max_distance, CV_8UC1) ;
    
    int input_image_1D_size = source_image.total() * source_image.elemSize();
    cl_mem input_image_buffer = clCreateBuffer( context,
                                    CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                    input_image_1D_size,
                                    (void*)source_image.data, NULL );

    clSetKernelArg(kernel, 0, sizeof(input_image_buffer), (void*) &input_image_buffer);

    int output_size = output_image.total() * output_image.elemSize(); //https://answers.opencv.org/question/21296/cvmat-data-size/
    cl_mem output_image_buffer = clCreateBuffer( context,
                                    CL_MEM_COPY_HOST_PTR | CL_MEM_WRITE_ONLY,
                                    output_size,
                                    (void*)output_image.data, NULL );

    clSetKernelArg(kernel, 1, sizeof(output_image_buffer), (void*) &output_image_buffer);
    int width = source_image.cols;
    int height = source_image.rows;
    clSetKernelArg(kernel, 2, sizeof(width), &width);
    clSetKernelArg(kernel, 3, sizeof(height),&height);
    clSetKernelArg(kernel, 4, sizeof(max_distance), &max_distance);

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
                      output_image_buffer,
                      CL_TRUE,
                      NULL,
                      output_size,
                      (void*)output_image.data, NULL, NULL, NULL );

    //source_image.release();
    //source_image = output_image;

    
    if(write_to_png){
        string output = "grey_" + image_path;
        cv::imwrite(output, output_image);
    }
    return output_image;

}


void guidedFilter(const string image_path, cv::Mat& image, cl_context context, cl_kernel kernel, cl_command_queue queue, bool write_to_png) {


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
        (void*)output.data, NULL);

    cl_mem cost_buffer = clCreateBuffer(context,
        CL_MEM_COPY_HOST_PTR,
        image_1D_size,
        (void*)cost.data, NULL);

    // 6. Launch the kernel. Let OpenCL pick the local work size.
    clSetKernelArg(kernel, 0, sizeof(buffer), (void*)&buffer);
    clSetKernelArg(kernel, 1, sizeof(output_buffer), (void*)&output_buffer);
    clSetKernelArg(kernel, 2, sizeof(cost_buffer), (void*)&cost_buffer);

    size_t global_work_size_image[] = { (size_t)image.cols, (size_t)image.rows };
    clEnqueueNDRangeKernel(queue,
        kernel,
        2,
        NULL,
        global_work_size_image,
        NULL,
        0,
        NULL, NULL);

    clFinish(queue); // syncing

    // 7. Look at the results via synchronous buffer map.

    clEnqueueReadBuffer(queue,
        output_buffer,
        CL_TRUE,
        NULL,
        image_1D_size,
        (void*)output.data, NULL, NULL, NULL);

    if(write_to_png){
        string output_name = "guided_" + image_path;
        cv::imwrite(output_name, output);
    }


}


void image_difference(cv::Mat &left_image, cv::Mat &right_image, cv::Mat &output_image,int max_distance, cl_context context, cl_kernel kernel, cl_command_queue queue, bool write_to_png){
    int image_1D_size = left_image.cols * left_image.rows * sizeof(char)*3;
    int original_width = left_image.cols - 2*max_distance; // cause left_image is the original image + the pading
    int original_height = left_image.rows - 2*max_distance;

    output_image = cv::Mat((original_height + 2*max_distance), (original_width+2*max_distance)*max_distance, CV_8U) ;   // each image will be next to each other and we add padding to it
    
    int output_size = output_image.total() * output_image.elemSize(); //https://answers.opencv.org/question/21296/cvmat-data-size/

    cl_mem left_image_buffer = clCreateBuffer( context,
                                    CL_MEM_COPY_HOST_PTR,
                                    image_1D_size,
                                    (void*)left_image.data, NULL );

    clSetKernelArg(kernel, 0, sizeof(left_image_buffer), (void*) &left_image_buffer);//https://stackoverflow.com/a/22101104

    cl_mem right_image_buffer = clCreateBuffer( context,
                                    CL_MEM_COPY_HOST_PTR,
                                    image_1D_size,
                                    (void*)left_image.data, NULL );

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

    size_t global_work_size_image[] = {(size_t) original_width, (size_t) original_height};
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