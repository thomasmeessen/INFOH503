#include <string>
#include <iostream>

using namespace std;

void show_build_error(const string* source_path, cl_program* program, cl_device_id device) {
    char* buff_error;
    cl_int errcode;
    size_t build_log_len;
    errcode = clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
    if (errcode) {
        printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
        exit(-1);
    }
    buff_error = (char*)malloc(build_log_len);
    if (!buff_error) {
        printf("malloc failed at line %d\n", __LINE__);
        exit(-2);
    }

    errcode = clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, build_log_len, buff_error, NULL);
    if (errcode) {
        printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
        exit(-3);
    }

    fprintf(stderr, "Build log: \n%s\n", buff_error); //Be careful with  the fprint
    free(buff_error);
    fprintf(stderr, "clBuildProgram failed\n");
    exit(EXIT_FAILURE);
}

void compile_source(const string* source_path, cl_program* program, cl_device_id device, cl_context context) {
    // 4. Perform runtime source compilation, and obtain kernel entry point.
    std::ifstream source_file(*source_path);
    std::string source_code(std::istreambuf_iterator<char>(source_file), (std::istreambuf_iterator<char>()));
    const char* c_string_code = &source_code[0];
    *program = clCreateProgramWithSource(context,
        1,
        (const char**)&c_string_code,
        NULL, NULL);

    cl_int result = clBuildProgram(*program, 1, &device, NULL, NULL, NULL);
    if (result != CL_SUCCESS) { //https://stackoverflow.com/a/29813956
        //  if error while building the kernel we print it
        cout << "Error while building \"" << *source_path << "\"" << endl;
        show_build_error(source_path, program, device);
    }
}

void to_greyscale_plus_padding(const string* image_path, cv::Mat& source_image, int max_distance, cl_context context, cl_kernel kernel, cl_command_queue queue, bool write_to_png) {
    //put a picture to greyscale and add padding around it.
    source_image = cv::imread(*image_path, cv::IMREAD_COLOR);
    source_image.convertTo(source_image, CV_8U);  // for greyscale
    // + 2* max_distance because will had max_distance column/row on the left/top and max_distance column/row to the right/bottom.
    cv::Mat output_image = cv::Mat(source_image.rows + 2 * max_distance, source_image.cols + 2 * max_distance, CV_8UC1);

    int input_image_1D_size = source_image.total() * source_image.elemSize();
    cl_mem input_image_buffer = clCreateBuffer(context,
        CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
        input_image_1D_size,
        (void*)source_image.data, NULL);

    clSetKernelArg(kernel, 0, sizeof(input_image_buffer), (void*)&input_image_buffer);

    int output_size = output_image.total() * output_image.elemSize(); //https://answers.opencv.org/question/21296/cvmat-data-size/
    cl_mem output_image_buffer = clCreateBuffer(context,
        CL_MEM_COPY_HOST_PTR | CL_MEM_WRITE_ONLY,
        output_size,
        (void*)output_image.data, NULL);

    clSetKernelArg(kernel, 1, sizeof(output_image_buffer), (void*)&output_image_buffer);
    int width = source_image.cols;
    int height = source_image.rows;
    clSetKernelArg(kernel, 2, sizeof(width), &width);
    clSetKernelArg(kernel, 3, sizeof(height), &height);
    clSetKernelArg(kernel, 4, sizeof(max_distance), &max_distance);

    size_t global_work_size_image[] = { (size_t)source_image.cols, (size_t)source_image.rows };
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
        output_image_buffer,
        CL_TRUE,
        NULL,
        output_size,
        (void*)output_image.data, NULL, NULL, NULL);

    source_image.release();
    source_image = output_image;

    if (write_to_png) {
        string output = "grey_" + *image_path;
        cv::imwrite(output, output_image);
    }

}


void guidedFilter(const string *image_path, cv::Mat& image, int max_distance, cl_context context, cl_kernel kernel, cl_command_queue queue, bool write_to_png) {

    int width = image.cols - 2*max_distance;
    int height = image.rows - 2*max_distance;

    int image_1D_size = image.total() * image.elemSize();
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
    clSetKernelArg(kernel, 3, sizeof(width), &width);
    clSetKernelArg(kernel, 4, sizeof(height),&height);
    clSetKernelArg(kernel, 5, sizeof(max_distance), &max_distance);
    size_t global_work_size_image[] = { (size_t)image.cols - 2*max_distance, (size_t)image.rows- 2*max_distance }; // don't work on pixels in the padding hence the "- 2*max_distance"
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

    if (write_to_png) {
        string output_name = "guided_" + *image_path;
        cv::imwrite(output_name, output);
    }

}










void image_difference(cv::Mat& left_image, cv::Mat& right_image, cv::Mat& output_image, int max_distance, cl_context context, cl_kernel kernel, cl_command_queue queue, bool write_to_png) {
    int image_1D_size = left_image.total() * left_image.elemSize();;
    int original_width = left_image.cols - 2 * max_distance; // cause left_image is the original image + the pading
    int original_height = left_image.rows - 2 * max_distance;

    output_image = cv::Mat((original_height + 2 * max_distance), (original_width + 2 * max_distance) * max_distance, CV_8U);   // each image will be next to each other and we add padding to it

    int output_size = output_image.total() * output_image.elemSize(); //https://answers.opencv.org/question/21296/cvmat-data-size/

    cl_mem left_image_buffer = clCreateBuffer(context,
        CL_MEM_COPY_HOST_PTR,
        image_1D_size,
        (void*)left_image.data, NULL);
    clSetKernelArg(kernel, 0, sizeof(left_image_buffer), (void*)&left_image_buffer);//https://stackoverflow.com/a/22101104

    cl_mem right_image_buffer = clCreateBuffer(context,
        CL_MEM_COPY_HOST_PTR,
        image_1D_size,
        (void*)right_image.data, NULL);

    clSetKernelArg(kernel, 1, sizeof(right_image_buffer), (void*)&right_image_buffer);//https://stackoverflow.com/a/22101104

    cl_mem destination_buffer = clCreateBuffer(context,
        CL_MEM_COPY_HOST_PTR,
        output_size,
        (void*)output_image.data, NULL);

    clSetKernelArg(kernel, 2, sizeof(destination_buffer), (void*)&destination_buffer);
    clSetKernelArg(kernel, 3, sizeof(original_height), &original_height);//set width value

    clSetKernelArg(kernel, 4, sizeof(original_width), &original_width);//set width value
    clSetKernelArg(kernel, 5, sizeof(max_distance), &max_distance);//set maxDistance value

    // 6. Launch the kernel. Let OpenCL pick the local work size.

    size_t global_work_size_image[] = { (size_t)original_width, (size_t)original_height };
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
        destination_buffer,
        CL_TRUE,
        NULL,
        output_size,
        (void*)output_image.data, NULL, NULL, NULL);
    if (write_to_png) {
        cv::imwrite("img_diff.png", output_image);
    }

}

struct opencl_stuff {
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
};

void image_padding (cv::Mat image, cv::Mat &dest, int padding_size)
{
    cv::Rect extract_zone (padding_size, padding_size, image.cols, image.rows);
    cv::Mat image_padded = cv::Mat::zeros(image.rows + 2*padding_size, image.cols + 2 * padding_size, image.type());
    image.copyTo(image_padded(extract_zone));
    dest = image_padded;
}

void cost_by_layer_isolated(string path_image_left, string path_image_right, int disparity, cl_device_id device , cl_context context, cl_command_queue queue) {

    const string cost_by_layer_source_path = "cost_volume_by_layer.cl";
    // - Kernel Compilation
    cl_program cost_by_layer_program;
    compile_source(&cost_by_layer_source_path, &cost_by_layer_program, device, context);
    cl_kernel cost_by_layer_kernel = clCreateKernel (cost_by_layer_program, "memset", NULL);

    // - Image Loading
    cv::Mat left_source_image = cv::imread(path_image_left, cv::IMREAD_GRAYSCALE);
    cv::Mat right_source_image = cv::imread(path_image_right, cv::IMREAD_GRAYSCALE);

    // - Padding
    int padding_size = disparity;

    float alpha_weight = 0.5;

    cv::Mat left_image_padded;
    image_padding(left_source_image,left_image_padded, padding_size);
    cv::Mat right_image_padded;
    image_padding(right_source_image, right_image_padded, padding_size);
    cv::Mat output_layer_cost = cv::Mat::zeros(right_source_image.size(), CV_32FC1); // float

    // - Merging into a single matrix with entrelacement
    cv::Mat source_images_padded;
    cv::Mat temp_array[2]  = {left_image_padded, right_image_padded};
    cv::merge(temp_array, 2, source_images_padded);

    // - Allocating the buffers
    cl_mem cost_input_buffer = clCreateBuffer(context,
                                              CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                              source_images_padded.total() * source_images_padded.elemSize(),
                                              (void*)source_images_padded.data, NULL);
    cl_mem cost_output_buffer = clCreateBuffer(context,
                                               CL_MEM_WRITE_ONLY,
                                               output_layer_cost.total() * output_layer_cost.elemSize(),
                                               NULL, NULL);
    // - Passing arguments to the kernel
    clSetKernelArg(cost_by_layer_kernel, 0, sizeof(cost_input_buffer), (void*)&cost_input_buffer);
    clSetKernelArg(cost_by_layer_kernel, 1, sizeof(cost_output_buffer), (void*)&cost_output_buffer);
    clSetKernelArg(cost_by_layer_kernel, 2, sizeof(padding_size), (void*)&padding_size);
    clSetKernelArg(cost_by_layer_kernel, 3, sizeof(disparity), (void*)&disparity);
    clSetKernelArg(cost_by_layer_kernel, 4, sizeof(alpha_weight), (void*)&alpha_weight);
    // - Enqueuing kernel
    size_t global_work_size_cost_layer[] = {(size_t)right_source_image.cols , (size_t) right_source_image.rows };
    clEnqueueNDRangeKernel(queue,
                           cost_by_layer_kernel,
                           2,
                           NULL,
                           global_work_size_cost_layer,
                           NULL,
                           0,
                           NULL, NULL);
    // - Waiting end execution

    clFinish(queue);

    // - Reading Results
    clEnqueueReadBuffer(queue,
                        cost_output_buffer,
                        CL_TRUE,
                        NULL,
                        output_layer_cost.total() * output_layer_cost.elemSize(),
                        (void*)output_layer_cost.data, NULL, NULL, NULL);
    string output_name = "cost_" + path_image_left;
    cv::imwrite(output_name, output_layer_cost);
}

void cost_by_layer_isolated(string path_image_left, string path_image_right, int disparity, opencl_stuff ocl_stuff){
    cost_by_layer_isolated(path_image_left, path_image_right, disparity, ocl_stuff.device, ocl_stuff.context, ocl_stuff.queue);
}

void cost_by_layer_integrated (const string& path_image_left, const string &path_image_right, int disparity, opencl_stuff ocl_stuff){
    const string cost_by_layer_source_path = "cost_layer_integrated_grayscale.cl";
    // - Kernel Compilation
    cl_program cost_by_layer_program;
    compile_source(&cost_by_layer_source_path, &cost_by_layer_program, ocl_stuff.device, ocl_stuff.context);
    cl_kernel cost_by_layer_kernel = clCreateKernel (cost_by_layer_program, "memset", NULL);

    // - Image Loading
    cv::Mat left_source_image = cv::imread(path_image_left, cv::IMREAD_COLOR);
    cv::Mat right_source_image = cv::imread(path_image_right, cv::IMREAD_COLOR);
    assert(left_source_image.type() == CV_8UC3 && right_source_image.type() == CV_8UC3);

    // - Image Padding
    int padding_size = disparity;
    cv::Mat left_image_padded;
    image_padding(left_source_image,left_image_padded, padding_size);
    cv::Mat right_image_padded;
    image_padding(right_source_image, right_image_padded, padding_size);
    cv::Mat output_layer_cost = cv::Mat::zeros(right_source_image.size(), CV_32FC1); // float

    //- Output allocation
    cv::Mat result = cv::Mat::zeros(left_source_image.size(), CV_32FC1);

    // - Buffer allocation
    assert(left_source_image.total() == right_source_image.total());
    int padded_input_image_1D_size = left_image_padded.total() * left_image_padded.elemSize();
    cl_mem left_input_image_buffer = clCreateBuffer(ocl_stuff.context,
                                               CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                                    padded_input_image_1D_size,
                                               (void*)left_source_image.data, NULL);
    cl_mem right_input_image_buffer = clCreateBuffer(ocl_stuff.context,
                                                    CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                                     padded_input_image_1D_size,
                                                    (void*)right_source_image.data, NULL);
    cl_mem output_buffer = clCreateBuffer(ocl_stuff.context,
                                                     CL_MEM_WRITE_ONLY,
                                                     result.total() * result.elemSize(),
                                                     (void*)right_source_image.data, NULL);
    // - Cost properties
    float alpha_weight = 0.5;

    // - Passing arguments to the kernel
    clSetKernelArg(cost_by_layer_kernel, 0, sizeof(left_input_image_buffer), (void*)&left_input_image_buffer);
    clSetKernelArg(cost_by_layer_kernel, 1, sizeof(right_input_image_buffer), (void*)&right_input_image_buffer);
    clSetKernelArg(cost_by_layer_kernel, 2, sizeof(output_buffer), (void*)&output_buffer);
    clSetKernelArg(cost_by_layer_kernel, 3, sizeof(padding_size), (void*)&padding_size);
    clSetKernelArg(cost_by_layer_kernel, 4, sizeof(disparity), (void*)&disparity);
    clSetKernelArg(cost_by_layer_kernel, 5, sizeof(alpha_weight), (void*)&alpha_weight);

    // - Executing Kernel
    // for each pixel - cost
    size_t global_work_size_cost_layer[] = {(size_t)right_source_image.cols , (size_t) right_source_image.rows };
    clEnqueueNDRangeKernel(ocl_stuff.queue,
                           cost_by_layer_kernel,
                           2,
                           NULL,
                           global_work_size_cost_layer,
                           NULL,
                           0,
                           NULL, NULL);

    clFinish(ocl_stuff.queue);

    // - Reading Results
    clEnqueueReadBuffer(ocl_stuff.queue,
                        output_buffer,
                        CL_TRUE,
                        NULL,
                        result.total() * result.elemSize(),
                        (void*)result.data, NULL, NULL, NULL);
    string output_name = "cost_integrated_" + path_image_left;
    cv::imwrite(output_name, result);
}