#include <iostream>

#include "tools.h"

using namespace std;

void print_device_info(cl_device_id device){
    printf("======Device Info======\n");
    // https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clGetDeviceInfo.html
    size_t retSize;
    char info[1024];
    cl_ulong buf_ulong;
    size_t buf_size_t;
    cl_device_fp_config fc;

    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(info), info, NULL);
    printf("Device name = %s\n", info);

    clGetDeviceInfo(device,CL_DEVICE_VERSION, sizeof(info), info,NULL);
    printf("OpenCL version = %s\n", info); 

    clGetDeviceInfo(device,CL_DRIVER_VERSION, sizeof(info), info,NULL);
    printf("Driver version = %s\n", info); 
    
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
    printf("Global memory size :%llu B\n", (unsigned long long)buf_ulong);

    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
    printf("Maximum allocatable memory size :%llu B\n", (unsigned long long)buf_ulong);

    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
    printf("Local memory size :%llu B\n", (unsigned long long)buf_ulong);

    clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
    printf("Max constant buffer size:%llu B\n", (unsigned long long)buf_ulong);
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
    printf("Max mem alloc size size:%llu B\n", (unsigned long long)buf_ulong);
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(buf_size_t), &buf_size_t, NULL);
    printf("Max work group size:%zu \n", buf_size_t);
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(buf_size_t), &buf_size_t, NULL);
    printf("Max work item size:%zu \n", buf_size_t);
        clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, //https://gist.github.com/aandergr/ca25ace6c11dbae8237feca7758d1da8
					sizeof(cl_device_fp_config), &fc, NULL);
    printf("%s:%s%s%s%s%s%s\n", "CL_DEVICE_SINGLE_FP_CONFIG",
            fc & CL_FP_DENORM ? " DENORM" : "",
            fc & CL_FP_INF_NAN ? " INF_NAN" : "",
            fc & CL_FP_ROUND_TO_NEAREST ? " ROUND_TO_NEAREST" : "",
            fc & CL_FP_ROUND_TO_ZERO ? " ROUND_TO_ZERO" : "",
            fc & CL_FP_ROUND_TO_INF ? " ROUND_TO_INF" : "",
            fc & CL_FP_FMA ? " FMA" : "");

    clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG,
					sizeof(cl_device_fp_config), &fc, NULL);
    printf("%s:%s%s%s%s%s%s\n", "CL_DEVICE_DOUBLE_FP_CONFIG",
            fc & CL_FP_DENORM ? " DENORM" : "",
            fc & CL_FP_INF_NAN ? " INF_NAN" : "",
            fc & CL_FP_ROUND_TO_NEAREST ? " ROUND_TO_NEAREST" : "",
            fc & CL_FP_ROUND_TO_ZERO ? " ROUND_TO_ZERO" : "",
            fc & CL_FP_ROUND_TO_INF ? " ROUND_TO_INF" : "",
            fc & CL_FP_FMA ? " FMA" : "");

    printf("=======================\n");

}

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



Opencl_buffer median_filter(Opencl_buffer input, cl_kernel kernel, int max_distance, Opencl_stuff ocl_stuff) {
    // - Buffer for the guiding image (added padding)
    Opencl_buffer median_image(input.rows - 2 * max_distance, input.cols - 2 * max_distance, ocl_stuff);


    int width = median_image.cols; // because the image is padded
    int height = median_image.rows;

    clSetKernelArg(kernel, 0, sizeof(input.buffer), (void*)&input.buffer);
    clSetKernelArg(kernel, 1, sizeof(median_image.buffer), (void*)&median_image.buffer);
    clSetKernelArg(kernel, 2, sizeof(max_distance), &max_distance);


    size_t global_work_size_image[] = { (size_t)width, (size_t)height}; // don't work on pixels in the padding hence the "- 2*max_distance"

    clEnqueueNDRangeKernel(ocl_stuff.queue,
        kernel,
        2,
        nullptr,
        global_work_size_image,
        nullptr,
        0,
        nullptr, nullptr);

    clFinish(ocl_stuff.queue); // syncing


    return median_image;
}



Opencl_buffer padding_calc(Opencl_buffer input, cl_kernel kernel, int max_distance, Opencl_stuff ocl_stuff) {
    // - Buffer for the guiding image (added padding)
    Opencl_buffer image_padded = input.clone(0);



    int width = image_padded.cols; // because the image is padded
    int height = image_padded.rows - 2 * max_distance;
    int kernel_width = width - 2 * max_distance;

    clSetKernelArg(kernel, 0, sizeof(image_padded.buffer), (void*)&image_padded.buffer);
    clSetKernelArg(kernel, 1, sizeof(max_distance), &max_distance);
    clSetKernelArg(kernel, 2, sizeof(width), &width);
    clSetKernelArg(kernel, 3, sizeof(height), &height);


    size_t global_work_size_image[] = { (size_t)kernel_width };

    clEnqueueNDRangeKernel(ocl_stuff.queue,
        kernel,
        1,
        nullptr,
        global_work_size_image,
        nullptr,
        0,
        nullptr, nullptr);

    clFinish(ocl_stuff.queue); // syncing

    image_padded.write_img("TESTESTEST.png", true);



    return image_padded;


}



Opencl_buffer integral_image_cost(Opencl_buffer costBuffer, cl_kernel padding_kernel, int max_distance, Opencl_stuff ocl_stuff) {
    cv::Mat matrix[16];
    costBuffer.get_values1((costBuffer.buffer_size / 16), costBuffer.rows / 16, matrix);
    Opencl_buffer first = Opencl_buffer(matrix[0], ocl_stuff, 0, CV_32FC1);
    compute_integral_image(first, ocl_stuff);
    first = padding_calc(first, padding_kernel, max_distance, ocl_stuff);
    first.write_img("concat_first.png", true);
    cv::Mat final_matrix = first.get_values();
   for (int i = 1; i < 16; i++) {
        Opencl_buffer test = Opencl_buffer(matrix[i], ocl_stuff, 0, CV_32FC1);
        compute_integral_image(test, ocl_stuff);
        Opencl_buffer integral_image_padded = padding_calc(test, padding_kernel, max_distance, ocl_stuff);

        cv::vconcat(final_matrix, integral_image_padded.get_values(), final_matrix);
       // cv::vconcat(final_matrix, first.get_values(), final_matrix);
    }
    Opencl_buffer test1 = Opencl_buffer(final_matrix, ocl_stuff, 0, CV_32FC1);
     test1.write_img("concat.png", true);

    return test1;
}


Opencl_buffer guidedFilter(string guiding_image_path, int max_distance, cl_kernel kernel, cl_kernel kernel0,
             struct Opencl_buffer costBuffer, Opencl_stuff ocl_stuff, cl_kernel padding_kernel, int radius = 9) {


    Opencl_buffer guiding_image_buffer(guiding_image_path, ocl_stuff, max_distance);
     Opencl_buffer integral_image = guiding_image_buffer.clone(0, CV_32FC1);
     Opencl_buffer integral_image_padded= padding_calc(integral_image, padding_kernel, max_distance, ocl_stuff);

    compute_integral_image(integral_image_padded, ocl_stuff);
    integral_image_padded.write_img("paddedThisTHIs.png", true);

    Opencl_buffer costBufferIntegral= integral_image_cost(costBuffer, padding_kernel, max_distance, ocl_stuff);
    //Opencl_buffer costBufferIntegral;
   // guiding_image_buffer.write_img("Guided_image_test.png", true);


    int width = guiding_image_buffer.cols - 2 * max_distance; // because the image is padded
    int height = guiding_image_buffer.rows - 2 * max_distance;


    // - Buffer to keep Ak an Bk - no initialisation
    Opencl_buffer b_k_buffer(guiding_image_buffer.rows * max_distance, guiding_image_buffer.cols, ocl_stuff);
    Opencl_buffer a_k_buffer(guiding_image_buffer.rows * max_distance, guiding_image_buffer.cols, ocl_stuff);

    // 6. Launch the kernel. Let OpenCL pick the local work size.
    clSetKernelArg(kernel, 0, sizeof(guiding_image_buffer.buffer), (void*)&guiding_image_buffer.buffer);
    clSetKernelArg(kernel, 1, sizeof(a_k_buffer.buffer), (void*)&a_k_buffer.buffer);
    clSetKernelArg(kernel, 2, sizeof(b_k_buffer.buffer), (void*)&b_k_buffer.buffer);
    clSetKernelArg(kernel, 3, sizeof(costBuffer.buffer), (void*)&costBuffer.buffer);
    clSetKernelArg(kernel, 4, sizeof(width), &width);
    clSetKernelArg(kernel, 5, sizeof(height),&height);
    clSetKernelArg(kernel, 6, sizeof(max_distance), &max_distance);
    clSetKernelArg(kernel, 7, sizeof(radius), &radius);
    clSetKernelArg(kernel, 8, sizeof(integral_image_padded.buffer), &integral_image_padded);
    clSetKernelArg(kernel, 9, sizeof(costBufferIntegral.buffer), &costBufferIntegral);

    size_t global_work_size_image[] = {(size_t)width, (size_t)height, (size_t)max_distance }; // don't work on pixels in the padding hence the "- 2*max_distance"

    clEnqueueNDRangeKernel(ocl_stuff.queue,
        kernel,
        3,
        nullptr,
        global_work_size_image,
        nullptr,
        0,
        nullptr, nullptr);

    clFinish(ocl_stuff.queue); // syncing

    // - Read Ak and Bk from the first pass


    a_k_buffer.write_img("guided_a_k_normalized_" + guiding_image_path, true);
    b_k_buffer.write_img("guided_b_k_normalized_" + guiding_image_path, true);


    //Seconde pass
    Opencl_buffer output_filter_buffer (height * max_distance, width,  ocl_stuff);

    clSetKernelArg(kernel0, 0, sizeof(guiding_image_buffer.buffer), (void*)&guiding_image_buffer.buffer);
    clSetKernelArg(kernel0, 1, sizeof(a_k_buffer.buffer), (void*)&a_k_buffer.buffer);
    clSetKernelArg(kernel0, 2, sizeof(b_k_buffer.buffer), (void*)&b_k_buffer.buffer);
    clSetKernelArg(kernel0, 3, sizeof(output_filter_buffer.buffer), (void*)&output_filter_buffer.buffer);
    clSetKernelArg(kernel0, 4, sizeof(width), &width);
    clSetKernelArg(kernel0, 5, sizeof(height), &height);
    clSetKernelArg(kernel0, 6, sizeof(max_distance), &max_distance);
    clSetKernelArg(kernel0, 7, sizeof(radius), &radius);


    clEnqueueNDRangeKernel(ocl_stuff.queue,
        kernel0,
        3,
        NULL,
        global_work_size_image,
        NULL,
        0,
        NULL, NULL);

    clFinish(ocl_stuff.queue); // syncing

    // - Freeing memory
    a_k_buffer.free();
    b_k_buffer.free();
    integral_image.free();
    guiding_image_buffer.free();

    return  output_filter_buffer;
}




void image_difference(cv::Mat& left_image, cv::Mat& right_image, cv::Mat& output_image, int max_distance, cl_context context, cl_kernel kernel, cl_command_queue queue, bool write_to_png) {
    int image_1D_size = left_image.total() * left_image.elemSize();
    int original_width = left_image.cols - 2 * max_distance; // cause left_image is the original image + the pading
    int original_height = left_image.rows - 2 * max_distance;

    output_image = cv::Mat((original_height + 2 * max_distance), (original_width + 2 * max_distance) * max_distance, CV_8U);   // each image will be next to each other and we add padding to it

    int output_size = output_image.total() * output_image.elemSize(); //https://answers.opencv.org/question/21296/cvmat-data-size/

    cl_mem left_image_buffer = clCreateBuffer(context,
        CL_MEM_COPY_HOST_PTR,
        image_1D_size,
        (void*)left_image.data, nullptr);
    clSetKernelArg(kernel, 0, sizeof(left_image_buffer), (void*)&left_image_buffer);//https://stackoverflow.com/a/22101104

    cl_mem right_image_buffer = clCreateBuffer(context,
        CL_MEM_COPY_HOST_PTR,
        image_1D_size,
        (void*)right_image.data, nullptr);

    clSetKernelArg(kernel, 1, sizeof(right_image_buffer), (void*)&right_image_buffer);//https://stackoverflow.com/a/22101104

    cl_mem destination_buffer = clCreateBuffer(context,
        CL_MEM_COPY_HOST_PTR,
        output_size,
        (void*)output_image.data, nullptr);

    clSetKernelArg(kernel, 2, sizeof(destination_buffer), (void*)&destination_buffer);
    clSetKernelArg(kernel, 3, sizeof(original_height), &original_height);//set width value

    clSetKernelArg(kernel, 4, sizeof(original_width), &original_width);//set width value
    clSetKernelArg(kernel, 5, sizeof(max_distance), &max_distance);//set maxDistance value

    // 6. Launch the kernel. Let OpenCL pick the local work size.

    size_t global_work_size_image[] = { (size_t)original_width, (size_t)original_height };
    clEnqueueNDRangeKernel(queue,
        kernel,
        2,
        nullptr,
        global_work_size_image,
        nullptr,
        0,
        nullptr, nullptr);

    clFinish(queue); // syncing

    // 7. Look at the results via synchronous buffer map.
    clEnqueueReadBuffer(queue,
        destination_buffer,
        CL_TRUE,
        0,
        output_size,
        (void*)output_image.data, 0, nullptr, nullptr);
    if (write_to_png) {
        cv::imwrite("img_diff.png", output_image);
    }


}



Opencl_buffer cost_range_layer(const string &start_image_path, const string &end_image_path, int disparity_range,
                               cl_kernel cost_volume_kernel, Opencl_stuff ocl_stuff, int disparity_sign) {


    // - Padding
    int padding_size = disparity_range;
    float alpha_weight = 0.9;
    float t1 = 7.;//tau 1
    float t2 = 2.;//tau 2 later add as parameter function

    Opencl_buffer start_image(start_image_path, ocl_stuff, 0);
    Opencl_buffer end_image(end_image_path, ocl_stuff, 0);

    // Allcoating a buffer with 0
    Opencl_buffer cost_output_buffer ((start_image.rows + 2 * padding_size) * disparity_range, start_image.cols + 2 * padding_size, ocl_stuff);


    // - Passing arguments to the kernel
    clSetKernelArg(cost_volume_kernel, 0, sizeof(start_image.buffer), (void*)&start_image.buffer);
    clSetKernelArg(cost_volume_kernel, 1, sizeof(end_image.buffer), (void*)&end_image.buffer);
    clSetKernelArg(cost_volume_kernel, 2, sizeof(cost_output_buffer.buffer), (void*)&cost_output_buffer.buffer);
    clSetKernelArg(cost_volume_kernel, 3, sizeof(padding_size), (void*)&padding_size);
    clSetKernelArg(cost_volume_kernel, 4, sizeof(alpha_weight), (void*)&alpha_weight);
    clSetKernelArg(cost_volume_kernel, 5, sizeof(t1), (void*)&t1);
    clSetKernelArg(cost_volume_kernel, 6, sizeof(t2), (void*)&t2);
    clSetKernelArg(cost_volume_kernel, 7, sizeof(disparity_sign),  (void*)&disparity_sign);

    // - Enqueuing kernel
    size_t global_work_size_cost_layer[] = {(size_t)start_image.cols , (size_t)start_image.rows, (size_t)disparity_range };
    clEnqueueNDRangeKernel(ocl_stuff.queue,
                           cost_volume_kernel,
                           3,
                           NULL,
                           global_work_size_cost_layer,
                           NULL,
                           0,
                           NULL, NULL);

    // - Waiting end execution
    clFinish(ocl_stuff.queue);

    //- Freeing memory
    start_image.free();
    end_image.free();
    return  cost_output_buffer;
}


Opencl_buffer cost_selection(Opencl_buffer filtered_cost, int disparity_range, cl_kernel kernel, Opencl_stuff ocl_stuff){
    const int image_width = filtered_cost.cols;
    const int image_height = filtered_cost.rows / disparity_range; // cause disparity layers


    Opencl_buffer output_buffer (image_height, image_width, ocl_stuff);

    clSetKernelArg(kernel, 0, sizeof(filtered_cost.buffer), (void*)&filtered_cost.buffer);
    clSetKernelArg(kernel, 1, sizeof(disparity_range), (void*)&disparity_range);
    clSetKernelArg(kernel, 2, sizeof(output_buffer.buffer), (void*)&output_buffer.buffer);
    size_t global_work_size_image[] = { (size_t)image_width, (size_t)image_height };
    clEnqueueNDRangeKernel(ocl_stuff.queue,
        kernel,
        2,
        NULL,
        global_work_size_image,
        NULL,
        0,
        NULL, NULL);
    // - Waiting end execution

    clFinish(ocl_stuff.queue);

    return output_buffer;
}

Opencl_buffer left_right_consistency(Opencl_buffer left_depth_map, Opencl_buffer right_depth_map, cl_kernel kernel, Opencl_stuff ocl_stuff){
    const int image_width = left_depth_map.cols;
    const int image_height = left_depth_map.rows; 

    Opencl_buffer output_buffer (image_height, image_width, ocl_stuff);
    
    clSetKernelArg(kernel, 0, sizeof(left_depth_map.buffer), (void*)&left_depth_map.buffer);
    clSetKernelArg(kernel, 1, sizeof(right_depth_map.buffer), (void*)&right_depth_map.buffer);
    clSetKernelArg(kernel, 2, sizeof(output_buffer.buffer), (void*)&output_buffer.buffer);
   
    size_t global_work_size_image[] = { (size_t)image_width, (size_t)image_height }; 
    clEnqueueNDRangeKernel(ocl_stuff.queue,
        kernel,
        2,
        NULL,
        global_work_size_image,
        NULL,
        0,
        NULL, NULL);
    clFinish(ocl_stuff.queue);
    return output_buffer;
}

void densification(Opencl_buffer left_depth_map, Opencl_buffer consistent_depth_map, cl_kernel kernel, Opencl_stuff ocl_stuff){
    
    clSetKernelArg(kernel, 0, sizeof(left_depth_map.buffer), (void*)&left_depth_map.buffer);
    clSetKernelArg(kernel, 1, sizeof(consistent_depth_map.buffer), (void*)&consistent_depth_map.buffer);
   
    size_t global_work_size_image[] = { (size_t)left_depth_map.cols, (size_t)left_depth_map.cols }; 
    clEnqueueNDRangeKernel(ocl_stuff.queue,
        kernel,
        2,
        NULL,
        global_work_size_image,
        NULL,
        0,
        NULL, NULL);
    clFinish(ocl_stuff.queue);
}
