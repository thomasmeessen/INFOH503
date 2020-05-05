#include <iostream>

#include "tools.h"

using namespace std;

struct Opencl_stuff {
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
};

struct Opencl_buffer {
    cl_mem buffer;
    std::size_t buffer_size;
    int cols, rows, type;

    void write_img(string path_to_write, Opencl_stuff ocl_stuff, bool to_normalize) {
        cv::Mat image_to_write = cv::Mat::zeros(rows, cols, type);
        clEnqueueReadBuffer(ocl_stuff.queue,
            buffer,
            CL_TRUE,
            NULL,
            buffer_size,
            (void*)image_to_write.data, NULL, NULL, NULL);
        if(to_normalize){
            cv::Mat normalized_image;
            cv::normalize(image_to_write, normalized_image, 0, 255, cv::NORM_MINMAX);
            cv::imwrite(path_to_write, normalized_image);
        }
        else{
            cv::imwrite(path_to_write, image_to_write);
        }
    }


    Opencl_buffer () = default;

    Opencl_buffer(const string & image_path, Opencl_stuff ocl_stuff){
        // - Image Loading
        cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        // - Setting parameters
        cols = image.cols;
        rows = image.rows;
        type = CV_8UC1;
        buffer_size = image.total() * image.elemSize();

        // - Allocating the buffers
        buffer = clCreateBuffer(ocl_stuff.context,
                                CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                buffer_size,
                                (void*)image.data, NULL);

    }

    /**
     * Create a float buffer initialized with 0
     * @param rows
     * @param cols
     */
    Opencl_buffer(int rows, int cols, Opencl_stuff ocl_stuff): cols(cols), rows(rows){
        cv::Mat zero_matrix = cv::Mat::zeros(rows, cols, CV_32FC1);
        buffer_size = zero_matrix.total() * zero_matrix.elemSize();
        type = CV_32FC1;
        buffer = clCreateBuffer(ocl_stuff.context,CL_MEM_COPY_HOST_PTR ,
                                buffer_size,
                                (void*)zero_matrix.data, NULL);
    }

};

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



void image_padding(cv::Mat& image, cv::Mat& dest, int padding_size){
    cv::Rect extract_zone(padding_size, padding_size, image.cols, image.rows);
    cv::Mat image_padded = cv::Mat::zeros(image.rows + 2 * padding_size, image.cols + 2 * padding_size, image.type());
    image.copyTo(image_padded(extract_zone));
    dest = image_padded;
}



Opencl_buffer guidedFilter(cv::Mat& left_source_image, int max_distance, cl_context context, cl_kernel kernel, cl_kernel kernel0, cl_command_queue queue, struct Opencl_buffer costBuffer, Opencl_stuff ocl_stuff, const string* image_path = NULL) {

    //added padding to images

    cv::Mat left_image_padded;
    image_padding(left_source_image, left_image_padded, max_distance);




    int width = left_image_padded.cols - 2 * max_distance; // because the image is padded
    int height = left_image_padded.rows - 2 * max_distance;


    cv::Mat final_image = cv::Mat(costBuffer.rows, costBuffer.cols, CV_32FC1);

    cv::Mat cost_list = cv::Mat(left_image_padded.rows * max_distance, left_image_padded.cols, costBuffer.type);

    clEnqueueReadBuffer(ocl_stuff.queue,
            costBuffer.buffer,
            CL_TRUE,
            NULL,
            costBuffer.buffer_size,
            (void*)cost_list.data, NULL, NULL, NULL);

    int image_1D_size = left_image_padded.total() * left_image_padded.elemSize();

    cl_mem buffer = clCreateBuffer(context,
        CL_MEM_COPY_HOST_PTR,
        image_1D_size,
        (void*)left_image_padded.data, NULL);


    cv::Mat output_a_k_list = cv::Mat(left_image_padded.rows * max_distance, left_image_padded.cols, CV_32FC1);
    cv::Mat output_b_k_list = cv::Mat(left_image_padded.rows * max_distance, left_image_padded.cols, CV_32FC1);




    cl_mem output_a_k_buffer_list = clCreateBuffer(context,
        CL_MEM_COPY_HOST_PTR,
        output_a_k_list.total() * output_a_k_list.elemSize(),
        (void*)output_a_k_list.data, NULL);

    cl_mem output_b_k_buffer_list = clCreateBuffer(context,
        CL_MEM_COPY_HOST_PTR,
        output_b_k_list.total() * output_b_k_list.elemSize(),
        (void*)output_b_k_list.data, NULL);

    cl_mem cost_buffer_list = clCreateBuffer(context,
        CL_MEM_COPY_HOST_PTR,
        cost_list.total() * cost_list.elemSize(),
        (void*)cost_list.data, NULL);


    // 6. Launch the kernel. Let OpenCL pick the local work size.
    clSetKernelArg(kernel, 0, sizeof(buffer), (void*)&buffer);
    clSetKernelArg(kernel, 1, sizeof(output_a_k_buffer_list), (void*)&output_a_k_buffer_list);
    clSetKernelArg(kernel, 2, sizeof(output_b_k_buffer_list), (void*)&output_b_k_buffer_list);
    clSetKernelArg(kernel, 3, sizeof(cost_buffer_list), (void*)&cost_buffer_list);
    clSetKernelArg(kernel, 4, sizeof(width), &width);
    clSetKernelArg(kernel, 5, sizeof(height),&height);
    clSetKernelArg(kernel, 6, sizeof(max_distance), &max_distance);

    size_t global_work_size_image[] = { (size_t)left_image_padded.cols - 2*max_distance, (size_t)left_image_padded.rows- 2*max_distance, (size_t)max_distance }; // don't work on pixels in the padding hence the "- 2*max_distance"

    clEnqueueNDRangeKernel(queue,
        kernel,
        3,
        NULL,
        global_work_size_image,
        NULL,
        0,
        NULL, NULL);

    clFinish(queue); // syncing

    // 7. Look at the results via synchronous buffer map.

    clEnqueueReadBuffer(queue,
        output_a_k_buffer_list,
        CL_TRUE,
        NULL,
        output_a_k_list.total()* output_a_k_list.elemSize(),
        (void*)output_a_k_list.data, NULL, NULL, NULL);


    clEnqueueReadBuffer(queue,
        output_b_k_buffer_list,
        CL_TRUE,
        NULL,
        output_b_k_list.total()* output_b_k_list.elemSize(),
        (void*)output_b_k_list.data, NULL, NULL, NULL);

    if (image_path != NULL) {
        string output_name_a_k = "guided_a_k_normalized_" + *image_path;
        string output_name_b_k = "guided_b_k_normalized_" + *image_path;
        cv::Mat normalized_image = cv::Mat::zeros(final_image.rows, final_image.cols, final_image.type());
        cv::normalize(output_a_k_list, normalized_image, 0, 255, cv::NORM_MINMAX);
        cv::imwrite(output_name_a_k, normalized_image);
        cv::normalize(output_b_k_list, normalized_image, 0.0, 255.0, cv::NORM_MINMAX);
        cv::imwrite(output_name_b_k, normalized_image);
    }

    //Seconde pass
    cv::Mat guidedFilter_image = cv::Mat((left_image_padded.rows - 2*max_distance)*max_distance, left_image_padded.cols- 2*max_distance, CV_32FC1);

    cl_mem guidedFilter_image_buffer = clCreateBuffer(context,
        CL_MEM_COPY_HOST_PTR,
        guidedFilter_image.total()*guidedFilter_image.elemSize(),
        (void*)guidedFilter_image.data, NULL);

    clSetKernelArg(kernel0, 0, sizeof(buffer), (void*)&buffer);
    clSetKernelArg(kernel0, 1, sizeof(output_a_k_buffer_list), (void*)&output_a_k_buffer_list);
    clSetKernelArg(kernel0, 2, sizeof(output_b_k_buffer_list), (void*)&output_b_k_buffer_list);
    clSetKernelArg(kernel0, 3, sizeof(guidedFilter_image_buffer), (void*)&guidedFilter_image_buffer);
    clSetKernelArg(kernel0, 4, sizeof(width), &width);
    clSetKernelArg(kernel0, 5, sizeof(height), &height);
    clSetKernelArg(kernel0, 6, sizeof(max_distance), &max_distance);


    clEnqueueNDRangeKernel(queue,
        kernel0,
        3,
        NULL,
        global_work_size_image,
        NULL,
        0,
        NULL, NULL);

    clFinish(queue); // syncing

    // - Packing Results
    Opencl_buffer result;
    result.buffer = guidedFilter_image_buffer;
    result.type = CV_32FC1;
    result.buffer_size = guidedFilter_image.total() * guidedFilter_image.elemSize();
    result.cols = guidedFilter_image.cols;
    result.rows = guidedFilter_image.rows;
    return  result;


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



Opencl_buffer cost_range_layer(const string &left_source_image_path,const string &right_source_image_path, int disparity_range, int disparity_sign, cl_kernel cost_volume_kernel, Opencl_stuff ocl_stuff)  {


    // - Padding
    int padding_size = disparity_range;
    float alpha_weight = 0.9;
    float t1 = 7.;//tau 1
    float t2 = 2.;//tau 2 later add as parameter function

    Opencl_buffer left_image_buffer(left_source_image_path, ocl_stuff);
    Opencl_buffer right_image_buffer(right_source_image_path, ocl_stuff);

    cv::Mat output_cost = cv::Mat::zeros((left_image_buffer.rows + 2 * padding_size) * disparity_range, left_image_buffer.cols + 2 * padding_size , CV_32FC1);// we generate all layers


    // - Allocating the buffers

    cl_mem cost_output_buffer = clCreateBuffer(ocl_stuff.context,
                                               CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR,
                                               output_cost.total() * output_cost.elemSize(),
                                               (void*)output_cost.data, NULL);

    // - Passing arguments to the kernel
    clSetKernelArg(cost_volume_kernel, 0, sizeof(left_image_buffer.buffer), (void*)&left_image_buffer.buffer);
    clSetKernelArg(cost_volume_kernel, 1, sizeof(right_image_buffer.buffer), (void*)&right_image_buffer.buffer);
    clSetKernelArg(cost_volume_kernel, 2, sizeof(cost_output_buffer), (void*)&cost_output_buffer);
    clSetKernelArg(cost_volume_kernel, 3, sizeof(padding_size), (void*)&padding_size);
    clSetKernelArg(cost_volume_kernel, 4, sizeof(alpha_weight), (void*)&alpha_weight);
    clSetKernelArg(cost_volume_kernel, 5, sizeof(t1), (void*)&t1);
    clSetKernelArg(cost_volume_kernel, 6, sizeof(t2), (void*)&t2);
    clSetKernelArg(cost_volume_kernel, 7, sizeof(disparity_sign),  (void*)&disparity_sign);

    // - Enqueuing kernel
    size_t global_work_size_cost_layer[] = { (size_t)left_image_buffer.cols , (size_t)left_image_buffer.rows, (size_t)disparity_range };
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

    // - Packing Results
    Opencl_buffer result;
    result.buffer = cost_output_buffer;
    result.type = CV_32FC1;
    result.buffer_size = output_cost.total() * output_cost.elemSize();
    result.cols = output_cost.cols;
    result.rows = output_cost.rows;
    return  result;
}


Opencl_buffer cost_selection(Opencl_buffer filtered_cost, int disparity_range, cl_kernel kernel, Opencl_stuff ocl_stuff, const string* image_path = NULL){
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

Opencl_buffer left_right_consistency(){
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