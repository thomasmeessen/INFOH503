#include "ocl_wrapper.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;


void Opencl_buffer::write_img(string path_to_write, Opencl_stuff ocl_stuff, bool to_normalize) {
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


Opencl_buffer::Opencl_buffer(const string & image_path, Opencl_stuff ocl_stuff){
        // - Image Loading
        cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        // - Setting parameters
        cols = image.cols;
        rows = image.rows;
        type = CV_8UC1;
        buffer_size = image.total() * image.elemSize();

        // - Allocating the buffers
        buffer = allocate_buffer(ocl_stuff,
                                CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                buffer_size,
                                (void*)image.data, NULL);

    }

Opencl_buffer::Opencl_buffer(int rows, int cols, Opencl_stuff ocl_stuff): cols(cols), rows(rows){
        cv::Mat zero_matrix = cv::Mat::zeros(rows, cols, CV_32FC1);
        buffer_size = zero_matrix.total() * zero_matrix.elemSize();
        type = CV_32FC1;
        buffer = allocate_buffer(ocl_stuff,CL_MEM_COPY_HOST_PTR ,
                                buffer_size,
                                (void*)zero_matrix.data, NULL);
    }

Opencl_buffer::Opencl_buffer(cl_mem buffer, std::size_t buffer_size, int type, int rows, int cols) : buffer(buffer), buffer_size(buffer_size), type(type), rows(rows), cols(cols) {}


cl_mem Opencl_buffer::allocate_buffer(Opencl_stuff ocl_stuff, cl_mem_flags flags, size_t size, void * data, cl_int * clInt) {
    return clCreateBuffer(ocl_stuff.context, flags , buffer_size, data, clInt);
}
