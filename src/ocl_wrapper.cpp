#include "ocl_wrapper.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;

int Opencl_buffer::used_memory =0;

void Opencl_buffer::write_img(string path_to_write, Opencl_stuff ocl_stuff, bool to_normalize) {
        cv::Mat image_to_write = cv::Mat::zeros(rows, cols, type);
        clEnqueueReadBuffer(ocl_stuff.queue,
                            buffer,
                            CL_TRUE,
                            0,
                            buffer_size,
                            (void*)image_to_write.data, 0, nullptr, nullptr);
        if(to_normalize){
            cv::Mat normalized_image;
            cv::normalize(image_to_write, normalized_image, 0, 255, cv::NORM_MINMAX);
            cv::imwrite(path_to_write, normalized_image);
        }
        else{
            cv::imwrite(path_to_write, image_to_write);
        }
    }


Opencl_buffer::Opencl_buffer(const string &image_path, Opencl_stuff ocl_stuff, int padding_size) {
        // - Image Loading
        cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        // - Image padding
        if( padding_size != 0){
            cv::Rect extract_zone (padding_size, padding_size, image.cols, image.rows);
            cv::Mat image_padded = cv::Mat::zeros(image.rows + 2 * padding_size, image.cols + 2 * padding_size, image.type());
            image.copyTo(image_padded(extract_zone));
            image = image_padded;
        }
        // - Setting parameters
        cols = image.cols;
        rows = image.rows;
        type = CV_8UC1;
        buffer_size = image.total() * image.elemSize();

        // - Allocating the buffers
        buffer = allocate_buffer(ocl_stuff,
                                CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                buffer_size,
                                (void*)image.data, nullptr);

    }

Opencl_buffer::Opencl_buffer(int rows, int cols, Opencl_stuff ocl_stuff, int type): cols(cols), rows(rows), type(type){
        cv::Mat zero_matrix = cv::Mat::zeros(rows, cols, type);
        buffer_size = zero_matrix.total() * zero_matrix.elemSize();
        buffer = allocate_buffer(ocl_stuff,CL_MEM_COPY_HOST_PTR ,
                                buffer_size,
                                (void*)zero_matrix.data, nullptr);
    }


cl_mem Opencl_buffer::allocate_buffer(Opencl_stuff ocl_stuff, cl_mem_flags flags, size_t size, void * data, cl_int * clInt) {
    if( ocl_stuff.memory_available > used_memory + (int) buffer_size ){
        used_memory += (int) buffer_size;
        return clCreateBuffer(ocl_stuff.context, flags , buffer_size, data, clInt);
    } else{
        throw Out_of_memory_exception(ocl_stuff.memory_available , used_memory + (int) buffer_size );
    }

}

void Opencl_buffer::free() {
    used_memory -= (int) buffer_size;
    clReleaseMemObject(buffer);
}
