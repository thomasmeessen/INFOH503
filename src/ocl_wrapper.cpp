#include "ocl_wrapper.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;

int Opencl_buffer::used_memory =0;





cv::Mat Opencl_buffer::get_values1(int size, int _rows, cv::Mat* matrix) {
    rows = _rows;
    cv::Mat final_matrix;
    for (int i = 0; i < 16; i++) {
        cv::Mat image_to_write = cv::Mat::zeros(rows, cols, type);
        clEnqueueReadBuffer(ocl_stuff.queue,
            buffer,
            CL_TRUE,
            size*i,
            size,
            (void*)image_to_write.data, 0, nullptr, nullptr);
        matrix[i] = image_to_write;
    }


    clFinish(ocl_stuff.queue);

    return final_matrix;


}


cv::Mat Opencl_buffer::get_values() {
    cv::Mat image_to_write = cv::Mat::zeros(rows, cols, type);
    assert(image_to_write.total() * image_to_write.elemSize() == buffer_size);
    clEnqueueReadBuffer(ocl_stuff.queue,
            buffer,
            CL_TRUE,
            0,
            buffer_size,
            (void*)image_to_write.data, 0, nullptr, nullptr);


    clFinish(ocl_stuff.queue);

    return image_to_write;
}

void Opencl_buffer::write_img(std::string path_to_write, bool to_normalize) {
        cv::Mat image_to_write = get_values();
        if(to_normalize){
            cv::Mat normalized_image;
            cv::normalize(image_to_write, normalized_image, 0, 255, cv::NORM_MINMAX);
            cv::imwrite(path_to_write, normalized_image);
        }
        else{
            cv::imwrite(path_to_write, image_to_write);
        }
    }

Opencl_buffer Opencl_buffer::clone( int new_padding_size, int _type){
    // - Reading Image From device
    cv::Mat image = get_values();
    // - Image padding
    int old_ps = padding_size;

    int type = image.type();
    if (_type != 0) {
        type = _type;
    }

    if( new_padding_size != 0 || old_ps != 0){
        // Area that does not include existing padding;
        cv::Rect data_to_copy_area(old_ps, old_ps, cols - 2*old_ps, rows - 2*old_ps);
        // A area to copy the data(with any existing padding removed)
        cv::Rect extract_zone (new_padding_size, new_padding_size, data_to_copy_area.width, data_to_copy_area.height);
        // The new matrix with space for padding
        cv::Mat image_padded = cv::Mat::zeros(data_to_copy_area.height + 2 * new_padding_size, data_to_copy_area.width + 2 * new_padding_size, type);
        image(data_to_copy_area).copyTo(image_padded(extract_zone));
        image = image_padded;

    }
    // - Allocating new buffer
   Opencl_buffer new_buffer(image, ocl_stuff, new_padding_size, type);
    return new_buffer;
}

Opencl_buffer::Opencl_buffer(const string &image_path, Opencl_stuff ocl_stuff, int padding_size,int type ):ocl_stuff(ocl_stuff) {
        // - Image Loading
        cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        // - Image padding
        if( padding_size != 0){
            cv::Rect extract_zone (padding_size, padding_size, image.cols, image.rows);
            cv::Mat image_padded = cv::Mat::zeros(image.rows + 2 * padding_size, image.cols + 2 * padding_size, image.type());
            image.copyTo(image_padded(extract_zone));
            image = image_padded;
        }
        // - Conversion
        image.convertTo(image, type);
        // - Setting parameters
        cols = image.cols;
        rows = image.rows;
        this->type = image.type();
        type = image.type();
        buffer_size = image.total() * image.elemSize();

        // - Allocating the buffers
        buffer = allocate_buffer(ocl_stuff,
                                CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE,
                                buffer_size,
                                (void*)image.data, nullptr);

    }

Opencl_buffer::Opencl_buffer(int rows, int cols, Opencl_stuff ocl_stuff, int type):type(type), cols(cols), rows(rows), ocl_stuff(ocl_stuff){
        cv::Mat zero_matrix = cv::Mat::zeros(rows, cols, CV_32FC1);
        buffer_size = zero_matrix.total() * zero_matrix.elemSize();
        padding_size = 0;
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

Opencl_buffer::Opencl_buffer(cv::Mat& data, Opencl_stuff ocl_stuff, int padding_size, int _type, int dupa):  type(data.type()),
                                                                            buffer_size(data.total() * data.elemSize()),
                                                                            cols(data.cols), rows(data.rows),
                                                                            ocl_stuff(ocl_stuff),
                                                                            padding_size(padding_size){

    if (dupa != 0) {
        cout << "data type " << _type << " " << data.type() << endl;
    }

    if (_type != data.type()) {
        cv::Mat zero_matrix;
        type = _type;
        data.convertTo(zero_matrix, _type);
        buffer_size = zero_matrix.total() * zero_matrix.elemSize();
        data = zero_matrix;
    }



    buffer = allocate_buffer(ocl_stuff,CL_MEM_COPY_HOST_PTR ,
                    buffer_size,
                    (void*)data.data, nullptr);

}


void test_implementation(Opencl_stuff &ocl_stuff, const std::string &path_to_image){
    Opencl_buffer test_image (path_to_image, ocl_stuff);
    test_image.write_img("test_reference.jpg");
    Opencl_buffer copy_with_padding = test_image.clone(20);
    copy_with_padding.write_img("test_20_padding.jpg");
    Opencl_buffer copy_no_padding = test_image.clone();
    copy_no_padding.write_img("test_no_padding.jpg");
    Opencl_buffer copy_from_padding_to_no_padding = copy_with_padding.clone(0);
    copy_from_padding_to_no_padding.write_img("test_padding_suppression.jpg");
}