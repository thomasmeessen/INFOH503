#ifndef CL_INFOH503_OCL_WRAPPER_H
#define CL_INFOH503_OCL_WRAPPER_H
#include <CL/cl.h>
#include <string>
#include <iostream>
#include <exception>

struct Opencl_stuff {
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    /** Device global available memory in byte **/
    cl_ulong memory_available;
};

struct Out_of_memory_exception : public std::exception{
    cl_ulong available;
    cl_ulong used;

    Out_of_memory_exception(cl_ulong available, cl_ulong used): available(available), used(used){};

    const char *what() const noexcept override {
        std::string message = "Trying to allocate more memory: " +
                std::to_string((unsigned long long)used) +
                " than available: " +
                std::to_string((unsigned long long) available);
        std::cout << message <<std::endl;
        // return  message.c_str();
        // This line fail to encode properly the message
        return "Trying to allocate more memory than available";
    }
};

namespace cv {
    class Mat;
}

struct Opencl_buffer {
    cl_mem buffer;
    std::size_t buffer_size;
    int cols, rows, type, padding_size;
    /** Device global memory usage in byte **/
    static int used_memory;
    Opencl_stuff ocl_stuff;

    void write_img(std::string path_to_write, bool to_normalize = false);


    Opencl_buffer () = default;

    Opencl_buffer(const std::string &image_path, Opencl_stuff ocl_stuff, int padding_size = 0);

    /**
     * Create a buffer from the given Opencv Matrix do not change the data.
     * Padding is set to zero by default.
     * @param ocl_stuff
     */
    Opencl_buffer(const cv::Mat&, Opencl_stuff ocl_stuff, int padding_size = 0);

    /**
     * Create a float buffer initialized with 0
     * @param rows
     * @param cols
     */
    Opencl_buffer(int rows, int cols, Opencl_stuff ocl_stuff);

    /** This method create a new buffer from the existing with the requested padding.
     * If the original buffer had already padding it is removed and replace by the new value.
     * @param padding the requested padding size (with zeros)
     * @return
     */
    Opencl_buffer clone( int padding = 0);

    cv::Mat get_values();


    void free();



private:
    /**
     * Wrapper around opencl allocate buffer for memory management
     * @param ocl_stuff
     */
    cl_mem allocate_buffer(Opencl_stuff ocl_stuff,
                         cl_mem_flags /* flags */,
                         size_t       /* size */,
                         void *       /* host_ptr */,
                         cl_int *     /* errcode_ret */);
};

void test_implementation(Opencl_stuff &ocl_stuff, const std::string &path_to_image);

#endif //CL_INFOH503_OCL_WRAPPER_H
