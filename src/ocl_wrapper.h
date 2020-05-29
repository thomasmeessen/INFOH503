#ifndef CL_INFOH503_OCL_WRAPPER_H
#define CL_INFOH503_OCL_WRAPPER_H
#include <CL/cl.h>
#include <string>
#include <iostream>
#include "integral_image.h"
#include <opencv2/core.hpp>

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

struct Opencl_buffer {
    cl_mem buffer;
    std::size_t buffer_size;
    int cols, rows, type;
    /** Device global memory usage in byte **/
    static int used_memory;

    void write_img(std::string path_to_write, Opencl_stuff ocl_stuff, bool to_normalize);


    Opencl_buffer () = default;

    Opencl_buffer(const std::string &image_path, Opencl_stuff ocl_stuff, int padding_size  = 0, int type = CV_8UC1);

    /**
     * Create a float buffer initialized with 0
     * @param rows
     * @param cols
     */
    Opencl_buffer(int rows, int cols, Opencl_stuff ocl_stuff, int type = CV_32FC1);

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



#endif //CL_INFOH503_OCL_WRAPPER_H
