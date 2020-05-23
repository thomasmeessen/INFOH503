#ifndef CL_INFOH503_OCL_WRAPPER_H
#define CL_INFOH503_OCL_WRAPPER_H
#include <CL/cl.h>
#include <string>

struct Opencl_stuff {
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
};

struct Opencl_buffer {
    cl_mem buffer;
    std::size_t buffer_size;
    int cols, rows, type;

    void write_img(std::string path_to_write, Opencl_stuff ocl_stuff, bool to_normalize);


    Opencl_buffer () = default;

    Opencl_buffer(const std::string & image_path, Opencl_stuff ocl_stuff);

    /**
     * Create a float buffer initialized with 0
     * @param rows
     * @param cols
     */
    Opencl_buffer(int rows, int cols, Opencl_stuff ocl_stuff);
};


#endif //CL_INFOH503_OCL_WRAPPER_H
