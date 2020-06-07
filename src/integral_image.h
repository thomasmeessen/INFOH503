 #ifndef CL_INFOH503_INTEGRAL_IMAGE_H
#define CL_INFOH503_INTEGRAL_IMAGE_H

struct Opencl_buffer;
struct Opencl_stuff;
#include <string>
#include <CL/cl.h>


 struct ScanParameters{
    /** Blocs number is defined by half the number of pixels on the padded image divided by the local work size **/
    int number_blocs;
    int local_size, global_size;

    /**
     * Setup parameters using GPU and Image data.
     * @param image Opencv matrix on which the Scan algorithm will be applied
     * @param ocl_stuff
     */
    ScanParameters(const Opencl_buffer &image, const Opencl_stuff &ocl_stuff, cl_kernel kernel_bloc);

};

void compute_integral_image(Opencl_buffer &image, const Opencl_stuff &ocl_stuff);

Opencl_buffer transpose(std::string image_path, int max_distance, Opencl_stuff ocl_stuff);


#endif //CL_INFOH503_INTEGRAL_IMAGE_H
