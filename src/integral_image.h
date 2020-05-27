 #ifndef CL_INFOH503_INTEGRAL_IMAGE_H
#define CL_INFOH503_INTEGRAL_IMAGE_H

struct Opencl_buffer;
struct Opencl_stuff;

struct ScanParameters{
    /** Blocs number is defined by the number of pixels on the padded image divided by the local work size **/
    int number_blocs;
    /** Scan depth is the number of step needed to scan a bloc of thread. it is equal to log2(number of pixels)**/
    int scan_depth;

    /**
     * Setup parameters using GPU and Image data.
     * @param image Opencv matrix on which the Scan algorithm will be applied
     * @param ocl_stuff
     */
    ScanParameters(const Opencl_buffer &image, const Opencl_stuff &ocl_stuff);

};

void compute_integral_image(const Opencl_buffer &image, const Opencl_stuff &ocl_stuff);


#endif //CL_INFOH503_INTEGRAL_IMAGE_H
