#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/cl2.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>

#define NWITEMS 512
const int numElements = 32;


using namespace std;
using namespace cl;

// A simple memset kernel
const string source_path = "threshold.cl";



int main(void) { // (int argc, const char * argv[]) {


    std::vector<Platform> platforms;
    Platform::get(&platforms);
    Platform platform;
    for (auto &p : platforms) {
    std::string platver = p.getInfo<CL_PLATFORM_VERSION>();
    if (platver.find("OpenCL 2.") != std::string::npos) {
        platform = p;
    }
    }
    if (platform() == 0)  {
        std::cout << "No OpenCL 2.0 platform found.";
        return -1;
    }

    cout << "Platform Name: " << (string)platform.getInfo<CL_PLATFORM_NAME>() << " version: "<< (string)platform.getInfo<CL_PLATFORM_VERSION>() << endl;

    Platform newP = cl::Platform::setDefault(platform);

    if (newP != platform) {
      cout << "Error setting default platform.";
      return -1;
    }

    // Use the wrapper getDeviceId
    std::vector<Device> vector_devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &vector_devices);
    for(auto &&device : vector_devices){
        std::string device_name = (string)device.getInfo<CL_DEVICE_NAME> ();
        cout << device_name << endl;
    }

    // 3. Create a context and command queue on that device.
    Context opencl_context(vector_devices, NULL, NULL, NULL);
    CommandQueue command_queue(opencl_context, vector_devices[0], 0, NULL);

    // 4. Perform runtime source compilation, and obtain kernel entry point.
    std::ifstream source_file(source_path);
    std::string source_code(std::istreambuf_iterator<char>(source_file), (std::istreambuf_iterator<char>()));
    Program cl_program( opencl_context, source_code);
    cl_program.build(vector_devices, NULL, NULL, NULL);
    Kernel kernel(cl_program, "memset", NULL);

    // 5. Load an image into a buffer
    cv::Mat source_image = cv::imread("paper0.png", cv::IMREAD_GRAYSCALE);
    source_image.convertTo(source_image, CV_32S);
    cv::Mat cp_mat = source_image.clone();
    cv::imwrite("pre.png", source_image);
    size_t image_1D_size = source_image.cols * source_image.rows * sizeof(int);
    cl::Buffer outputBuffer(opencl_context, CL_MEM_COPY_HOST_PTR, image_1D_size, (void*)source_image.data, NULL );

    // 6. Launch the kernel. Let OpenCL pick the local work size.
    kernel.setArg(0,outputBuffer);
    command_queue.enqueueNDRangeKernel(kernel,cl::NullRange, cl::NDRange(source_image.cols, source_image.rows));
    command_queue.finish();

    // 7. Look at the results via synchronous buffer map.
    cl_int ret_code = CL_SUCCESS;
    //cv::MatIterator_<uint8_t>
    cl::Event event;

    source_image = 0;
    command_queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, image_1D_size, (void*)source_image.data);


    cout << source_image <<endl;
    cv::imwrite("post.png", source_image);
     //*/
   return 0;
}
