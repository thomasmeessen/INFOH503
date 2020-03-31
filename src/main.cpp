#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/cl2.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

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

    // 5. Create a data buffer.
    std::vector<int> output(numElements, 0);
    cl::Buffer outputBuffer(opencl_context, begin(output), end(output), false);

    // 6. Launch the kernel. Let OpenCL pick the local work size.
    //size_t global_work_size = NWITEMS;
    kernel.setArg(0,outputBuffer);
    command_queue.enqueueNDRangeKernel(kernel,cl::NullRange, cl::NDRange(numElements), cl::NullRange);
    command_queue.finish();

    // 7. Look at the results via synchronous buffer map.
    cl::copy(outputBuffer, begin(output), end(output));
    //void* hostPtr = command_queue.enqueueMapBuffer(outputBuffer, CL_TRUE, CL_MAP_READ, 0, output.size(), NULL, NULL  );

    //if(hostPtr == 0) throw std::runtime_error("ERROR - NULL host pointer");
    for(auto && value: output ){
        cout << value << endl;
    }

     //*/
   return 0;
}
