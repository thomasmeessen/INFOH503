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
const char *source =
        "kernel void memset(   global uint *dst )             \n"
        "{                                                    \n"
        "    dst[get_global_id(0)] = get_global_id(0);        \n"
        "}                                                    \n";


int main(void) { // (int argc, const char * argv[]) {

  // Get Platform ID tuto
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
  Program cl_program( opencl_context, source, NULL);
  cl_program.build(vector_devices, NULL, NULL, NULL);
  Kernel kernel(cl_program, "memset", NULL);

    // 5. Create a data buffer.
    std::vector<int> output(numElements, 0xdeadbeef);
    cl::Buffer outputBuffer(begin(output), end(output), false);

    // 6. Launch the kernel. Let OpenCL pick the local work size.
    size_t global_work_size = NWITEMS;
    kernel.setArg(0,sizeof(outputBuffer), (void*) &outputBuffer);

   return 0;
}
