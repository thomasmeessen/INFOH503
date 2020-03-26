#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/cl2.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

const int numElements = 32;


using namespace std;
using namespace cl;

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
    std::string device_name;
    //device.getInfo(CL_DEVICE_NAME, device_name);
    //cout << device_name << endl;
  }




   return 0;
}
