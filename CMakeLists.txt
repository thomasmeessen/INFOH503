cmake_minimum_required(VERSION 3.16)

# Build type
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'Debug' as none was specified.")
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif ()

project(CL_hello)

find_package(OpenCL REQUIRED)
include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 14)

message("${OpenCL_INCLUDE_DIRS}")

add_executable(${PROJECT_NAME}  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCL_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)


# Copying files with OpenCL kernels
#set(OPENCL_FILES
#    kernels/fft1D_1024_kernel_src.cl
#)
#if(OPENCL_FILES)
#  foreach(OPENCL_FILE ${OPENCL_FILES})
#    configure_file(${OPENCL_FILE} ${OPENCL_FILE} COPYONLY)
#  endforeach()
#endif()
