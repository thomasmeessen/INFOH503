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


find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 14)

message("${OpenCL_INCLUDE_DIRS}")

add_executable(${PROJECT_NAME}  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCL_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)


# Copying files containing OpenCL kernels
FILE(GLOB OPENCL_FILES  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cl" )
if(OPENCL_FILES)
  foreach(OPENCL_FILE ${OPENCL_FILES})
    file(COPY ${OPENCL_FILE}  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  endforeach()
endif()
