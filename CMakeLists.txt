cmake_minimum_required(VERSION 3.16)

# Build type
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'Debug' as none was specified.")
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif ()

project(CL_INFOH503)

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 14)

message("${OpenCL_INCLUDE_DIRS}")

add_executable(${PROJECT_NAME}  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/src/ocl_wrapper.cpp )
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCL_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${OpenCV_LIBS}
)


# Copying files containing OpenCL kernels
add_custom_target(
        update_kernels ALL
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/update_kernels.cmake ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Copying new version of kernel files..."
)
add_dependencies(${PROJECT_NAME} update_kernels)

# Copying image to build folder
FILE(GLOB TEST_IMAGES "${CMAKE_CURRENT_SOURCE_DIR}/resources/*.png" )
if(TEST_IMAGES)
    foreach(TEST_IMAGE ${TEST_IMAGES})
        file(COPY ${TEST_IMAGE}  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    endforeach()
endif()