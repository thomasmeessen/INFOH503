
# ${CMAKE_ARGV3} is the source path from the caller
FILE(GLOB OPENCL_FILES  "${CMAKE_ARGV3}/kernels/*.cl" )
if(OPENCL_FILES)
    foreach(OPENCL_FILE ${OPENCL_FILES})
        file(COPY ${OPENCL_FILE}  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    endforeach()
endif()