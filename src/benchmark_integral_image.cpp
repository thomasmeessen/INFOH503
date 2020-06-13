#include "benchmark_integral_image.hpp"
#include "integral_image.h"
#include "ocl_wrapper.h"
#include "iostream"
#include <fstream>
#include <chrono>

using namespace std;

void cpu_integral_image(cv::Mat m){
    // Limitation due to the usage of the .at operator in the loop
    assert(m.type() == CV_32FC1);
    cv::Mat ii = cv::Mat::zeros(m.size(), m.type());
    for( int row = 1; row < m.rows ; row++){
        float s =0;
        for (int col = 0; col < m.cols - 1 ; col ++){
            s += m.at<float>(row-1, col);
            ii.at<float>(row, col+1) = s + ii.at<float>(row-1, col+1);
        }
    }
}


void run_integral_image_benchmark(Opencl_stuff ocl_stuff) {
    cout << "Running Benchmark for integral image." <<endl;
    cv::Mat results = cv::Mat::zeros(3, MAX_FACTOR, CV_32FC1);
    for (int i = 1; i < MAX_FACTOR; i++){
        // - Setup
        int mat_size = i * BASE_SIZE;
        cv::Mat image_to_integrate = cv::Mat::ones(mat_size, mat_size, CV_32FC1);

        // - CPU process
        auto start_cpu = std::chrono::steady_clock::now();
        cpu_integral_image(image_to_integrate);
        auto end_cpu = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds_cpu = end_cpu - start_cpu;

        // - Load to gpu
        Opencl_buffer gpu_loaded_image_to_integrate(image_to_integrate, ocl_stuff, 0);
        // - GPU process
        auto start_gpu = std::chrono::steady_clock::now();
        compute_integral_image(gpu_loaded_image_to_integrate, ocl_stuff);
        auto end_gpu = std::chrono::steady_clock::now();
        gpu_loaded_image_to_integrate.free();
        assert(Opencl_buffer::used_memory == 0);
        // - Storing results
        std::chrono::duration<double> elapsed_seconds_gpu = end_gpu - start_gpu;
        auto cpu_time = elapsed_seconds_cpu.count();
        auto gpu_time = elapsed_seconds_gpu.count();
        cout << " - Size " << mat_size << "x" << mat_size << " time cpu: " << cpu_time
        <<" gpu: " << gpu_time << endl;
        results.at<float>(0,i) = mat_size;
        results.at<float>(1,i) = cpu_time;
        results.at<float>(2,i) = gpu_time;
    }
    // - Writing rsults to a csv;
    ofstream out_file;
    out_file.open("ii_benchmark.csv");
    out_file<< cv::format(results, cv::Formatter::FMT_CSV) << std::endl;
    out_file.close();
    cout << "End Benchmark for Integral image" << endl;
    cout << "=======================" << endl;
}