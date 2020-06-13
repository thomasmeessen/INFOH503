#include "benchmark_integral_image.hpp"
#include "iostream"
#include <fstream>
#include <chrono>

using namespace std;

void run_integral_image_benchmark(){
    cout << "Running Benchmark for integral image." <<endl;
    cv::Mat results = cv::Mat::zeros(3, MAX_FACTOR, CV_32FC1);
    for (int i = 1; i < MAX_FACTOR; i++){
        // - Setup
        int mat_size = i * BASE_SIZE;
        cv::Mat image_to_integrate = cv::Mat::ones(mat_size, mat_size, CV_32FC1);

        // - CPU process
        auto start_cpu = std::chrono::steady_clock::now();

        auto end_cpu = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds_cpu = end_cpu - start_cpu;

        // - GPU process
        auto start_gpu = std::chrono::steady_clock::now();

        auto end_gpu = std::chrono::steady_clock::now();

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