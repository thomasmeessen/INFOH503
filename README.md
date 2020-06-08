# INFOH503
Course project for ULB INFOH503 - Introduction to GPU Programming

This repository is te fruit of the work of 4 master student from the polytechnic of the ULB.
It contain a crude implementation of a algorithm computing a Stereo Disparity trough Cost Aggregation with Guided Filter described in Tan and Monasse in 2014.
This project was developed in the particular setup of the covid-19 pandemic and suffered many setbacks caused by diversity in the team IT background and by a stiff learning curved caused by OpenCL and it's multi-platform deployment.
At the project's end the team achieved intermediate skills with OpenCL, a working implementation of the Stereo Disparity and had a close look at the computation of an integral image.

## Setup 

Dependencies of the projects are OpenCV and Opencl 1.2
The project is build using cmake;




## Stereo Disparity

## Integral Image
  
## SetBacks & Trivia

In no specific order the following events were cause for significant delay:

- Setup Opencv & OpenCL on windows
- Backtracking from learning the C++ Opencl interface due to hardware incompatibility
- Inverting left and right Stereo Image
- Kernel does not launch but error code is not mapped in documentation
- Writing a function that print the kernel compilation errors
- Difficulties in knowledge and code-related information spreading due to lockdown leading to several avoidable and hard to spot bugs
- Working only with integrated GPU necessitating added complexity due to small block size (256) 
- Code behavior variation between platforms making bug harder to spot.

Some important notes:

- Commits history displayed in the repository do not reflect each authors participation due to unequal git and C++ prior experience.
- It is best if a linux distribution is used to compile the code, it can be tricky on windows to setup the dependencies.


## Resources

- [C++ wrappers reference](https://github.khronos.org/OpenCL-CLHPP/)

- [Quick reference guide](https://www.khronos.org/registry/OpenCL/sdk/2.1/docs/OpenCL-2.1-refcard.pdf)

- [A programming guide](https://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/Opencl-programming-guide.html#programming-model)


