# INFOH503
Course project for ULB INFOH503 - Introduction to GPU Programming

This repository is te fruit of the work of 4 master student from the polytechnic of the ULB.
It contain a crude implementation of a algorithm computing a Stereo Disparity trough Cost Aggregation with Guided Filter described in Tan and Monasse in 2014.
This project was developed in the particular setup of the covid-19 pandemic and suffered many setbacks caused by diversity in the team IT background and by a stiff learning curved caused by OpenCL and it's multi-platform deployment.
At the project's end the team achieved intermediate skills with OpenCL, a working implementation of the Stereo Disparity and had a close look at the computation of an integral image.

## Setup 

Dependencies of the projects are OpenCV and Opencl 1.2
The project is build using cmake;




## 1. Stereo Disparity
### 1.1 Paper implementation
First of all it is important to look at their implementation in order to speculate on the kind of acceleration we could expect. And when looking at it we of course find a well optimized implementation.

Notably:
- Filtering each layer as soon as it is generated.
- Once filtered  we update the disparity of each pixel for the final depth map;
- OpenMP is used for parallelization of occlusion filling.

But compared to our guided filter theirs use 3 channels. Where we only use 1 as we use a grayscale image. So we should be doing 2\*x\*y\*disparity less operations.

**maybe try on other small images?**
### 1.2 speed Up we could expect
We can divide their implementation in 4 parts :

|   |   |
|---|---|
| LR depth map generation  |  0.337987 s | 
| RL depth map generation |  0.313787 s | 
| occlusion detection|  0.000444775 s|
| occlusion filling| 0.106078 s|

And Obviously we had different expectation regarding the different speed up we could achieve.

1.  Depth map
    
    The Depth map is computed in 4 step :
    
    1. Computing the cost of every pixel for each disparity

    2. Filtering each disparity layer

    3. Selecting the best disparity for every pixel

    For each of those steps we can expect a speedup by working on all the layers and all the pixels of each layer at the same time. Compared to their implementation wheere everything is accessed sequentially. The bottleneck of our implementation should be the memory transfer.

2. Occlusion detection
      We should see a speed up for big images but on small images we should not get one.

3. occlusion filling
    As they rely on openmp to do the filling we can't really expect that much of a speedup against their implementation;




### 1.3 actual speed up
Add data of transfer time on our different computers to be able to estimate how much time we're losing on data transfer only on average. 
**add actual speed ups we got I can add timer to their code at the designated areas and we need to remove every write image and terminal print and useless ifs beofre measuring**

## 2. Integral Image
  
## 3. SetBacks & Trivia

In no specific order the following events were cause for significant delay:

- Setup Opencv & OpenCL on windows
- Backtracking from learning the C++ Opencl interface due to hardware incompatibility
- Inverting left and right Stereo Image
- Kernel does not launch but error code is not mapped in documentation
- Writing a function that print the kernel compilation errors
- Difficulties in knowledge and code-related information spreading due to lockdown leading to several avoidable and hard to spot bugs
- Working only with integrated GPU necessitating added complexity due to small block size (256) 
- Code behavior variation between platforms making bug harder to spot.
- Memory leaks


Some important notes:

- Commits history displayed in the repository do not reflect each authors participation due to unequal git and C++ prior experience.
- It is best if a linux distribution is used to compile the code, it can be tricky on windows to setup the dependencies.



## Resources

- [C++ wrappers reference](https://github.khronos.org/OpenCL-CLHPP/)

- [Quick reference guide](https://www.khronos.org/registry/OpenCL/sdk/2.1/docs/OpenCL-2.1-refcard.pdf)

- [A programming guide](https://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/Opencl-programming-guide.html#programming-model)


