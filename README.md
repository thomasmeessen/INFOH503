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

Their result on the paper image is :
|   |   |
|---|---|
| filter_cost_volume 1 time  |  0.337987 | 
| filter_cost_volume 2 time |  0.313787 | 
| occlusion detection time|  0.000444775|
| occlusion filling time| 0.106078|
**maybe try on other small images?**
### 1.2 speed Up we could expect

- biggest speed up we should expect would be on the first and biggest operation/loop /more heavy loaded : generating all the layers and using the guided filter on them. . In their implementation it is done in 
 for(|disparity|){ for every pixel} where as us we only do for(disparity) and then every pixel is parallelizedd and this the biggest cost is the memory transfer. 
 so here the bottleneck is really the memory transfer. we could expect a speedp for that part of the disparity size in a perfect world so at most disparity size. let's say taht for small data data transfer is a big drawback so let's say half the size of disparity. 
 and we also use an integral image to compute the sums for the a_k and b_k. which adds the load of computing the integral image (and thus memory back and forth) but reduce by a factor **add factor**  the first step of our first step of the filtering.
 - left right consistency : probably not much speed up
 - occlusion detection : could expect a small speed up. they do *for every pixel for every disparity* where as we parallelize for every pixel and every thread check for every disparity. once again memory  => bottleneck
 - on occlusion filling it would be nice to have a little bit of speed up against their openmp implementation but we can't expect much. they build an historigram and stuff.

 so we can expect large speed up + small speed ups.
 so for a disparity 16 we get a 8x and for the small speed ups between 2 and 4 (really depends on data transfer).
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


