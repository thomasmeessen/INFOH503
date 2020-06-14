kernel void matrix_mult(__global float* matrix1, __global float* matrix2, __global float* output) {

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int width = get_global_size(0);
    const int pixel = x + y * width;

    output[pixel] = matrix1[pixel] * matrix2[pixel];


}