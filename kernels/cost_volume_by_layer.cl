kernel void memset(   global unsigned char *input_images, global float *output_cost, int padding_size, int disparity, float weight, float t1, float t2){
    // A thread per pixel of the right image, hence global_size do not padding size
    const int x = get_global_id(0) + padding_size;
    const int y = get_global_id(1) + padding_size;
    const int z = get_global_id(2);
    const int size_x = get_global_size(0);
    const int size_y = get_global_size(1);
    const int number_col = size_x + 2*padding_size;
    const int index_left = 2*(x + y * number_col);
    const int index_right = index_left  +1 +z*2;
    const int output_index = get_global_id(0) + get_global_id(1) * size_x + z*size_x*size_y;
    // Accounting for entrelaced image

    float color_difference = abs( input_images[index_left] - input_images[index_right]);
    color_difference = (color_difference < t1) ? color_difference : t1;

    float gradient_left = (float)(input_images[index_left + 2] - input_images[index_left - 2]) / 2;
    float gradient_right = (float)(input_images[index_right + 2] - input_images[index_right - 2]) / 2;
    float gradient_difference = fabs( gradient_left - gradient_right);

    gradient_difference = (gradient_difference < t2) ? gradient_difference : t2;
    float cost = (1 - weight)  * color_difference + weight * gradient_difference;
    output_cost[output_index] = 40*cost;
}