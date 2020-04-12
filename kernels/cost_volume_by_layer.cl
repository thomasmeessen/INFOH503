kernel void memset(   global unsigned char *input_images, global float *output_cost, int padding_size, int disparity, float weight){

    // A thread per pixel of the right image, hence global_size do not padding size
    int x = get_global_id(0) + padding_size;
    int y = get_global_id(1) + padding_size;
    int number_col = get_global_size(0) + 2*padding_size;
    int index_left = x + y * number_col;
    int index_right = index_left + disparity*2 +1;
    int output_index = get_global_id(0) + get_global_id(1) * get_global_size(0);
    // Accounting for entrelaced image
    index_left = 2*index_left;
    index_right = 2*index_right;

    float color_difference = abs( input_images[index_left] - input_images[index_right]);
    float gradient = abs( input_images[index_left - 2] - input_images[index_left +2]) / 2;
    float cost = (1 - weight)  * color_difference + weight * gradient;
    output_cost[output_index] = cost;
}
