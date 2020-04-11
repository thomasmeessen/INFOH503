kernel void memset(   global unsigned char *left_src, global unsigned char *right_src, global float *dest, int padding_size, int disparity, float weight){
    // A thread per pixel of the right image, hence global_size do not padding size
    int x = get_global_id(0) + padding_size;
    int y = get_global_id(1) + padding_size;

    int number_col = get_global_size(0);
    // int number_rows = get_global_size(1);

    int index_left = x + y * number_col;
    int index_right = index_left + disparity;

    float color_difference = abs( left_src[index_left] - right_src[index_right]);
    float gradient = abs( left_src[index_left - 1] - left_src [index_left +1]) / 2;
    float cost = (1 - weight)  * color_difference + weight * gradient;
    dest[index_right] = cost;
}
