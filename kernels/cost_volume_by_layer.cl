kernel void memset(   global unsigned char *input_images, global float *output_cost, int padding_size, int disparity, float weight){
    float t1 = 7.;//tau 1
    float t2 = 2.;//tau 2 later add as parameter function
    // A thread per pixel of the right image, hence global_size do not padding size
    int x = get_global_id(0) + padding_size;
    int y = get_global_id(1) + padding_size;
    int number_col = get_global_size(0) + 2*padding_size;
    int index_left = 2* (x + y * number_col);
    int index_right =  2* (x + y * number_col + disparity) +1;
    int output_index = get_global_id(0) + get_global_id(1) * get_global_size(0);
    // Accounting for entrelaced image


    float color_difference = abs( input_images[index_left] - input_images[index_right]);
    color_difference = (color_difference > t1) ? t1 : color_difference  ;

    float gradient_left = (float)(input_images[index_left + 2] - input_images[index_left - 2]) / 2;
    float gradient_right = (float)(input_images[index_right + 2] - input_images[index_right - 2]) / 2;
    float gradient_difference = fabs( gradient_left - gradient_right);

    gradient_difference = (gradient_difference > t2) ? t2 : gradient_difference;
    float cost = (1 - weight)  * color_difference + weight * gradient_difference;
    output_cost[output_index] = cost;
}