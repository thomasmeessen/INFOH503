kernel void memset(   global unsigned char *input_images, global float *output_cost, int padding_size, int disparity, float weight, float t1, float t2){
    // A thread per pixel of the right image, hence global_size do not padding size
    const int kernel_x = get_global_id(0);
    const int kernel_y = get_global_id(1);
    const int x = kernel_x + padding_size;
    const int y = kernel_y + padding_size;
    const int z = get_global_id(2);
    const int kernel_size_x = get_global_size(0);
    const int kernel_size_y = get_global_size(1);
    const int size_x = kernel_size_x +2*padding_size;
    const int size_y = kernel_size_y +2*padding_size;
    const int index_left = 2*(x + y * size_x);
    const int index_right = index_left  +1 +z*2;

    const int output_index = x + y* size_x + z*size_x*size_y ; //(z+1)*disparity*size_x;
    
    // Accounting for entrelaced image
    float color_difference = abs( input_images[index_left] - input_images[index_right]);
    color_difference = (color_difference < t1) ? color_difference : t1;

    float gradient_left = (float)(input_images[index_left + 2] - input_images[index_left - 2]) / 2;
    float gradient_right = (float)(input_images[index_right + 2] - input_images[index_right - 2]) / 2;
    float gradient_difference = fabs( gradient_left - gradient_right);

    gradient_difference = (gradient_difference < t2) ? gradient_difference : t2;
    float cost = (1 - weight)  * color_difference + weight * gradient_difference;
    cost = 40*cost;


    output_cost[output_index] = cost;

    //dégueulasse mais jsut for the time being
    //padding
    if(kernel_x==0){//first column, every pixel on the left has the same color
        for(int i=1; i <= padding_size; i++){
            output_cost[output_index - i] = cost;
        }
    }
    else if(kernel_x==(kernel_size_x-1)){//last column, every pixel on the left has the same color
        for(int i=1; i <= padding_size; i++){
            output_cost[output_index + i] = cost;
        }
    }
    if(kernel_y==0){//first line we copy the color to every pixel above this one
        for(int i=1; i <= padding_size; i++){
            output_cost[output_index - i*size_x] = cost;
        }
    }
    else if(kernel_y==(kernel_size_y-1)){//last line we copy the color to every pixel below this one
        for(int i=1; i <= padding_size; i++){
            output_cost[output_index + i*size_x] = cost;
        }
    }
}