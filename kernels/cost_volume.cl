
kernel void cost_volume_in_range(global unsigned char *left_image, global unsigned char *right_image, global float *output_cost, int padding_size, float weight, float t1, float t2){
    // Disparity start at 1 and the id at 0
    const int disparity = get_global_id(2) +1;
    // Index for the input image
    // - thread info
    const int in_col = get_global_id(0);
    const int in_row = get_global_id(1);
    const int in_row_size = get_global_size(0);
    const int in_col_size = get_global_size(1);
    // - left
    const int index_left = ( in_col + in_row * in_row_size );
    // - right
    int index_max_row =  (in_row +1) * in_row_size;
    int candidate_index_right = index_left + disparity;
    const int index_right = ( candidate_index_right < index_max_row)? candidate_index_right : index_max_row - 1;

    // Index for output
    const int out_col = get_global_id(0) + padding_size;
    const int out_row = get_global_id(1) + padding_size;
    const int out_row_size = get_global_size(0) + 2* padding_size;
    const int out_col_size = get_global_size(1) + 2* padding_size;
    const int output_disparity_offset = out_row_size* out_col_size * get_global_id(2);
    const int output_index = out_col +  out_row * out_row_size + output_disparity_offset ;

    int color_difference =  abs(left_image[index_left] - right_image[index_right]);
    color_difference = (color_difference < t1) ? color_difference : (int)t1;


    float gradient_left = (float)(left_image[index_left + 1] - left_image[index_left - 1]) / 2.0;
    float gradient_right = (float)(right_image[index_right + 1] - right_image[index_right - 1]) / 2.0;
    float gradient_difference = fabs( gradient_left - gradient_right);
    gradient_difference = (gradient_difference < t2) ? gradient_difference : t2;

    float cost = (1 - weight)  * (float)color_difference / t1 + weight * gradient_difference / t2;


    output_cost[output_index] = cost;

    //padding
    if(in_col==0){//first column, every pixel on the left has the same color
        for(int i=1; i <= padding_size; i++){
            output_cost[output_index - i] = cost;
        }
    }
    else if(in_col==(in_row_size-1)){//last column, every pixel on the right has the same color
        for(int i=1; i <= padding_size; i++){
            output_cost[output_index + i] = cost;
        }
    }
    if(in_row==0){//first line we copy the color to every pixel above this one
        for(int i=1; i <= padding_size; i++){
            output_cost[output_index - i*out_row_size] = cost;
        }
    }
    else if(in_row==(in_col_size-1)){//last line we copy the color to every pixel below this one
        for(int i=1; i <= padding_size; i++){
            output_cost[output_index + i*out_row_size] = cost;
        }
    }
}