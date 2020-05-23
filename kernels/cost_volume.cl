
kernel void cost_volume_in_range(global unsigned char *start_image, global unsigned char *end_image, global float *output_cost, int padding_size, float weight, float t1, float t2, int disparity_sign){
    // Disparity start at 0
    const int disparity = get_global_id(2);
    // Index for the input image
    // - thread info
    const int in_col = get_global_id(0);
    const int in_row = get_global_id(1);
    const int in_row_size = get_global_size(0);
    const int in_col_size = get_global_size(1);
    // - left
    const int index_start = ( in_col + in_row * in_row_size );
    // - right
    const int index_max_row =  (in_row +1) * in_row_size;
    const int index_min_row =  in_row  * in_row_size -1;
    int candidate_index_end = index_start + disparity_sign * disparity;
    int index_end;
    index_end = ( candidate_index_end > 0)? candidate_index_end : 0;
    index_end = ( candidate_index_end > index_max_row)? index_max_row : candidate_index_end;

    // Index for output
    const int out_col = get_global_id(0) + padding_size;
    const int out_row = get_global_id(1) + padding_size;
    const int out_row_size = get_global_size(0) + 2* padding_size;
    const int out_col_size = get_global_size(1) + 2* padding_size;
    const int output_disparity_offset = out_row_size* out_col_size * get_global_id(2);
    const int output_index = out_col +  out_row * out_row_size + output_disparity_offset ;

    int color_difference =  abs(start_image[index_start] - end_image[index_end]);
    color_difference = (color_difference < t1) ? color_difference : (int)t1;


    float gradient_start = (float)(start_image[index_start + 1] - start_image[index_start - 1]) / 2.0;
    float gradient_end = (float)(end_image[index_end + 1] - end_image[index_end - 1]) / 2.0;
    float gradient_difference = fabs( gradient_start - gradient_end);
    gradient_difference = (gradient_difference < t2) ? gradient_difference : t2;

    float cost = (float)color_difference / t1 + weight * gradient_difference / t2;


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