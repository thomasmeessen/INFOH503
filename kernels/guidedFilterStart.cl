kernel void memset(__global unsigned char* src, __global float* dst_a_k, __global float* dst_b_k, __global unsigned char* cost) {

    const int x = get_global_id(0);
    const int y = get_global_id(1);


    int sum_color_over_window = 0;
    int sum_color_over_window_square = 0;
    int sum_cost_over_window = 0;
    int sum_product_color_cost = 0;
    int epsilon = 0.001;

    int central_pixel = x + y * get_global_size(0);


    // For each pixel on a square window of specified radius

    // Describing the square window
    int radius = 2;
    // formulated window size with ifs
    int low_x = (x - radius > 0)? x-radius:0;
    int high_x = (x + radius > get_global_size(0)) ? get_global_size(0): x+radius;
    int low_y = (y - radius > 0)? y-radius:0;
    int high_y = (y + radius > get_global_size(1)) ? get_global_size(1): y+radius;

    // The window include the extremum value of x and y so add + 1 the size of interval [3-2] is 2 not 1
    int number_pixels_in_window = (high_x - low_x +1) * (high_y - low_y +1) ;



    for (int i = low_y; i <= high_y; i++) {
        if ( x == 370 && y == 20){
           // printf("for loop: %d \n", i);
            }
        for (int j = low_x; j <= high_x; j++) {

            const int id = j + i * get_global_size(0);
            if ( x == 370 && y == 20){
                                //printf(" image:  %d ", src[id]);
                               // printf(" cost:  %d \n", cost[id]);
                                }
            sum_color_over_window += src[id];
            sum_color_over_window_square += (src[id] * src[id]);
            sum_cost_over_window += cost[id];
            sum_product_color_cost += src[id] * cost[id];
        }
    }


    int average_color_in_window = sum_color_over_window / number_pixels_in_window;

    int sigma_k_square = (sum_color_over_window_square / number_pixels_in_window) - (average_color_in_window * average_color_in_window);
    int avg_cost_over_window = sum_cost_over_window / number_pixels_in_window;

    int a_k = ((sum_product_color_cost / number_pixels_in_window) - average_color_in_window * avg_cost_over_window) / (sigma_k_square  + epsilon);
    int b_k = avg_cost_over_window - a_k * average_color_in_window;


    if ( x == 370 && y == 20){
        //printf("debug: \n");
        //("a_k %d \n" , a_k);
        //printf("b_k %d \n" , b_k);
        //printf("number_pixels_in_window %d \n" , number_pixels_in_window);
        //printf("average_color_in_window %d \n" , average_color_in_window);
        //printf("sigma_k %d \n" , sigma_k_square);
        //printf("avg_cost_over_window %d \n" , avg_cost_over_window);
        //printf("sum_product_color_cost %d \n" , sum_product_color_cost);
    }

    dst_a_k[central_pixel] = a_k;
    dst_b_k[central_pixel] = b_k;

}