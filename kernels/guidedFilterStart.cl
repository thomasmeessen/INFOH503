kernel void memset(__global unsigned char* src, __global unsigned char* dst_a_k, __global unsigned char* dst_b_k, __global unsigned char* cost) {

    const int x = get_global_id(0);
    const int y = get_global_id(1);


    int sum_color_over_window = 0;
    int sum_color_over_window_square = 0;
    int sum_cost_over_window = 0;
    int sum_product_color_cost = 0;
    int epsilon = 0;

    int central_pixel = x + y * get_global_size(0);

    // For each pixel on a square window of specified radius

    // Describing the square window
    int radius = 2;
    // formulated window size with ifs
    int low_x = (x - radius > 0)? x-radius:0;
    int high_x = (x + radius > get_global_size(0)) ? get_global_size(0): x+radius;
    int low_y = (y - radius > 0)? y-radius:0;
    int high_y = (y + radius > get_global_size(1)) ? get_global_size(1): y+radius;

    int number_pixels_in_window = (high_x - low_x ) * (high_y - low_y) ;

    for (int i = low_y; i <= high_y; i++) {
        for (int j = low_x; j <= high_y; j++) {
            const int id = j + i * get_global_size(0);
            sum_color_over_window += src[id];
            sum_color_over_window_square += (src[id] * src[id]);
            sum_cost_over_window += cost[id];
            sum_product_color_cost += src[id] * cost[id];
        }
    }

    int average_color_in_window = sum_color_over_window / number_pixels_in_window;

    int sigma_k = (sum_color_over_window_square / number_pixels_in_window) - (average_color_in_window * average_color_in_window);
    int avg_cost_over_window = sum_cost_over_window / number_pixels_in_window;

    int a_k = ((sum_product_color_cost / number_pixels_in_window) - average_color_in_window * avg_cost_over_window) / ((sigma_k * sigma_k) + epsilon);
    int b_k = avg_cost_over_window - a_k * average_color_in_window;
    a_k = x;
    b_k = y;
    //printf("%d \n", a_k);
    dst_a_k[central_pixel] = 255;
    dst_b_k[central_pixel] = 255;

}