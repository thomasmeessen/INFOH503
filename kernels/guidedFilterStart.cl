kernel void memset(__global unsigned char* src, __global float* dst_a_k, __global float* dst_b_k, __global float* cost, int original_width, int original_height, int max_dist) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    const int image_size = get_global_size(0)*get_global_size(1);
    const int padded_image_size = (get_global_size(0)+2*max_dist)*(get_global_size(1)+2*max_dist);
    //const id = (y * get_global_size(0)) + x;
    //printf("yes");

    int radius = 2;
    int omega_size = 25;
    int src_image_pixels_sum = 0;
    int src_image_pixels_sum_square = 0;
    float p_k_sum = 0;
    float p_pixels_product = 0;
    int epsilon = 0;
    int new_y = y + max_dist;
    int new_x = x + max_dist;
    int central_pixel = new_x + new_y * (original_width + 2 * max_dist) + z*padded_image_size;
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            const int src_id = ((new_y + j) * (original_width + 2 * max_dist)) + (new_x + i);
            const int cost_id = ((new_y + j) * (original_width + 2 * max_dist)) + (new_x + i)+ z*image_size;
            float source_pixel = src[src_id];
            float cost_pixel = cost[cost_id];
            src_image_pixels_sum += source_pixel;
            src_image_pixels_sum_square += (source_pixel * source_pixel);
            p_k_sum += cost_pixel;
            p_pixels_product += source_pixel *  cost_pixel;
            //printf("%f \n", cost[id]);
        }
    }

    float mu_k = src_image_pixels_sum / omega_size;
    float sigma_k = (src_image_pixels_sum_square / omega_size) - (mu_k * mu_k);
    float p_k = p_k_sum / omega_size;

    //int a_k = ((p_pixels_product / omega_size) - mu_k) / ((sigma_k * sigma_k) + epsilon);
    float a_k = ((p_pixels_product / omega_size) - mu_k * p_k) / ((sigma_k * sigma_k) + epsilon);
    float b_k = p_k - a_k * mu_k;

    dst_a_k[central_pixel] = a_k;
    dst_b_k[central_pixel] = b_k;

}