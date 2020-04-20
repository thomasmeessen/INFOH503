kernel void memset(__global unsigned char* src, __global unsigned char* dst_a_k, __global unsigned char* dst_b_k, __global unsigned char* cost, int original_width, int original_height, int max_dist) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    //const id = (y * get_global_size(0)) + x;

    int radius = 2;
    int omega_size = 25;
    int good_pixels = 0;
    int src_image_pixels_sum = 0;
    int src_image_pixels_sum_square = 0;
    int p_k_sum = 0;
    int p_pixels_product = 0;
    int epsilon = 0;
    int new_y = y + max_dist;
    int new_x = x + max_dist;
    int central_pixel = new_x + new_y * (original_width + 2 * max_dist);

    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            const int id = ((new_y + j) * (original_width + 2 * max_dist)) + (new_x + i);
            good_pixels++;
            src_image_pixels_sum += src[id];
            src_image_pixels_sum_square += (src[id] * src[id]);
            p_k_sum += cost[id];
            p_pixels_product += src[id] * cost[id];
        }
    }

    int mu_k = src_image_pixels_sum / omega_size;
    int sigma_k = (src_image_pixels_sum_square / omega_size) - (mu_k * mu_k);
    int p_k = p_k_sum / omega_size;

    //int a_k = ((p_pixels_product / omega_size) - mu_k) / ((sigma_k * sigma_k) + epsilon);
    int a_k = ((p_pixels_product / omega_size) - mu_k * p_k) / ((sigma_k * sigma_k) + epsilon);
    int b_k = p_k - a_k * mu_k;
    //printf("%d \n", a_k);
    dst_a_k[central_pixel] = a_k;
    dst_b_k[central_pixel] = b_k;

}