kernel void memset(__global unsigned char* src, __global float* dst_a_k, __global float* dst_b_k, __global float* cost, int original_width, int original_height, int padding_size) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    const int padded_image_size = (get_global_size(0)+2*padding_size)*(get_global_size(1)+2*padding_size);

    int radius = 2;
    float omega_size = 25;
    int src_image_pixels_sum = 0;
    int src_image_pixels_sum_square = 0;
    float p_k_sum = 0;
    float p_pixels_product = 0;
    float epsilon = (255*255)/10000;
    int new_y = y + padding_size;
    int new_x = x + padding_size;
    int central_pixel = new_x + new_y * (original_width + 2 * padding_size) + z*padded_image_size;
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            const int src_id = ((new_y + j) * (original_width + 2 * padding_size)) + (new_x + i);
            const int cost_id = ((new_y + j) * (original_width + 2 * padding_size)) + (new_x + i)+ z*padded_image_size;
            int source_pixel = src[src_id];
            float cost_pixel = cost[cost_id];
            src_image_pixels_sum += source_pixel;
            src_image_pixels_sum_square += (source_pixel * source_pixel);
            p_k_sum += cost_pixel;
            p_pixels_product += (float) source_pixel * cost_pixel;
            //printf("%f \n", cost[cost_id]);
        }
    }

    float mu_k = src_image_pixels_sum / omega_size;
    float sigma_k_square = (src_image_pixels_sum_square / omega_size) - (mu_k * mu_k);
    float p_k = p_k_sum / omega_size;

    float a_k = ((p_pixels_product / omega_size) - mu_k * p_k) / (sigma_k_square + epsilon);
    float b_k = p_k - a_k * mu_k;

    dst_a_k[central_pixel] = a_k;
    dst_b_k[central_pixel] = b_k;

}