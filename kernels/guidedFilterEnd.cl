kernel void memset(__global unsigned char* src, __global unsigned char* src_a_k, __global unsigned char* src_b_k, __global unsigned char* dst, int original_width, int original_height, int max_dist) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    int radius = 2;
    int omega_size = 25;
    int a_k_sum = 0;
    int b_k_sum = 0;
    int new_y = y + max_dist;
    int new_x = x + max_dist;
    int central_pixel = new_x + new_y * (original_width + 2 * max_dist);

    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            const int id = ((new_y + j) * (original_width + 2 * max_dist)) + (new_x + i);
            a_k_sum += src_a_k[id];
            b_k_sum += src_b_k[id];

        }
    }
   // printf("%d \n", a_k_sum);
    int a_k_sum_mean = a_k_sum / omega_size;
    int b_k_sum_mean = b_k_sum / omega_size;
    dst[central_pixel] = a_k_sum_mean * src[central_pixel] + b_k_sum_mean;

}