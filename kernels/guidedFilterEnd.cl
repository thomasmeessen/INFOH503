kernel void memset(__global unsigned char* src, __global float* src_a_k, __global float* src_b_k, __global float* dst, int original_width, int original_height, int padding_size, int radius) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    const int padded_image_size = (get_global_size(0)+2*padding_size)*(get_global_size(1)+2*padding_size);

    float omega_size = (2 * radius + 1) * (2 * radius + 1);
    float a_k_sum = 0;
    float b_k_sum = 0;
    int new_y = y + padding_size;
    int new_x = x + padding_size;
    int central_pixel = new_x + new_y * (original_width + 2 * padding_size) + z*padded_image_size;

    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            const int id = ((new_y + j) * (original_width + 2 * padding_size)) + (new_x + i)+ z*padded_image_size;
            a_k_sum += src_a_k[id];
            b_k_sum += src_b_k[id];
        }
    }
    float a_k_sum_mean = a_k_sum / omega_size;
    float b_k_sum_mean = b_k_sum / omega_size;
    float source_pixel = (float) src[central_pixel- z*padded_image_size];
    dst[x+y*get_global_size(0) + z*get_global_size(0)*get_global_size(1)] = (a_k_sum_mean * source_pixel + b_k_sum_mean);
    

}