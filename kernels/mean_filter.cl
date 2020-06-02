kernel void mean_filter(__global float* src, __global float* dst_mean, int original_width, int original_height, int padding_size) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    const int padded_image_size = (get_global_size(0) + 2 * padding_size) * (get_global_size(1) + 2 * padding_size);
    int a[25];
    int b = 0;
    int n, i, j, position, swap;
    n = 25;
    int median = 13;

    int radius = 2;
    float omega_size = 25.0;
    int src_image_pixels_sum = 0;
    int src_image_pixels_sum_square = 0;
    float p_k_sum = 0;
    float p_pixels_product = 0;
    float epsilon = (255 * 255) / 10000;
    int new_y = y + padding_size;
    int new_x = x + padding_size;
    int central_pixel = new_x + new_y * (original_width + 2 * padding_size) + z * padded_image_size;
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            const int src_id = ((new_y + j) * (original_width + 2 * padding_size)) + (new_x + i);
            int source_pixel = src[src_id];
            src_image_pixels_sum += source_pixel;
            a[b] = source_pixel;
            b += 1;
        }
    }




    for (i = 0; i < n - 1; i++) {
        position = i;
        for (j = i + 1; j < n; j++) {
            if (a[position] > a[j])
                position = j;
        }
        if (position != i) {
            swap = a[i];
            a[i] = a[position];
            a[position] = swap;
        }
    }




   // float mu_k = src_image_pixels_sum / omega_size;
    //printf("%f \n", mu_k);

    dst_mean[central_pixel] = a[13];

}