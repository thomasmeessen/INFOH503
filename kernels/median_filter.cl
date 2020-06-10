kernel void median_filter(__global float* src, __global float* dst_mean, int padding_size) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int padded_image_size = (get_global_size(0) + 2 * padding_size) * (get_global_size(1) + 2 * padding_size);
    const int original_width = get_global_size(0);


    int radius = 2;
    int omega[25];
    int count = 0;
    int i, j, position, swap;
    int median = (radius+1)/2;

    int omega_size = (radius*2 + 1) * (radius * 2 + 1);
    int new_y = y + padding_size;
    int new_x = x + padding_size;
    int central_pixel = new_x + new_y * (original_width + 2 * padding_size);
    int pixel = x + y * original_width;
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            const int src_id = ((new_y + j) * (original_width + 2 * padding_size)) + (new_x + i);
            omega[count] = src[src_id];
            count += 1;
        }
    }

    for (i = 0; i < omega_size - 1; i++) {
        position = i;
        for (j = i + 1; j < omega_size; j++) {
            if (omega[position] > omega[j])
                position = j;
        }
        if (position != i) {
            swap = omega[i];
            omega[i] = omega[position];
            omega[position] = swap;
        }
    }

    if (pixel < get_global_size(0) * get_global_size(1) - 1) {
        dst_mean[pixel] = omega[median];
    }

}