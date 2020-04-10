kernel void memset(__global unsigned char* src, __global unsigned char* dst, __global unsigned char* cost) {
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

    for (int i = -radius; i <= radius; i++) {
        if (x + i >= 0 && x + i < get_global_size(0)) {
            for (int j = -radius; j <= radius; j++) {
                if (y + j >= 0 && y + j < get_global_size(1)) {
                    const id = ((y + j) * get_global_size(0)) + (x + i);
                    good_pixels++;
                    src_image_pixels_sum += src[id];
                    src_image_pixels_sum_square += (src[id] * src[id]);
                    p_k_sum += cost[id];
                    p_pixels_product += src[id] * cost[id];
                    //printf("%d \n", id);
                    dst[id] = src[id];
                }
            }
        }
    }
    if (good_pixels == 9) {
        printf("%d \n", good_pixels);
    }

    int mu_k = (1 / omega_size) * src_image_pixels_sum;
    int sigma_k = (1 / omega_size) * src_image_pixels_sum_square - (mu_k * mu_k);
    int p_k = (1 / omega_size) * p_k_sum;

    int a_k = ((1 / omega_size) * p_pixels_product - mu_k * p_k) / ((sigma_k * sigma_k) + epsilon);
    int b_k = p_k - a_k * mu_k;

}