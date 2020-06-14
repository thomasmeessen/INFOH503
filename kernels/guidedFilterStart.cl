//kernel void memset(__global unsigned char* src, __global float* dst_a_k, __global float* dst_b_k, __global float* cost, int original_width, int original_height, int padding_size, int radius, __global float* integral_image, __global float* integral_image_cost,
 //  __global float* integral_image_squared, __global float* sum_s_c) {
kernel void memset(__global unsigned char* src, __global float* dst_a_k, __global float* dst_b_k, __global float* cost, int original_width, int original_height, int padding_size, int radius) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    const int padded_image_size = (get_global_size(0)+2*padding_size)*(get_global_size(1)+2*padding_size);

    float omega_size = (2*radius + 1)*(2 * radius + 1);
    int src_image_pixels_sum = 0;
    int src_image_pixels_sum_square = 0;
    float p_k_sum = 0;
    float p_pixels_product = 0;
    float epsilon = (255*255)/10000;
    int new_y = y + padding_size;
    int new_x = x + padding_size;
    int central_pixel = new_x + new_y * (original_width + 2 * padding_size) + z*padded_image_size;

    /*//bottom right
    int p1 = ((new_y + radius) * (original_width + 2 * padding_size)) + (new_x + radius);   // jx, jy
    //bottom left
    int p2 = ((new_y + radius) * (original_width + 2 * padding_size)) + (new_x - radius);  //ix-1, jy
    //top right
    int p3 = ((new_y - radius) * (original_width + 2 * padding_size)) + (new_x + radius);  //jx, iy-1
    //top left
    int p4 = ((new_y - radius) * (original_width + 2 * padding_size)) + (new_x - radius);  //ix-1, iy-1*/





   /* int line_size = (original_width + 2 * padding_size);
    int ix = new_x - radius;
    int iy = new_y - radius;
    int jx = new_x + radius;
    int jy = new_y + radius;

    int p1 = jx + jy*line_size;
    int p2 = (ix-1) + jy*line_size;
    int p3 = jx + (iy-1)*line_size;
    int p4 = (ix-1) + (iy-1)*line_size;*/


 /*  float integral_image_sum = integral_image[p1] - integral_image[p2] - integral_image[p3] + integral_image[p4];

    /*if (x == 100 && y == 100 && z == 0) {
        printf("bottom right  %f %i\n", integral_image[p1], p1);
        printf("bottom left  %f %i\n", integral_image[p2], p2);
        printf("top right  %f %i\n", integral_image[p3], p3);
        printf("top left  %f %i\n", integral_image[p4], p4);
    }*/

    //float integral_image_cost_sum = integral_image_cost[p1] - integral_image_cost[p2] - integral_image_cost[p3] + integral_image_cost[p4];
//    float integral_image_squared_sum = integral_image_squared[p1] - integral_image_squared[p2] - integral_image_squared[p3] + integral_image_squared[p4];


  /*  //bottom right
     p1 = ((new_y + radius) * (original_width + 2 * padding_size)) + (new_x + radius) + z * padded_image_size;   // jx, jy
    //bottom left
    p2 = ((new_y + radius) * (original_width + 2 * padding_size)) + (new_x - radius) + z * padded_image_size;  //ix-1, jy
    //top right
    p3 = ((new_y - radius) * (original_width + 2 * padding_size)) + (new_x + radius) + z * padded_image_size;  //jx, iy-1
    //top left
    p4 = ((new_y - radius) * (original_width + 2 * padding_size)) + (new_x - radius) + z * padded_image_size;  //ix-1, iy-1
    float sum_src_cost = sum_s_c[p1] - sum_s_c[p2] - sum_s_c[p3] + sum_s_c[p4];*/
       

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
        }
    }

     float mu_k = src_image_pixels_sum / omega_size;
    // float mu_k = integral_image_sum / omega_size;

   /* if (x >= 383 && y>= 284 && (mu_k1 - mu_k) < 1) {
     //   printf("x %i \n", x);
  
        printf("y %i \n", y);
    }*/


   /* if (x == 100 && y == 100 && z == 0) {
        printf("mu_k  %f \n", mu_k);
        printf("mu_k1  %f \n", mu_k1);

    }*/
   /* p_k_sum = integral_image_cost_sum;
    src_image_pixels_sum_square = integral_image_squared_sum;
    p_pixels_product = sum_src_cost;*/

    float sigma_k_square = (src_image_pixels_sum_square / omega_size) - (mu_k * mu_k);
    float p_k = p_k_sum / omega_size;

    float a_k = ((p_pixels_product / omega_size) - mu_k * p_k) / (sigma_k_square + epsilon);
    float b_k = p_k - a_k * mu_k;


        dst_a_k[central_pixel] = a_k;
        dst_b_k[central_pixel] = b_k;


}