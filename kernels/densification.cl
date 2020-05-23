
float choose_nonoccluded(__global float* ocluded_pixels, int x_, int rows_offset, int inc){
        while( x_ >= 0  && x_ < get_global_size(0)){
            if(ocluded_pixels[x_ + rows_offset] < 255.)
                return ocluded_pixels[x_ + rows_offset];
            x_ = x_ + inc;
        }
        return -1;
}

void simple_filling(__global float* left_image, __global float* ocluded_pixels, int x, int pixel, int rows_offset){
    if(ocluded_pixels[pixel] > 255.){
        const float left_nonoccluded = choose_nonoccluded(ocluded_pixels, x-1, rows_offset, -1);
        const float right_nonoccluded = choose_nonoccluded(ocluded_pixels, x+1, rows_offset, 1);

        left_image[pixel] = (left_nonoccluded > right_nonoccluded) ? left_nonoccluded : right_nonoccluded;
    }
}

kernel void densification(__global float* left_image, __global float* ocluded_pixels){
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int rows_offset = y * get_global_size(0);
    const int pixel = x + rows_offset;
    simple_filling(left_image, ocluded_pixels, x, pixel, rows_offset);
    
}