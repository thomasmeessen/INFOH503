kernel void consistency( __global float* left_image, __global float* right_image, __global float* output) {

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int left_pixel = x + y * get_global_size(0);
    const int right_pixel = left_pixel + (int) left_image[left_pixel];
    const float d_LR= 1.; // equation 28 
    // const int image_size = get_global_size(0)*get_global_size(1);
    const float disparity_difference = fabs(left_image[left_pixel] - right_image[right_pixel]); // disparity of right image is positive so we substract them
    if(disparity_difference >= d_LR){//reject
                output[left_pixel] = 255000.;
    }
    else{//accept
        output[left_pixel] = left_image[left_pixel];
    }
    
}
