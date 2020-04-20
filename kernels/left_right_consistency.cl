kernel void memset(__global unsigned char* left_to_right_disparity, __global unsigned char* right_to_left_disparity, __global unsigned char* output, int d_LR, float reject_color) {
    /*
    d_LR is the 
    we will check if our disparity is right and consistent with the right image 
    if for a pixel the disparity isn't right then we "reject" it and we "return" (need to determine how)
    all the rejected pixels because it'd mean that they are occluded.

    What we do is basically check if for a pixel p on the left image with a disparity d if the pixel at position (p + d) has the same disparity

    
    */
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int pixel = x + y*get_global_size(); // no padding anymore
    const int left_disparity = left_to_right_disparity[pixel];
    const int right_disparity = right_to_left_disparity[pixel + left_disparity]

    // code to illustrate pixel precision rejection. (return are just her for the example, this is not runnable code)
    // if(left_disparity != right_disparity){return 0;}
    // else{return 1;}

    // d_lr = 0; // no subpixel
    // d_lr = 1; // handle 1 subpixel disparity?
    if(abs(left_disparity + right_disparity) >= d_LR){//reject
        output[pixel] = reject_color // we say "hoho there's a problem there"
    }
    else{//accept
        output[pixel] = left_to_right_disparity[pixel]; // we write the right disparity back
    }

    //one way to write output 
}