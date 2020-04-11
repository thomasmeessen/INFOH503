
kernel void memset(   global unsigned char *left_src, global unsigned char *right_src, global unsigned char *dst, int original_height, int original_width, int max_dist)
/*!

@var right_src = right image
@var left_src = left image
@var dst = destination of the retult. size of 16* left image

here each thread will compute result for the max_dist diffenret image for the pixel

Equation (3) in section 2.1 in the paper.
*/
{   const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int input_id = (x + (y*(original_width+2*max_dist)) ) + max_dist ;

    unsigned char r_left = left_src[input_id];

    for(int i = 0; i<max_dist; i++){ // compare with pixel at the same position until pixel at max_dist-1
        unsigned char r_right = right_src[input_id+i];
        const int output_id = x + (y*((original_width +2*max_dist)*max_dist)) + max_dist + i*(original_width + 2*max_dist);
        dst[output_id]   = abs(r_left - r_right); 
    }
} 