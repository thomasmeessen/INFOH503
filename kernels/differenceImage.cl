
kernel void memset(   global unsigned char *left_src, global unsigned char *right_src, global unsigned char *dst, int height, int width, int maxDistance)
/*!

@var right_src = right image
@var left_src = left image
@var dst = destination of the retult. size of 16* left image

here each thread will compute result for the maxDistance diffenret image for the pixel

Equation (3) in section 2.1 in the paper.
*/
{   const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int originial_id = x + (y*(get_global_size(0)));
    unsigned char grey = (left_src[originial_id*3] + left_src[originial_id*3+1] + left_src[originial_id*3+2])/3;
    //code if we put eacch image in a column
    // for(int i = 0; i<=maxDistance; i++){
    //     const int output_id = originial_id + (width*height)*i;
        
    //     dst[output_id]   = grey; 
    //     dst[output_id+1] = grey;
    //     dst[output_id+2] = grey;

    // }

    //code if we put each image in a row
    for(int i = 0; i<=maxDistance; i++){
        const int output_id = x + (y*(width*maxDistance)) + i*width;
        
        dst[output_id]   = grey; 
        dst[output_id+1] = grey;
        dst[output_id+2] = grey;

    }

    // const int id = x + (y*(width*maxDistance));
    // const int id = x + (y*(get_global_size(0)));
} 