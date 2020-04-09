
kernel void memset(__global unsigned char *src, __global unsigned char* dst)
{   const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int id = x + (y*get_global_size(0));
   /* unsigned char grey = (dst[id*3] + dst[id*3+1] + dst[id*3+2])/3;
    dst[id*3]   = grey; 
    dst[id*3+1] = grey;
    dst[id*3+2] = grey;*/
    unsigned char gray = 0.299 * src[id * 3] + 0.587 * src[id * 3 + 1] + 0.0721 * src[id * 3 + 2];
    dst[id] = gray;
}