
kernel void memset(   global unsigned char *dst )
{
    const int id = get_global_id(0);
    unsigned char grey = (dst[id*3] + dst[id*3+1] + dst[id*3+2])/3;
    dst[id*3]   = grey; 
    dst[id*3+1] = grey;
    dst[id*3+2] = grey;
}