
kernel void memset(   global uint *dst )
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int id = x + (y*get_global_size(0));
    if ( dst[id] > 135)
    {
        dst[id] = 0;
    } else
    {
        dst[id] = 255;
    }
}