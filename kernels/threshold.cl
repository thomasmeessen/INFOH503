
kernel void memset(   global uint *dst )
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const id = y * get_global_size(0) + x ;

    if ( dst[id] > 135)
    {
        dst[id] = 200;
    } else
    {
        dst[id] = dst[id];
    }
}