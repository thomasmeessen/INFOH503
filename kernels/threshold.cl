
kernel void memset(   global uint *dst )
{
    const int id = get_global_id(0);

    if ( dst[id] > 135)
    {
        dst[id] = 100;
    } else
    {
        dst[id] = dst[id];
    }
}