
kernel void memset(   global unsigned char *src, global unsigned char *dst, int original_width, int original_height, int max_dist)
{   //I put padding above and below for the grid
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int id = x + (y*get_global_size(0));
    unsigned char grey = (src[id*3] + src[id*3+1] + src[id*3+2])/3;
    int new_width = original_width + 2*max_dist;
    int new_x = x + max_dist ;
    int new_y = y + max_dist;
    int new_id = new_x + (new_y * new_width ) ;
    dst[new_id]   = grey;  // R

    if(x==0){//first column, every pixel on the left has the same color
        for(int i=1; i <= max_dist; i++){
            dst[new_id - i] = grey;
        }
    }
    else if(x==(original_width-1)){//last column, every pixel on the left has the same color
        for(int i=1; i <= max_dist; i++){
            dst[new_id + i] = grey;
        }
    }
    if(y==0){//first line we copy the color to every pixel above this one
        for(int i=1; i <= max_dist; i++){
            dst[new_id - i*new_width] = grey;
        }
    }
    else if(y==(original_height-1)){//last line we copy the color to every pixel below this one
        for(int i=1; i <= max_dist; i++){
            dst[new_id + i*new_width] = grey;
        }
    }
}