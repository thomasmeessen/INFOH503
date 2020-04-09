kernel void memset(global uint* dst, global uint* o, global uint* cost) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    //const id = (y * get_global_size(0)) + x;

    int radius = 2;
    int omega_size = 25;
    int good_pixels = 0;


    for (int i = -radius; i <= radius; i++) {
        if (x + i >= 0 && x + i < get_global_size(0)) {
            for (int j = -radius; j <= radius; j++) {
                if (y + j >= 0 && y + j < get_global_size(1)) {
                    const id1 = ((y + j) * get_global_size(0)) + (x + i);
                    o[id1] = 0;
                    good_pixels++;
                }
            }
        }
    }
    if (good_pixels == 9) {
        printf("%d \n", good_pixels);
    }
    //printf("%d \n", good_pixels);

    //int omega_k = 
    //int theta_k_squared = (1/size)*
    //int sigma_k = 

}