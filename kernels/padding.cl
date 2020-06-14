kernel void padding(__global float* src, int padding, int width, int height) {
    // this kernel replace the black padding with the padding corresponding to the value of the pixel at the edges.
    const int x = get_global_id(0);
    int original_width = width - 2 * padding;
    int padded_height = height + 2 * padding;



    if (x == 0) {
            for (int j = 0; j < height; j++) {
                for (int i = 0; i < padding; i++) {
                    src[(x + (padding +j) * width) + i] = src[(x + (padding + j) * width) + padding];
                    src[(x + (padding+j) * width + width - 1) - i] = src[x + (padding + j) * width + width - padding-1];
                }
            }
            for (int i = 0; i < padding; i++) {
                for (int j = 0; j < padding; j++) {
                    src[i + j * width] = src[padding + padding * width];  //left upper edge
                    src[(width-1) + j*(width) - i] = src[(width - 1) + padding * (width)-padding]; //right upper edge
                    src[(i + j * width) + (height + padding) * width] = src[(x + padding) + (padded_height - 1 - padding) * width]; //left bottom edge
                    src[(width-1 + j * width - i) + (height+padding)*width] = src[padding + original_width + (padded_height - 1 - padding) * width]; //right bottom edge
                }
            }
    }
        for (int i = 0; i < padding; i++) {
            src[(x + padding) + i * width] = src[(x + padding) + (padding) * width];
            src[(x + padding) + (padded_height - 1 - i) * width] = src[(x + padding) + (padded_height - 1 - padding) * width];
        }


}
