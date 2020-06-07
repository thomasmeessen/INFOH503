

__kernel void transpose(__global float* input, __global float* output, int width, int height, __local float* block){
	// read the matrix tile into shared memory
	size_t xGroup = get_group_id(0);
	size_t yGroup = get_group_id(1);
	size_t xLocal = get_local_id(0);
	size_t yLocal = get_local_id(1);
	size_t bloc_width = get_local_size(0);


    size_t xIndex = xGroup * bloc_width + xLocal;
    size_t yIndex = yGroup * bloc_width + yLocal;
    size_t index_in = yIndex * width + xIndex;
    size_t local_index = yLocal *bloc_width + xLocal;
    block[local_index] = input[index_in];


	barrier(CLK_LOCAL_MEM_FENCE);

	
	
	// write the transposed matrix tile to global memory
    xIndex = yGroup * bloc_width + xLocal;
    yIndex = xGroup * bloc_width + yLocal;
    local_index = xLocal * bloc_width + yLocal;
    size_t index_out = yIndex * height + xIndex;

    if(xLocal == 0 && yLocal == 0){
        printf(" %f ", block[local_index]);
    }
    output[index_out] = block[local_index];


}
