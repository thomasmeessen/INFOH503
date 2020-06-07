
#define BLOCK_DIM 16

__kernel void transpose(__global float* input, __global float* output, int width, int height, __local float* block){
	// read the matrix tile into shared memory
	size_t xIndex = get_global_id(0);
	size_t yIndex = get_global_id(1);
	size_t xLocal = get_local_id(0);
	size_t yLocal = get_local_id(1);

	if((xIndex < width) && (yIndex < height)){
		size_t index_in = yIndex * width + xIndex;
		size_t local_index = get_local_id(1)*BLOCK_DIM+get_local_id(0);
		block[local_index] = input[index_in];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	// write the transposed matrix tile to global memory
	if((xIndex < height) && (yIndex < width)){
		xIndex = get_group_id(1) * BLOCK_DIM + get_local_id(0);
		yIndex = get_group_id(0) * BLOCK_DIM + get_local_id(1);
		size_t index_out = yIndex * height + xIndex;
		output[index_out] = block[get_local_id(0)*BLOCK_DIM+get_local_id(1)];
	}

}
