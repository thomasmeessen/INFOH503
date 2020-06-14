
/**
Following the work presented in class and by DOI: 10.1109/IVS.2010.5548142
and inspired by: Accelerated Filtering using OpenCL, J. Waage
The local buffer have a size of 2 * local_size.
Each work item load 2 number into local memory and then perform an addition on those.
-- Rest of the algorithm
Then each work item load 2 number into global memory.
**/

kernel void scan( __global  float *indat, __local float *temp, __global float *bloc_sum, int pixel_per_row, int num_bloc_per_row) {

    size_t local_size = get_local_size(0);
    size_t local_group_number = get_num_groups(0);
    size_t idx = get_global_id(0);
    size_t local_id = get_local_id(0);
    size_t group_id = get_group_id(0);



    // Each group work on a subset of the global memory
    int row_number = group_id / num_bloc_per_row;
    int memory_offset = row_number * pixel_per_row ;
    // Within this subset, a row this is the thread id
    int row_id = idx - row_number *  local_size * num_bloc_per_row;
    // Because their are more thread than pixel in a subset there is a activation condition
    bool in_memory = (2*row_id+1 < pixel_per_row );
    int n = local_size *2;

    // Copy a element to the local memory and converting it to float
    temp[2*local_id] = (in_memory)? indat[2*row_id + memory_offset]:0;
    temp[2*local_id +1] = (in_memory)? indat[2*row_id+1 + memory_offset] : 0;


    int offset = 1;
    for(int depth = n>>1 ; depth > 0; depth >>=1)
    {
        // Divide each time the number of active work item by 2

        barrier(CLK_LOCAL_MEM_FENCE);

        if(local_id < depth)
        {
            // offset will not create out of bound index because idx is capped by a number progressively divided by2
            int ai = offset*(2*local_id+1)-1;
            int bi = offset*(2*local_id+2)-1;

            /**
            if(idx == 0 ){
                                    printf(" offset = %i ", memory_offset);
                                    printf(" bi = %i ", bi);
                                    printf(" , %f ", temp[bi]);
                                    printf(" ai = %i ",  ai);
                                    printf(" , %f \n", temp[ai]);
                                    }
                                    **/



            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    // The sum of all pixels is stored to be used later
    if(local_id == 0) {
        if(group_id < local_group_number -1){
            bloc_sum[group_id] = temp[n- 1];
        }
        temp[n - 1] = 0;
    }

    for(int d = 1; d < n ; d *= 2)
    {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if(local_id < d)
        {

            int ai = offset*(2*local_id+1)-1;
            int bi = offset*(2*local_id+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);


    if(in_memory) {
        indat[2*row_id + memory_offset] = temp[2*local_id];
        indat[2*row_id+1 + memory_offset] = temp[2*local_id +1];
    }
    /**
    if (idx == 0){
                printf(" -- In Kernel -- \n");
                printf("local_size = %i \n", (int) local_size);
                printf("Number of group = %i \n", (int) local_group_number);
                printf("Offset %i ", memory_offset);
                printf("Copied value %i ", (int) temp[2*local_id]);
            }**/
     /**
    if (group_id == 0){
        if (local_id == 0){
            printf(" -- In Kernel -- \n");
            printf("local_size = %i \n", (int) local_size);
            printf("Number of group = %i \n", (int) local_group_number);
            printf("Copied value %i ", (int) temp[local_id]);
        }else{
            printf(" %i ", (int) answer[2*idx]);
        }
    }
    **/


}
