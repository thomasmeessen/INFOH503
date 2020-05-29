/**
Following the work presented in class and by DOI: 10.1109/IVS.2010.5548142
and inspired by: Accelerated Filtering using OpenCL, J. Waage

The local buffer have a size of 2 * local_size.
Each work item load 2 number into local memory and then perform an addition on those.
-- Rest of the algorithm
Then each work item load 2 number into global memory.

**/

kernel void scan( __global  uchar2 *indat, __global float2 *answer, __local float *temp) {

    size_t local_size = get_local_size(0);
    size_t local_group_number = get_num_groups(0);
    size_t idx = get_global_id(0);
    size_t idy = get_global_id(1);
    size_t local_id = get_local_id(0);
    size_t group_id = get_group_id(0);

    // Copy a element to the local memory and converting it to float

    uchar2 in = indat[idx];
    temp[2*local_id] = (float) in.x;
    temp[2*local_id +1] = (float) in.y;


    int offset = 1;
    for(int depth = local_size ; depth > 0; depth >>=1)
    {
        // Divide each time the number of active work item by 2

        barrier(CLK_LOCAL_MEM_FENCE);

        if(local_id < depth)
        {
            // offset will not create out of bound index because idx is capped by a number progressively divided by2
            int ai = offset*(2*local_id+1)-1;
            int bi = offset*(2*local_id+2)-1;
            /**
            if(local_id == 0 && group_id ==1){
                                    printf(" offset = %i ", offset);
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
        temp[local_size * 2- 1] = 0;
    }

    for(int d = 1; d < local_size *2 ; d *= 2)
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

    answer[idx] = (float2) {temp[2*local_id], temp[2*local_id]};


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