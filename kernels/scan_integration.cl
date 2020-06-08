
/**
On the all array each work item add to a "pixel" the cumulative sum computed at the end of the previous bloc
**/
kernel void scan_integration( __global  float *bloc_based_integral_image, __global float *bloc_sums, int pixel_per_row, int bloc_per_row) {

    size_t local_size = get_local_size(0);
    size_t local_group_number = get_num_groups(0);
    size_t idx = get_global_id(0);
    size_t local_id = get_local_id(0);
    size_t group_id = get_group_id(0);

    int row_number = group_id / bloc_per_row;
    // One id per pixel
    int thread_id_in_row = idx - row_number *  local_size * bloc_per_row;
    int memory_offset = row_number * pixel_per_row ;
    int observed_bloc_size = 2 * local_size;
    // We add nothing to the first bloc;
    int memory_index = thread_id_in_row + memory_offset  ;
    bool in_memory = memory_index < (row_number +1) * pixel_per_row;

    // Each scan bloc affect 2 time it size. So each sum result must be applied on 2 group size.
    // We add the sum of the previous bloc
    int group_id_offset = row_number * bloc_per_row;
    int observed_pixel_group_id = floor((float)group_id /(float)2) ;




    // Each pixel receive the cumulative sum of the pixel included in the previous groups.

    if(in_memory) {
        bloc_based_integral_image[memory_index] +=  bloc_sums[observed_pixel_group_id];
        /**
        if(bloc_sums[group_id] < bloc_sums[group_id -1] && get_local_id(0) == 0){
                printf("id = %i, group_id %i blocSum = %f next bloc sum = %f local: %i \n", global_id, group_id, bloc_sums[group_id], bloc_sums[group_id +1], get_local_id(0));
            }
        **/
    }


}