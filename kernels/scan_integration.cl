
/**
On the all array each work item add to a "pixel" the cumulative sum computed at the end of the previous bloc
**/
kernel void scan_integration( __global  float *bloc_based_integral_image, __global float *bloc_sums, int limit) {
    int bloc_size = get_local_size(0);
    // The first bloc do not need to be changed
    int global_id = get_global_id(0);
    bool outside_memory = global_id > limit;
    int shifted_id = global_id + bloc_size;
    // Each scan bloc affect 2 time it size. So each sum result must be applied on 2 group size.
    // We add the sum of the previous bloc
    int group_id = (shifted_id /( 2* bloc_size)) -1;

    /**
    if(get_local_id(0) == 0 && group_id< 4 ){
        printf("id = %i, index %i blocSum = %f \n", global_id, group_id, bloc_sums[group_id]);
    }
    **/
    // Each pixel receive the cumulative sum of the pixel included in the previous groups.

    if(!outside_memory) bloc_based_integral_image[shifted_id] +=  bloc_sums[group_id];
    //if(!outside_memory)bloc_based_integral_image[shifted_id] = shifted_id;

}