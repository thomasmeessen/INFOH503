
/**
On the all array each work item add to a "pixel" the cumulative sum computed at the end of the previous bloc
**/
kernel void scan_integration( __global  float *bloc_based_integral_image, __global float *bloc_sums, int limit, int offset) {

    int bloc_size = get_local_size(0);

    // The first bloc do not need to be changed
    int global_id = get_global_id(0);
    bool outside_memory = global_id > limit -1;
    // Each scan bloc affect 2 time it size. So each sum result must be applied on 2 group size.
    // We add the sum of the previous bloc
    int group_id = (int)floor((float)global_id /(float)( 2* bloc_size)) ;



    // Each pixel receive the cumulative sum of the pixel included in the previous groups.

    if(!outside_memory) {
        bloc_based_integral_image[global_id + offset] +=  bloc_sums[group_id];
        /**
        if(bloc_sums[group_id] < bloc_sums[group_id -1] && get_local_id(0) == 0){
                printf("id = %i, group_id %i blocSum = %f next bloc sum = %f local: %i \n", global_id, group_id, bloc_sums[group_id], bloc_sums[group_id +1], get_local_id(0));
            }
        **/
    }
    //if(!outside_memory)bloc_based_integral_image[shifted_id] = shifted_id;

}