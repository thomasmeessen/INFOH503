
/**
On the all array each work item add to a "pixel" the cumulative sum computed at the end of the previous bloc
**/
kernel void scan_integration( __global  float *bloc_based_integral_image, __global float *bloc_sums) {
    int global_id = get_global_id(0);
    int bloc_size = get_local_size(0);
    // The first step have its work item impacting each 2 pixels. So each sum result must be applied on 2 group size.
    int group_id = global_id /( 2* bloc_size);

    /**
    if(get_local_id(0) == 0 && group_id< 4 ){
        printf("id = %i, index %i blocSum = %f \n", global_id, group_id, bloc_sums[group_id]);
    }
    **/
    // Each pixel receive the cumulative sum of the pixel included in the previous groups.
    bloc_based_integral_image[global_id] +=  bloc_sums[group_id];

}