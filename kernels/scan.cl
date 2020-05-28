/**
Following the work presented in class and by DOI: 10.1109/IVS.2010.5548142

 and inspired by: Accelerated Filtering using OpenCL, J. Waage
**/

kernel void scan( __global unsigned char *indat, __global float *answer, __local float *temp) {

    size_t local_size = get_local_size(0);
    size_t local_group_number = get_num_groups(0);
    size_t idx = get_global_id(0);
    size_t idy = get_global_id(1);
    size_t local_id = get_local_id(0);
    size_t group_id = get_group_id(0);

    // Copy a element to the local memory and converting it to float
    temp[local_id] = (float) indat[group_id * local_size + 2 * local_id];
    temp[local_id + 1] = (float) indat[group_id * local_size + 2 * local_id + 1];

    /**
    if (group_id == 0){
        if (local_id == 0){
            printf(" -- In Kernel -- \n");
            printf("local_size = %i \n", (int) local_size);
            printf("Number of group = %i \n", (int) local_group_number);
            printf("Copied value %i ", (int) temp[local_id]);
        }else{
            printf(" %i ", (int) temp[local_id]);
        }
    }
    **/

}