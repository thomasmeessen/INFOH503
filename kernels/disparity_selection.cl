kernel void memset( __global double* cost, int disparity_range, __global double* output) {
    // prend un vecteur contenant les images � diff�rentes valeurs de disparit� (les diff�rents couches)
    // la valeur de chaque "pixel" de ces couches donne le co�t
    // la fonction donne en retour une nouvelle couche o� chaque pixel a une disparit� qui minimise le co�t
    //layer_size = nombres de couche

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int pixel = x + y * get_global_size(0);
    const int image_size = get_global_size(0)*get_global_size(1);

    double current_min_cost = cost[pixel];
    int current_best_disparity = 1;
    for (int i = 1; i < disparity_range; i++){
        if((double)cost[pixel+i*image_size] < current_min_cost){
            current_min_cost = (double)cost[pixel+i*image_size];
            current_best_disparity = i;
        }
    }
    double layer  = (double) current_best_disparity;

    output[pixel] = layer;
}