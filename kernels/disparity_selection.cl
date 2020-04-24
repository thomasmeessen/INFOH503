kernel void memset( global float* cost, int number_of_layers, __global unsigned char* output) {
    // prend un vecteur contenant les images � diff�rentes valeurs de disparit� (les diff�rents couches)
    // la valeur de chaque "pixel" de ces couches donne le co�t
    // la fonction donne en retour une nouvelle couche o� chaque pixel a une disparit� qui minimise le co�t
    //layer_size = nombres de couche

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int pixel = x + y * get_global_size(0);
    const int image_size = get_global_size(0)*get_global_size(1);

    float current_best = cost[pixel]; 
    int current_best_index = 1;
    for (int i = 1; i < number_of_layers; i++){
        if(cost[pixel+i*image_size] < current_best){
            current_best = cost[pixel+i*image_size];
            current_best_index = i+1; 
        }
    }
    output[pixel] = current_best_index;
}
