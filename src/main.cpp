#include "synapse.h"
#include <stdio.h>

using namespace std;

int main()
{
    int num_neurons = 3;
    int *weight, *current;
    int i, j;
    iq_neuron *neurons;
    char filename[] = "output_Number.txt";
    FILE *fp[3];

    srand((unsigned) time(NULL));
    neurons = new iq_neuron[num_neurons];
    weight = new int[num_neurons * num_neurons];
    current = new int[num_neurons];
    for(i = 0; i < num_neurons; i++) {
        if(!(neurons + i)->is_set()) {
            (neurons + i)->set(50, 70, 80, 1, 1, 5);
        }
        *(current + i) = 0;
    }
    (neurons + 0)->set(40, 70, 40, 1, 1, 3);
    get_weight(fp[0], num_neurons, weight);
    for(i = 0; i < num_neurons; i++) {
        sprintf(filename, "output_%d.txt", i);
        fp[i] = fopen(filename, "w");
    }
    printf("fp OK\n");
    for(j = 0; j < 10000; j++) {
        send_synapse(num_neurons, neurons, weight, 5, current);
        if(j < 100) {
            *(current + 1) = 50;
            *(current + 2) = 50;
        }
        for(i = 0; i < num_neurons; i++) {
            fprintf(fp[i], "%d\n", (neurons+i)->potential());
        }
        //*(current + 0) = 1000;
    }
    delete_all(weight, current, neurons);
    for(i = 0; i < num_neurons; i++) {
        fclose(fp[i]);
    }
    return 0;
}

