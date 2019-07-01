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
            (neurons + i)->set(5000, 15000, 20000, 1, 1);
        }
        *(current + i) = 0;
    }
    (neurons + 0)->set(5000, 15000, 10000, 1, 1);
    get_weight(fp[0], num_neurons, weight);
    for(i = 0; i < num_neurons; i++) {
        sprintf(filename, "output_%d.txt", i);
        fp[i] = fopen(filename, "w");
    }
    printf("fp OK\n");
    for(j = 0; j < 10000; j++) {
        send_synapse(num_neurons, neurons, weight, 40, current);
        for(i = 0; i < num_neurons; i++) {
            if(j < 1000) {
                *(current + 1) = 1000;
                *(current + 2) = 1000;
            }
            fprintf(fp[i], "%d\n", (neurons+i)->potential());
        }
    }
    delete_all(weight, current, neurons);
    for(i = 0; i < num_neurons; i++) {
        fclose(fp[i]);
    }
    return 0;
}

