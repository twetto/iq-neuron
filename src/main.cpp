#include "synapse.h"
#include <stdio.h>
#include <random>

using namespace std;

int main()
{
    int num_neurons = linenum_neuronParameter();
    int *weight, *current;
    int i, j;
    iq_neuron *neurons;
    char filename[] = "output_Number.txt";
    //FILE** fp = (FILE**) malloc(sizeof(FILE*) * num_neurons);
    int last_spike_is_at[num_neurons] = {0};

    srand((unsigned) time(NULL));
    neurons = new iq_neuron[num_neurons];
    weight = new int[num_neurons * num_neurons];
    current = new int[num_neurons];
    
    set_neurons(num_neurons, neurons);
    get_weight(num_neurons, weight);
    /*
    for(i = 0; i < num_neurons; i++) {
        sprintf(filename, "output_%d.txt", i);
        fp[i] = fopen(filename, "w");
    }
    */
    for(j = 0; j < 5000; j++) {
        send_synapse(num_neurons, neurons, weight, 500, current); if(j < 100) {
            *(current + 1) = 4;
            *(current + 2) = 4;
        }
        
        for(i = 0; i < num_neurons; i++) {
            //fprintf(fp[i], "%d\n", (neurons+i)->potential());
            if((neurons + i)->is_firing()) {
                last_spike_is_at[i] = j;
            }
        }
        
        //*(current + 0) = 1000;
    }
    
    for(i = 0; i < num_neurons; i++) {
        printf("%d\n", last_spike_is_at[i]);
    }
    
    delete_all(weight, current, neurons);
    /*
    for(i = 0; i < num_neurons; i++) {
        fclose(fp[i]);
    }
    */
    return 0;
}

