#include "iq_network.h"
#include "iz_network.h"
#include <stdio.h>
#include <random>
#include <time.h>

using namespace std;

int main()
{
    clock_t start, end;
    double cpu_time_used;

    int i, j, k;
    char filename[] = "iq_output_Number.txt";
    iq_network network_iq;
    int iq_num_neurons = network_iq.num_neurons();
    FILE** fp_iq = (FILE**) malloc(sizeof(FILE*) * iq_num_neurons);

    srand((unsigned) time(NULL));

    for(i = 0; i < iq_num_neurons; i++) {
        sprintf(filename, "iq_output_%d.txt", i);
        fp_iq[i] = fopen(filename, "w");
    }

    int num_steps, index, bias;
    printf("This is a customized testbench. If you are finished simulating, please type '-1' to quit.\n");
    printf("How many timesteps do you want?\n");
    scanf(" %d", &num_steps);
    while(num_steps >= 0) {
        printf("num_steps: %d\n", num_steps);
        printf("Which neuron do you want to insert bias current?\n");
        scanf(" %d", &index);
        while(index >= 0) {
            printf("How much current do you want to insert to neuron %d?\n", index);
            scanf(" %d", &bias);
            network_iq.set_biascurrent(index, bias);
            printf("neuron %d is receiving current %d. Waiting for another input...\n", index, bias);
            printf("Which neuron do you want to insert bias current?\n");
            scanf(" %d", &index);
        }
        printf("Set complete; sending synapses...\n");
        for(i = 0; i < num_steps; i++) {
            network_iq.send_synapse();
            network_iq.printfile(fp_iq);
        }
        printf("Synapse OK. Waiting for next period...\n");
        printf("How many timesteps do you want?\n");
        scanf(" %d", &num_steps);
    }
    printf("Simulation finished. Quitting...\n");

    for(i = 0; i < iq_num_neurons; i++) {
        fclose(fp_iq[i]);
    }
    free(fp_iq);
    
    return 0;
}

