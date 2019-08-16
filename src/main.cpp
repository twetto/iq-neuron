#include "iq_network.h"
#include "iz_network.h"
#include <stdio.h>
#include <random>

using namespace std;

int main()
{
    int i, j;
    char filename[] = "iq_output_Number.txt";
    iq_network network_iq;
    iz_network network_iz;
    int num_neurons = network_iq.num_neurons();
    FILE** fp_iq = (FILE**) malloc(sizeof(FILE*) * num_neurons);
    FILE** fp_iz = (FILE**) malloc(sizeof(FILE*) * num_neurons);

    srand((unsigned) time(NULL));
    
    for(i = 0; i < num_neurons; i++) {
        sprintf(filename, "iq_output_%d.txt", i);
        fp_iq[i] = fopen(filename, "w");
        network_iq.set_biascurrent(i, 4);
        sprintf(filename, "iz_output_%d.txt", i);
        fp_iz[i] = fopen(filename, "w");
        network_iz.set_biascurrent(i, 4);
    }
    for(j = 0; j < 2000; j++) {
        network_iq.send_synapse();
        network_iq.printfile(fp_iq);
        network_iz.send_synapse();
        network_iz.printfile(fp_iz);
    }
    for(i = 0; i < num_neurons; i++) {
        network_iq.set_biascurrent(i, 0);
        network_iz.set_biascurrent(i, 0);
    }
    for(j = 0; j < 1000; j++) {
        network_iq.send_synapse();
        network_iq.printfile(fp_iq);
        network_iz.send_synapse();
        network_iz.printfile(fp_iz);
    }
    for(i = 0; i < num_neurons; i++) {
        fclose(fp_iq[i]);
        fclose(fp_iz[i]);
    }
    
    return 0;
}

