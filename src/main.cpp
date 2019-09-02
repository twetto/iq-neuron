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
    iz_network network_iz;
    int iq_num_neurons = network_iq.num_neurons();
    int iz_num_neurons = network_iz.num_neurons();
    FILE** fp_iq = (FILE**) malloc(sizeof(FILE*) * iq_num_neurons);
    FILE** fp_iz_v = (FILE**) malloc(sizeof(FILE*) * iz_num_neurons);
    FILE** fp_iz_u = (FILE**) malloc(sizeof(FILE*) * iz_num_neurons);

    srand((unsigned) time(NULL));


    for(i = 0; i < iq_num_neurons; i++) {
        sprintf(filename, "iq_output_%d.txt", i);
        fp_iq[i] = fopen(filename, "w");
    }
    for(i = 0; i < iz_num_neurons; i++) {
        sprintf(filename, "iz_output_v_%d.txt", i);
        fp_iz_v[i] = fopen(filename, "w");
        sprintf(filename, "iz_output_u_%d.txt", i);
        fp_iz_u[i] = fopen(filename, "w");
    }
    
    for(i = 0; i < iq_num_neurons; i++) {
        network_iq.set_biascurrent(i, 4);
    }
    for(i = 0; i < iz_num_neurons; i++) {
        network_iz.set_biascurrent(i, 4);
    }
    for(j = 0; j < 2000; j++) {
        network_iz.send_synapse();
        network_iz.printfile(fp_iz_v, fp_iz_u);
        network_iq.send_synapse();
        network_iq.printfile(fp_iq);
    }
    for(i = 0; i < iq_num_neurons; i++) {
        network_iq.set_biascurrent(i, 0);
    }
    for(i = 0; i < iz_num_neurons; i++) {
        network_iz.set_biascurrent(i, 0);
    }
    for(j = 0; j < 2000; j++) {
        network_iz.send_synapse();
        network_iz.printfile(fp_iz_v, fp_iz_u);
        network_iq.send_synapse();
        network_iq.printfile(fp_iq);
    }
    
    /*
    start = clock();
    for(k = 0; k < 50000; k++) {
        for(i = 0; i < num_neurons; i++) {
            network_iq.set_biascurrent(i, 4);
        }
        for(j = 0; j < 2000; j++) {
            network_iq.send_synapse();
            //network_iq.printfile(fp_iq);
        }
        for(i = 0; i < num_neurons; i++) {
            network_iq.set_biascurrent(i, 0);
        }
        for(j = 0; j < 1000; j++) {
            network_iq.send_synapse();
            //network_iq.printfile(fp_iq);
        }
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("IQIF took %f seconds to execute.\n", cpu_time_used);

    start = clock();
    for(k = 0; k < 50000; k++) {
        for(i = 0; i < num_neurons; i++) {
            network_iz.set_biascurrent(i, 4);
        }
        for(j = 0; j < 2000; j++) {
            network_iz.send_synapse();
            //network_iz.printfile(fp_iz);
        }
        for(i = 0; i < num_neurons; i++) {
            network_iz.set_biascurrent(i, 0);
        }
        for(j = 0; j < 1000; j++) {
            network_iz.send_synapse();
            //network_iz.printfile(fp_iz);
        }
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Izhikevich took %f seconds to execute.\n", cpu_time_used);
    */
    for(i = 0; i < iq_num_neurons; i++) {
        fclose(fp_iq[i]);
    }
    for(i = 0; i < iz_num_neurons; i++) {
        fclose(fp_iz_v[i]);
        fclose(fp_iz_u[i]);
    }
    free(fp_iq);
    free(fp_iz_v);
    free(fp_iz_u);
    
    return 0;
}

