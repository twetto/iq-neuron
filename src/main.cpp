/* main
 * This is a custom main for optimization testing.
 * Chen-Fu Yeh, 19/10/09
 */

#include "iq_network.h"
#include "iz_network.h"
#include <stdio.h>
#include <random>
#include <time.h>

using namespace std;

int main(void)
{
    clock_t start, end;
    double time_total;

    int i;
    //iq_network network_iq("../inputs/neuronParameter_IQIF_bistable.txt", "../inputs/Connection_Table_IQIF_bistable.txt");
    iz_network network_iz("neuron_iz.txt", "table_iz.txt");
    //int iq_num_neurons = network_iq.num_neurons();
    int iz_num_neurons = network_iz.num_neurons();
    start = clock();
    //char filename[] = "iq_output_number.txt";
    char filename[] = "iz_output_p_number.txt";
    //FILE** fp = (FILE**) malloc(sizeof(FILE*) * iq_num_neurons);
    FILE** fp_p = (FILE**) malloc(sizeof(FILE*) * iz_num_neurons);
    FILE** fp_a = (FILE**) malloc(sizeof(FILE*) * iz_num_neurons);

    srand((unsigned) time(NULL));
    //network_iq.set_num_threads(4);
    //network_iz.set_num_threads(4);

    //for(i = 0; i < iq_num_neurons; i++) {
    for(i = 0; i < iz_num_neurons; i++) {
        //sprintf(filename, "iq_output_%d.txt", i);
        //fp[i] = fopen(filename, "w");
        sprintf(filename, "iz_output_p_%d.txt", i);
        fp_p[i] = fopen(filename, "w");
        sprintf(filename, "iz_output_a_%d.txt", i);
        fp_a[i] = fopen(filename, "w");
        //network_iq.set_biascurrent(i, 4);
        //network_iz.set_biascurrent(i, 4);
    }

    int steps, idx;
    float bias;
    //printf("How many timesteps do you want?\n");
    scanf(" %d", &steps);
    while(steps >= 0) {
        //printf("steps: %d\n", steps);

        //printf("Neuron to insert bias current:\n");
        scanf(" %d", &idx);
        while(idx >= 0) {
            //printf("How much current do you want to insert?\n");
            scanf(" %f", &bias);
            //network_iq.set_biascurrent(idx, bias);
            printf("%f\n", bias);
            network_iz.set_biascurrent(idx, bias);
            //printf("neuron %d is receiving current %d\n", idx, bias);

            //printf("Neuron to insert bias current:\n");
            scanf(" %d", &idx);
        }

        //printf("Set complete; sending synapses...\n");
        for(i = 0; i < steps; i++) {
            //network_iq.send_synapse();
            //network_iq.printfile(fp);
            network_iz.send_synapse();
            network_iz.printfile(fp_p, fp_a);
        }
        //printf("Synapse OK. Waiting for next period...\n");
        //printf("How many timesteps do you want?\n");
        scanf(" %d", &steps);
    }

    /* set bias current */
    //network_iq.set_biascurrent(0, 3);

    /* send synapse */
    /*
    for(i = 0; i < 2000; i++) {
        //printf("%d\n", i);
        network_iq.send_synapse();
        //network_iz.send_synapse();
        network_iq.printfile(fp);
        //network_iz.printfile(fp_p, fp_a);
    }
    */

    //printf("Simulation finished. Quitting...\n");

    //for(i = 0; i < iq_num_neurons; i++) {
    for(i = 0; i < iz_num_neurons; i++) {
        //fclose(fp[i]);
        fclose(fp_p[i]);
        fclose(fp_a[i]);
    }
    //free(fp);
    free(fp_p);
    free(fp_a);
    end = clock();
    time_total = (double) (end - start) / CLOCKS_PER_SEC;
    //printf("total execution time: %f sec\n", time_total);
    return 0;
}

