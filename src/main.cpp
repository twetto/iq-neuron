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
    start = clock();
    double time_total;
    double time_synapse = 0;
    double time_ode = 0;
    double time_decay = 0;

    int i, j;
    iq_network network_iq;
    int iq_num_neurons = network_iq.num_neurons();
    char filename[] = "iq_output_number.txt";
    //FILE** fp = (FILE**) malloc(sizeof(FILE*) * iq_num_neurons);

    srand((unsigned) time(NULL));

    /* set bias current */
    for(i = 0; i < iq_num_neurons; i++) {
        //sprintf(filename, "iq_output_%d.txt", i);
        //fp[i] = fopen(filename, "w");
        network_iq.set_biascurrent(i, 4);
    }
    /* send synapse */
    for(i = 0; i < 100000; i++) {
        //printf("%d\n", i);
        network_iq.send_synapse(time_synapse, time_ode, time_decay);
        //network_iq.printfile(fp);
    }
    /*
    for(i = 0; i < iq_num_neurons; i++) {
        fclose(fp[i]);
    }
    free(fp);
    */
    end = clock();
    time_total = (double) (end - start) / CLOCKS_PER_SEC;
    printf("total execution time: %f sec\n", time_total);
    printf("synapse accumulate time: %f sec\n", time_synapse);
    printf("inject current & ODE time: %f sec\n", time_ode);
    printf("synapse decay time: %f sec\n", time_decay);
    return 0;
}

