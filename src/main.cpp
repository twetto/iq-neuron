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
    iq_network network_iq;
    //iz_network network_iz;
    int iq_num_neurons = network_iq.num_neurons();
    start = clock();
    //int iz_num_neurons = network_iz.num_neurons();
    //char filename[] = "iq_output_number.txt";
    //char filename[] = "iz_output_p_number.txt";
    //FILE** fp = (FILE**) malloc(sizeof(FILE*) * iq_num_neurons);
    //FILE** fp_p = (FILE**) malloc(sizeof(FILE*) * iz_num_neurons);
    //FILE** fp_a = (FILE**) malloc(sizeof(FILE*) * iz_num_neurons);

    srand((unsigned) time(NULL));
    //network_iq.set_num_threads(4);
    //network_iz.set_num_threads(1);

    /* set bias current */
    for(i = 0; i < iq_num_neurons; i++) {
    //for(i = 0; i < iz_num_neurons; i++) {
        //sprintf(filename, "iq_output_%d.txt", i);
        //fp[i] = fopen(filename, "w");
        //sprintf(filename, "iz_output_p_%d.txt", i);
        //fp_p[i] = fopen(filename, "w");
        //sprintf(filename, "iz_output_a_%d.txt", i);
        //fp_a[i] = fopen(filename, "w");
        network_iq.set_biascurrent(i, 4);
        //network_iz.set_biascurrent(i, 4);
    }

    /* send synapse */
    for(i = 0; i < 1000; i++) {
        printf("%d\n", i);
        network_iq.send_synapse();
        //network_iz.send_synapse();
        //network_iq.printfile(fp);
        //network_iz.printfile(fp_p, fp_a);
    }

    //for(i = 0; i < iq_num_neurons; i++) {
    //for(i = 0; i < iz_num_neurons; i++) {
        //fclose(fp[i]);
        //fclose(fp_p[i]);
        //fclose(fp_a[i]);
    //}
    //free(fp);
    //free(fp_p);
    //free(fp_a);
    end = clock();
    time_total = (double) (end - start) / CLOCKS_PER_SEC;
    printf("total execution time: %f sec\n", time_total);
    return 0;
}

