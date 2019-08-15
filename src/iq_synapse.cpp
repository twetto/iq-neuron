#include "iq_synapse.h"

using namespace std;

int linenum_neuronParameter()
{
    int i[7], linenum = 0;
    FILE *fp = fopen("../inputs/neuronParameter.txt", "r");
    
    while(fscanf(fp, " %d %d %d %d %d %d %d", &i[0], &i[1], &i[2],
            &i[3], &i[4], &i[5], &i[6]) == 7) {
        linenum++;
    }
    fclose(fp);
    return linenum;
}

void set_neurons(int num_neurons, iq_neuron *neurons)
{
    int i, rest, threshold, reset, a, b, noise;
    FILE *fp;

    fp = fopen("../inputs/neuronParameter.txt", "r");
    while(fscanf(fp, " %d %d %d %d %d %d %d", &i, &rest, &threshold,
            &reset, &a, &b, &noise) == 7) {
        (neurons + i)->set(rest, threshold, reset, a, b, noise);
    }
    fclose(fp);
    return;
}

void get_weight(int num_neurons, int *weight)
{
    int i, j, temp;
    FILE *fp;
    for(i = 0; i < num_neurons; i++) {
        for(j = 0; j < num_neurons; j++) {
            *(weight + num_neurons*i + j) = 0;
        }
    }
    fp = fopen("../inputs/Connection_Table.txt", "r");
    while(fscanf(fp, "%d %d %d", &i, &j, &temp) == 3) {
        *(weight + num_neurons*i + j) = temp;       // CAUTION: i/j relation
    }
    fclose(fp);
    return;
}

void send_synapse(int num_neurons, iq_neuron *neurons,
                  int *weight, int tau, int *current, int *biascurrent)
{
    int i, j, temp;
    static int n = 0;
    static int f = (int) (log10(0.9) / log10((tau-1)/(float) tau));
    //if(n == 0) printf("%d\n", f);

    for(i = 0; i < num_neurons; i++) {
        if((neurons + i)->is_firing()) {
            //printf("neuron %d has fired!\n", i);
            for(j = 0; j < num_neurons; j++) {
                *(current + j) += *(weight + num_neurons*i + j);
            }
        }
    }
    for(i = 0; i < num_neurons; i++) {
        (neurons + i)->iq(*(current + i) + *(biascurrent + i));
        //printf("neuron %d current: %d\n", i, *(current + i));
        if(n >= f) {
            *(current + i) = *(current + i) * 9 / 10;
        }
        //*(current + i) = *(current + i) * (tau-1) / tau;
    }
    if(n >= f) n = 0;
    n++;
    return;
}

void delete_all(int *weight, int *current, iq_neuron *neurons)
{
    delete weight;
    delete current;
    delete neurons;
    return;
}

