/* simple implementation of integer theta neuron.
 * credit to Alex White.
 * Chen-Fu Yeh, 2019.01.28
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>

#define NEURON_NUMBER 27
#define MAX_POTENTIAL 65535

int theta(int x, int rest,
          int a, int b,
          int threshold);

int main(void)
{
    int rest = 5000;
    int threshold = 15000;
    int a = 1, b = 2;               // slope of the V
    int reset[3] = {14999, 15000, 15001};
                                    // reset parameter that gives bistability
    int x1[3], x0[3];
    int input;
    int is_fired[3];
    int spike_count[3] = {0};
    int i, j, temp;
    char filename[] = "output_Number.txt";
    FILE *output_fp[3];
    time_t t;

    printf("RAND_MAX = %d\n", RAND_MAX);
    srand((unsigned) time(&t));
    for(i = 0; i < 3; i++) {
        sprintf(filename, "output_%d.txt", i);
        output_fp[i] = fopen(filename, "w");
        x0[i] = rest;      // initialize to rest potential
    }
    for(i = 0; i < 5000; i++) {
        for(j = 0; j < 3; j++) {
            x0[j] += theta(x0[j], rest, a, b, threshold)/100 + rand()%21-10;
            if(i < 1000) x0[j] += 100;
                                                // main theta function
            is_fired[j] = 0;
            if(x0[j] > 65535) {
                spike_count[j]++;               // spikes when overflow
                is_fired[j] = 1;
                x0[j] = reset[j];               // reset condition
            }
            fprintf(output_fp[j], "%d\n", x0[j]);  // output to list
        }
    }
    for(i = 0; i < 3; i++) {
        fclose(output_fp[i]);
    }
    return 0;
}

int theta(int x, int rest,
          int a, int b,
          int threshold)
{
    if(x < (a*rest + b*threshold) / (a+b)) return a * (rest - x);
    else return b * (x - threshold);
}

