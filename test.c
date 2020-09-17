#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "network.h"


static const float two_pi = 3.14159;


int main() {
    sparknet_network my_net;

    srand(time(NULL));

    sparknet_network_init(&my_net, 4, 2);

    int middle_layer = sparknet_network_add_layer(&my_net, 2, 1.0, 0.001);

    sparknet_network_connect_input(&my_net, middle_layer, 0.75);
    sparknet_network_connect_output(&my_net, middle_layer, 0.8);

    float actual = 0.0;
    float desired = 0.0;

    for (int iteration = 0; iteration < 500; iteration++) {
        desired = 5 * sinf((float) iteration * two_pi / 16);

        my_net.inputs[0] = (iteration % 4 == 0);
        my_net.inputs[1] = (iteration % 8 == 0);
        my_net.inputs[2] = (iteration % 12 == 0);
        my_net.inputs[3] = 1.0 * rand() / RAND_MAX;

        sparknet_network_tick(&my_net, 1.0);

        actual += my_net.outputs[0] - my_net.outputs[1];

        sparknet_network_apply_reward(&my_net, 2.5 - fabsf(desired - actual), 1.0);

        printf("[Iteration %d]  Expected %s%.3f, got %s%.3f (+=%.3f -=%.3f)\n", iteration, (desired < 0 ? "" : "+"), desired, (actual < 0 ? "" : "+"), actual, my_net.outputs[0], my_net.outputs[1]);
    }

    sparknet_network_deinit(&my_net);
}
