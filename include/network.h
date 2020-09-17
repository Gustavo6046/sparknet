#ifndef NETWORK_H_DEFINED
#define NETWORK_H_DEFINED

#include "layer.h"
#include "synapse.h"

typedef struct sparknet_network sparknet_network;

struct sparknet_network {
    sparknet_layer *layers;
    sparknet_synapse *synapses;
    float **reward_buffers;
    int *reward_synapses;
    int *propagation_targets, *propagation_sources;

    int num_reward_synapses;
    int num_layers, num_synapses;
    int input_size, output_size;

    float *inputs;
    float *outputs;
};

void sparknet_network_init(sparknet_network *target, int input_size, int output_size);
void sparknet_network_deinit(sparknet_network *target);

void sparknet_network_tick(sparknet_network *target, float time_delta);
void sparknet_network_apply_reward(sparknet_network *target, float reward, float time_delta);

int sparknet_network_add_layer(sparknet_network *target, int size, float fire_threshold, float leakage);
int sparknet_network_connect_buffers(sparknet_network *target, float* input_buffer, float* output_buffer, int input_buffer_size, int output_buffer_size, float learning_rate);
int sparknet_network_connect_layers(sparknet_network *target, int input_layer, int output_layer, float learning_rate);
int sparknet_network_connect_input(sparknet_network *target, int output_layer, float learning_rate);
int sparknet_network_connect_output(sparknet_network *target, int input_layer, float learning_rate);

#endif
