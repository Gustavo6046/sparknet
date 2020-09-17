#include <stdlib.h>

#include "network.h"


void sparknet_network_init(sparknet_network *target, int input_size, int output_size) {
    target->layers = 0;
    target->synapses = 0;

    target->reward_buffers = 0;
    target->reward_synapses = 0;
    
    target->propagation_targets = 0;
    target->propagation_sources = 0;

    target->num_layers = 0;
    target->num_synapses = 0;
    target->num_reward_synapses = 0;

    target->input_size = input_size;
    target->output_size = output_size;

    target->inputs = calloc(sizeof(float), input_size);
    target->outputs = calloc(sizeof(float), output_size);
}

void sparknet_network_deinit(sparknet_network *target) {
    for (int li = 0; li < target->num_layers; li++) {
        sparknet_layer_deinit(target->layers + li);
        free(target->reward_buffers[li]);
    }

    for (int si = 0; si < target->num_synapses; si++) {
        sparknet_synapse_deinit(target->synapses + si);
    }

    free(target->reward_buffers);
    free(target->reward_synapses);
    free(target->layers);
    free(target->synapses);

    free(target->propagation_targets);
    free(target->propagation_sources);

    free(target->inputs);
    free(target->outputs);
}

void sparknet_network_tick(sparknet_network *target, float time_delta) {
    int i;

    for (i = 0; i < target->num_layers; i++) {
        sparknet_layer_tick(&target->layers[i], time_delta);
    }

    for (i = 0; i < target->num_layers; i++) {
        sparknet_layer_reset_inputs(&target->layers[i]);
    }

    for (i = 0; i < target->num_synapses; i++) {
        sparknet_synapse_tick(&target->synapses[i], time_delta);
    }

    for (i = 0; i < target->num_layers; i++) {
        sparknet_layer_reset_outputs(&target->layers[i]);
    }
}

static void sparknet_network_clear_reward_buffers(sparknet_network *target) {
    int i, j;

    for (i = 0; i < target->num_layers; i++) {
        for (j = 0; j < target->layers[i].size; j++) {
            target->reward_buffers[i][j] = 0;
        }
    }
}

#define reward_buffer_for(what) (( (what) == (-1) ) ? (0) : (((target)->reward_buffers)[(what)]) )
void sparknet_network_apply_reward(sparknet_network *target, float reward, float time_delta) {
    sparknet_network_clear_reward_buffers(target);

    unsigned char *propag_found = (unsigned char *) calloc(target->num_layers, sizeof(unsigned char));
    int propagated = 0;

    for (int i = 0; i < target->num_reward_synapses; i++) {
        int synapse_index = target->reward_synapses[i];

        sparknet_synapse_learn_static(&target->synapses[synapse_index], reward, reward_buffer_for(target->propagation_targets[synapse_index]), time_delta);

        if (target->propagation_targets[synapse_index] != -1) {
            propag_found[target->propagation_targets[synapse_index]] = 1;
        }
    }

    while (propagated < target->num_layers) {
        for (int i = 0; i < target->num_synapses; i++) {
            if (target->propagation_sources[i] != -1 && propag_found[target->propagation_sources[i]] == 1) {
                sparknet_synapse_learn(&target->synapses[i], target->reward_buffers[target->propagation_sources[i]], reward_buffer_for(target->propagation_targets[i]), time_delta);
                
                if (target->propagation_sources[i] != -1) {
                    propag_found[target->propagation_sources[i]] = 2;
                }

                propagated++;
                
                if (target->propagation_targets[i] != -1 && propag_found[target->propagation_targets[i]] == 0) {
                    propag_found[target->propagation_targets[i]] = 1;
                }
            }
        }
    }

    free(propag_found);
}

int sparknet_network_add_layer(sparknet_network *target, int size, float fire_threshold, float leakage) {
    if (fire_threshold == 0) fire_threshold = 1.0;
    if (leakage == 0) leakage = 0.001;

    int index = target->num_layers;

    target->layers = realloc(target->layers, sizeof(sparknet_layer) * ++(target->num_layers));
    target->reward_buffers = realloc(target->reward_buffers, sizeof(float *) * target->num_layers);
    
    sparknet_layer_init(target->layers + index, size, fire_threshold, leakage);
    target->reward_buffers[index] = malloc(sizeof(float) * size);

    return index;
}

int sparknet_network_connect_buffers(sparknet_network *target, float* input_buffer, float* output_buffer, int input_buffer_size, int output_buffer_size, float learning_rate) {
    if (learning_rate == 0) learning_rate = 0.3;

    int index = target->num_synapses;

    target->synapses = realloc(target->synapses, sizeof(sparknet_synapse) * ++(target->num_synapses));

    target->propagation_targets = realloc(target->propagation_targets, sizeof(int) * target->num_synapses);
    target->propagation_sources = realloc(target->propagation_sources, sizeof(int) * target->num_synapses);

    target->propagation_targets[index] = -1;
    target->propagation_sources[index] = -1;

    sparknet_synapse_init(&target->synapses[index], input_buffer_size, output_buffer_size, input_buffer, output_buffer, learning_rate);

    return index;
}

int sparknet_network_connect_layers(sparknet_network *target, int input_layer, int output_layer, float learning_rate) {
    int index = sparknet_network_connect_buffers(target, target->layers[input_layer].outputs, target->layers[output_layer].inputs, target->layers[input_layer].size, target->layers[output_layer].size, learning_rate);

    target->propagation_targets[index] = input_layer;
    target->propagation_sources[index] = output_layer;

    return index;
}

int sparknet_network_connect_input(sparknet_network *target, int output_layer, float learning_rate) {
    int index = sparknet_network_connect_buffers(target, target->inputs, target->layers[output_layer].inputs, target->input_size, target->layers[output_layer].size, learning_rate);

    target->propagation_sources[index] = output_layer;

    return index;
}

int sparknet_network_connect_output(sparknet_network *target, int input_layer, float learning_rate) {
    int index = sparknet_network_connect_buffers(target, target->layers[input_layer].outputs, target->outputs, target->layers[input_layer].size, target->output_size, learning_rate);

    target->reward_synapses = realloc(target->reward_synapses, sizeof(int) * (++target->num_reward_synapses));
    target->reward_synapses[target->num_reward_synapses - 1] = index;

    target->propagation_targets[index] = input_layer;

    return index;
}
