#include <stdlib.h>
#include <math.h>

#include "synapse.h"



void sparknet_synapse_init(sparknet_synapse *target, int in_size, int out_size, float *in_layer, float *out_layer, float learning_rate) {
    target->in_size = in_size;
    target->out_size = out_size;

    target->weights = (float *) calloc(sizeof(float), in_size * out_size);

    for (int w = 0; w < in_size * out_size; w++) {
        target->weights[w] = (1.0 * rand() / RAND_MAX) * 5.0 - 2.5;
    }

    target->in_layer = in_layer;
    target->out_layer = out_layer;
    target->learning_rate = learning_rate;
}

void sparknet_synapse_deinit(sparknet_synapse *target) {
    free(target->weights);
}

void sparknet_synapse_tick(sparknet_synapse *target, float time_delta) {
    int weights_index = 0;
    float accum;

    int out_layer_size = target->out_size;

    for (int out_index = 0; out_index < out_layer_size; out_index++) {
        accum = 0.0;

        int in_layer_size = target->in_size;

        for (int in_index = 0; in_index < in_layer_size; in_index++) {
            accum += target->in_layer[in_index] * target->weights[weights_index];
            target->in_layer[in_index] = 0;

            weights_index++;
        }

        target->out_layer[out_index] += accum;
    }
}

void sparknet_synapse_learn(sparknet_synapse *target, float *in_rewards, float *out_reward_propagation, float time_delta) {
    int weights_index = 0;

    for (int out_index = 0; out_index < target->out_size; out_index++) {
        float orig_w = target->weights[weights_index];
        float delta = in_rewards[out_index] * sqrtf(fabsf(orig_w)) * ((0.0 < orig_w) - (orig_w < 0.0));

        int in_layer_size = target->in_size;

        if (out_reward_propagation != 0) {
            float delta_propagated = delta / in_layer_size;

            for (int in_index = 0; in_index < in_layer_size; in_index++) {
                out_reward_propagation[in_index] += delta_propagated;
                target->weights[weights_index] += delta * powf(target->learning_rate, time_delta);

                weights_index++;
            }
        }

        else {
            for (int in_index = 0; in_index < in_layer_size; in_index++) {
                target->weights[weights_index] += delta * powf(target->learning_rate, time_delta);

                weights_index++;
            }
        }
    }

}

void sparknet_synapse_learn_static(sparknet_synapse *target, float reward, float *out_reward_propagation, float time_delta) {
    int weights_index = 0;

    for (int out_index = 0; out_index < target->out_size; out_index++) {
        float orig_w = target->weights[weights_index];
        float delta = reward * sqrtf(fabsf(orig_w)) * ((0.0 < orig_w) - (orig_w < 0.0));

        int in_layer_size = target->in_size;

        if (out_reward_propagation != 0) {
            float delta_propagated = delta / in_layer_size;

            for (int in_index = 0; in_index < in_layer_size; in_index++) {
                out_reward_propagation[in_index] += delta_propagated;
                target->weights[weights_index] += delta * powf(target->learning_rate, time_delta);

                weights_index++;
            }
        }

        else {
            for (int in_index = 0; in_index < in_layer_size; in_index++) {
                target->weights[weights_index] += delta * powf(target->learning_rate, time_delta);

                weights_index++;
            }
        }
    }
}
