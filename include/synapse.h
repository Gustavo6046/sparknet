#ifndef SYNAPSE_H_DEFINED
#define SYNAPSE_H_DEFINED


typedef struct sparknet_synapse sparknet_synapse;

struct sparknet_synapse {
    float *in_layer, *out_layer;
    int in_size, out_size;

    float *weights; // in size * out size... high ouch-ability levels detected!
    
    // learning parameters
    float learning_rate;
};

void sparknet_synapse_init(sparknet_synapse *target, int in_size, int out_size, float *in_layer, float *out_layer, float learning_rate);
void sparknet_synapse_deinit(sparknet_synapse *target);
void sparknet_synapse_tick(sparknet_synapse *target, float time_delta);
void sparknet_synapse_learn(sparknet_synapse *target, float *in_rewards, float *out_reward_propagation, float time_delta);
void sparknet_synapse_learn_static(sparknet_synapse *target, float reward, float *out_reward_propagation, float time_delta);

#endif // SYNAPSE_H_DEFINED
