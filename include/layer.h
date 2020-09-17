#ifndef LAYER_H_DEFINED
#define LAYER_H_DEFINED

typedef struct sparknet_layer sparknet_layer;

struct sparknet_layer {
    int size;
    float fire_threshold, leakage;
    float *values;
    float *inputs;
    float *outputs;
};

void sparknet_layer_init(sparknet_layer *target, int desired_size, float fire_threshold, float leakage);
void sparknet_layer_deinit(sparknet_layer *target);
void sparknet_layer_tick(sparknet_layer *target, float time_delta);

void sparknet_layer_reset_inputs(sparknet_layer *target);
void sparknet_layer_reset_outputs(sparknet_layer *target);

#endif // LAYER_H_DEFINED
