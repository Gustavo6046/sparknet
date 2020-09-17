#include <math.h>
#include <stdlib.h>

#include "layer.h"



float sparknet_layer_sigmoid(float f) {
    float ef = expf(f);

    return ef / (ef + 1);
}

void sparknet_layer_init(sparknet_layer *target, int desired_size, float fire_threshold, float leakage) {
    target->size = desired_size;
    target->values = (float *) calloc(sizeof(float), desired_size);
    target->inputs = (float *) calloc(sizeof(float), desired_size);
    target->outputs = (float *) calloc(sizeof(float), desired_size);

    target->fire_threshold = fire_threshold;
    target->leakage = leakage;
}

void sparknet_layer_deinit(sparknet_layer *target) {
    free(target->values);
    free(target->inputs);
    free(target->outputs);
}

void sparknet_layer_reset_inputs(sparknet_layer *target) {
    for (int i = 0; i < target->size; i++) {
        target->inputs[i] = 0;
    }
}

void sparknet_layer_reset_outputs(sparknet_layer *target) {
    for (int i = 0; i < target->size; i++) {
        target->outputs[i] = 0;
    }
}

void sparknet_layer_tick(sparknet_layer *target, float time_delta) {
    float tick_leakage = 1.0f + powf(target->leakage, time_delta);
    int do_fire, neg_value;

    for (int which = 0; which < target->size; which++) {
        if (target->values[which] != 0) {
            target->values[which] /= tick_leakage;
        }

        // TODO: think about perhaps making branchless (ultimate optimization)
        // later - assuming CPU cache can still be hurt by if's that don't run!
        if (target->inputs[which] != 0) {
            target->values[which] += sparknet_layer_sigmoid(target->inputs[which]);
            target->inputs[which] = 0;

            neg_value = (target->values[which] < 0);

            target->values[which] -= target->values[which] * neg_value;

            do_fire = (!neg_value) && (target->values[which] > target->fire_threshold);

            target->outputs[which] = do_fire * target->values[which] / target->fire_threshold;
            target->values[which] -= do_fire * target->values[which];
        }
    }
}
