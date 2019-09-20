#ifndef PRIORBOX_LAYER_H
#define PRIORBOX_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_priorbox_layer(int batch, int input_w, int input_h, int input_c, int net_w, int net_h, float min_size, float max_size, int n_ar, float* all_ar, int flip, int clip, float variance_center, float variance_size, int classes);
void create_priorbox(layer l, int net_w, int net_h);
void resize_priorbox_layer(layer *l, int w, int h, int net_w, int net_h);

void forward_priorbox_layer_gpu(const layer l, network net);
void backward_priorbox_layer_gpu(layer l, network net);

#endif