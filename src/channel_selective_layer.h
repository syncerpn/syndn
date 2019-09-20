#ifndef CHANNEL_SELECTIVE_LAYER_H
#define CHANNEL_SELECTIVE_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_channel_selective_layer(int batch, int w, int h, int c, int* list, int list_length);
void resize_channel_selective_layer(layer *l, int w, int h);

void forward_channel_selective_layer_gpu(const layer l, network net);
void backward_channel_selective_layer_gpu(layer l, network net);

#endif