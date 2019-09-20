#ifndef MULTIBOX_LAYER_H
#define MULTIBOX_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_multibox_layer(int batch, int n_layer, int max_boxes, int* all_layer, network* net);
void resize_multibox_layer(layer *l, network* net);

void forward_multibox_layer_gpu(const layer l, network net);
void backward_multibox_layer_gpu(layer l, network net);

#endif