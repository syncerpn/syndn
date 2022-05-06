#ifndef PARTIAL_LAYER_H
#define PARTIAL_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_partial_layer(network* net, int batch, int src_layer, int offset_w, int offset_h, int offset_c, int des_w, int des_h, int des_c);
void forward_partial_layer_gpu(const layer l, network net);
void backward_partial_layer_gpu(layer l, network net);

#endif