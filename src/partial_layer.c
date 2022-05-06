#include "partial_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_partial_layer(network* net, int batch, int src_layer, int offset_w, int offset_h, int offset_c, int des_w, int des_h, int des_c) {
	layer l = {0};
	l.type = PARTIAL;
	l.input_layers = calloc(1, sizeof(int));
	l.input_layers[0] = src_layer;
	l.batch = batch;

	layer src = net->layers[src_layer];
	assert(offset_w >= 0 && offset_w < src.out_w);
	assert(offset_h >= 0 && offset_h < src.out_h);
	assert(offset_c >= 0 && offset_c < src.out_c);

	if (des_w <= 0 || des_w > (src.out_w - offset_w)) des_w	= src.out_w - offset_w;
	if (des_h <= 0 || des_h > (src.out_h - offset_h)) des_h	= src.out_h - offset_h;
	if (des_c <= 0 || des_c > (src.out_c - offset_c)) des_c	= src.out_c - offset_c;
	
	l.w = src.out_w;
	l.h = src.out_h;
	l.c = src.out_c;

	l.inputs = l.w * l.h * l.c;

	l.offset_w = offset_w;
	l.offset_h = offset_h;
	l.offset_c = offset_c;

	l.out_w = des_w;
	l.out_h = des_h;
	l.out_c = des_c;

	l.outputs = l.out_w * l.out_h * l.out_c;

    l.output = calloc(batch * l.outputs, sizeof(float));
    l.delta = calloc(batch * l.outputs, sizeof(float));

    l.output_gpu = cuda_make_array(l.output, batch * l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch * l.outputs);

    l.forward_gpu = forward_partial_layer_gpu;
    l.backward_gpu = backward_partial_layer_gpu;

    return l;
}