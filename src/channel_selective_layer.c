#include "channel_selective_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_channel_selective_layer(int batch, int w, int h, int c, int* list, int list_length) {
	layer l = {0};
	l.type = CHANNEL_SELECTIVE;

	l.batch = batch;
	l.w = w;
	l.h = h;
	l.c = c;

	l.indexes = calloc(list_length, sizeof(int));
	memcpy(l.indexes, list, list_length*sizeof(int));
	l.indexes_gpu = cuda_make_int_array(l.indexes, list_length);

	l.out_w = l.w;
	l.out_h = l.h;
	l.out_c = list_length;

	l.outputs = l.out_w * l.out_h * l.out_c;
	l.inputs = l.w * l.h * l.c;
	l.delta = calloc(batch * l.outputs, sizeof(float));
	l.output = calloc(batch * l.outputs, sizeof(float));

	l.forward_gpu = forward_channel_selective_layer_gpu;
	l.backward_gpu = backward_channel_selective_layer_gpu;

	l.output_gpu = cuda_make_array(l.output, batch * l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch * l.outputs);

	return l;
}

void resize_channel_selective_layer(layer* l, int w, int h) {
	l->w = w;
	l->h = h;

	l->out_w = w;
	l->out_h = h;

	l->outputs = h*w*l->out_c;
	l->inputs = h*w*l->c;

	l->output = realloc(l->output, l->batch * l->outputs * sizeof(float));
	l->delta = realloc(l->delta, l->batch * l->outputs * sizeof(float));

	cuda_free(l->delta_gpu);
	cuda_free(l->output_gpu);

	l->output_gpu = cuda_make_array(l->output_gpu, l->batch * l->outputs);
	l->delta_gpu = cuda_make_array(l->delta_gpu, l->batch * l->outputs);
}