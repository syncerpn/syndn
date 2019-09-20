#include "multibox_layer.h"
#include "activations.h"
#include "softmax_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>


layer make_multibox_layer(int batch, int n_layer, int max_boxes, int* all_layer, network* net) {
    int i;
    layer l = {0};
    l.type = MULTIBOX;
    l.cost = calloc(1, sizeof(float));
    l.batch = batch;
    l.max_boxes = max_boxes;
    l.truths = max_boxes * (4 + 1);

    l.n = n_layer;
    l.input_layers = calloc(n_layer, sizeof(int));
    l.input_sizes = calloc(n_layer, sizeof(int));
    l.classes = -1;
    l.w = 0;
    l.h = 0;
    l.c = 1;
    l.nweights = 0;
    for (i = 0; i < n_layer; ++i) {
    	layer tmp = net->layers[all_layer[i]];

    	//only accept priorbox layers
    	assert(tmp.type == PRIORBOX);
    	if (l.classes <= 0) {
    		l.h = tmp.out_h;
    		l.classes = tmp.classes;
    	} else {
    		//and they should have the same number of classes ~ same entry size
    		assert(l.classes == tmp.classes);
    		assert(l.h == tmp.out_h);
    	}

    	l.input_sizes[i] = tmp.out_w;
    	l.w += tmp.out_w;
		l.nweights += tmp.nweights;
    	l.input_layers[i] = all_layer[i];
    }

    l.inputs = l.w * l.h * l.c;

    l.out_w = l.w;
    l.out_h	= l.h;
    l.out_c = l.c;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.output = calloc(batch * l.outputs, sizeof(float));
    l.delta = calloc(batch * l.outputs, sizeof(float));

    l.output_gpu = cuda_make_array(l.output, batch * l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch * l.outputs);

    l.weights = calloc(l.nweights, sizeof(float));

    int idx = 0;
    for (i = 0; i < n_layer; ++i) {
    	layer tmp = net->layers[all_layer[i]];
		assert(tmp.nweights == l.input_sizes[i] * 4);
    	memcpy(l.weights + idx, tmp.weights, tmp.nweights * sizeof(float));
    	idx += l.input_sizes[i] * 4;
    }

    l.weights_gpu = cuda_make_array(l.weights, l.nweights);
    l.forward_gpu = forward_multibox_layer_gpu;
    l.backward_gpu = backward_multibox_layer_gpu;

    return l;
}

void resize_multibox_layer(layer *l, network* net) {
    int i;

    l->w = 0;
    l->nweights = 0;
    for (i = 0; i < l->n; ++i) {
    	layer tmp = net->layers[l->input_layers[i]];
    	l->input_sizes[i] = tmp.out_w;
    	l->w += tmp.out_w;
		l->nweights += tmp.nweights;
    }

    l->inputs = l->w * l->h * l->c;

    l->out_w = l->w;
    l->out_h = l->h;
    l->out_c = l->c;

    l->outputs = l->out_w * l->out_h * l->out_c;

    l->weights = realloc(l->weights, l->nweights * sizeof(float));
    l->output = realloc(l->output, l->batch * l->outputs * sizeof(float));
    l->delta = realloc(l->delta, l->batch * l->outputs * sizeof(float));

    cuda_free(l->weights_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);

    l->output_gpu = cuda_make_array(l->output, l->batch * l->outputs);
    l->delta_gpu = cuda_make_array(l->delta, l->batch * l->outputs);

    int idx = 0;
    for (i = 0; i < l->n; ++i) {
    	layer tmp = net->layers[l->input_layers[i]];
		assert(tmp.nweights == l->input_sizes[i] * 4);
    	memcpy(l->weights + idx, tmp.weights, tmp.nweights * sizeof(float));
		idx += l->input_sizes[i] * 4;
    }

    l->weights_gpu = cuda_make_array(l->weights, l->nweights);
}

void correct_multibox_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative) {
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        
        dets[i].bbox = b;
    }
}

int get_multibox_detections(layer l, int w, int h, int netw, int neth, float thresh, int relative, detection *dets) {
	int i,j;
	float* pred = l.output;
	int n_boxes = l.out_w;
	int classes = l.classes;
	int entry_size = l.out_h;
	int count = 0;
	for (i = 0; i < n_boxes; ++i) {
        float objectness = pred[i * entry_size + 4];
		if (objectness <= thresh) continue;
		dets[count].classes = classes;
		dets[count].bbox.x = pred[i * entry_size + 0];
		dets[count].bbox.y = pred[i * entry_size + 1];
		dets[count].bbox.w = pred[i * entry_size + 2];
		dets[count].bbox.h = pred[i * entry_size + 3];
		dets[count].objectness = objectness;
		for (j = 0; j < classes; ++j) {
            float prob = objectness * pred[i * entry_size + 5 + j];
			dets[count].prob[j] = prob > thresh ? prob : 0;
		}
		++count;
	}
    correct_multibox_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}