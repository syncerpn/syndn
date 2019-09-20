#include "priorbox_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>


layer make_priorbox_layer(int batch, int input_w, int input_h, int input_c, int net_w, int net_h, float min_size, float max_size, int n_ar, float* all_ar, int flip, int clip, float variance_center, float variance_size, int classes) {
    int i;
    layer l = {0};
    l.type = PRIORBOX;

    l.batch = batch;
    l.h = input_h;
    l.w = input_w;
    l.c = input_c;
    
    l.prior_clip = clip;
    l.prior_flip = flip;
    l.variance_center = variance_center;
    l.variance_size = variance_size;

    l.min_size = min_size;
    l.max_size = max_size;

    l.classes = classes;
    l.nbiases = n_ar;
    l.biases = calloc(l.nbiases, sizeof(float));

    for (i = 0; i < n_ar; ++i) {
        l.biases[i] = all_ar[i];
    }
    int n_default_box = 1 + (l.min_size != l.max_size) + n_ar;
    if (flip) {
    	n_default_box += n_ar;
    }

    l.n = n_default_box;

    assert(l.c == n_default_box * (4 + 1 + classes)); //classes + background + delta location

    l.out_w = l.w * l.h * n_default_box; //classes + background + delta location + default_box
    l.out_h = (4 + 1 + classes);
    l.out_c = 1;

    l.outputs = l.out_w * l.out_h * l.out_c;
    l.output = calloc(l.batch * l.outputs, sizeof(float));
    l.output_gpu = cuda_make_array(l.output, batch * l.outputs);

    l.inputs = l.outputs;
    l.delta = calloc(l.batch * l.outputs, sizeof(float));
    l.delta_gpu = cuda_make_array(l.delta, l.batch * l.outputs);

    l.nweights = 4 * n_default_box * l.w * l.h;
    l.weights = calloc(l.nweights, sizeof(float));
    create_priorbox(l, net_w, net_h);
    l.weights_gpu = cuda_make_array(l.weights, l.nweights);

    l.forward_gpu = forward_priorbox_layer_gpu;
    l.backward_gpu = backward_priorbox_layer_gpu;
    
    return l;
}

void create_priorbox(layer l, int img_width, int img_height) {
	int h, w, s;

	float step_w = (float)img_width / (float)l.w;
	float step_h = (float)img_height / (float)l.h;

	int i, n;
	for (h = 0; h < l.h; ++h) {
		for (w = 0; w < l.w; ++w) {
			n = 0;
			i = 4 * (n * l.w * l.h + h * l.w + w);
			float center_x = (w + 0.5) * step_w;
			float center_y = (h + 0.5) * step_h;
			float box_width, box_height;

			//output 0: default square box
			box_width = l.min_size;
			box_height = l.min_size;

			l.weights[i++] = (center_x - box_width / 2.) / img_width; //xmin
			l.weights[i++] = (center_y - box_height / 2.) / img_height; //ymin
			l.weights[i++] = (center_x + box_width / 2.) / img_width; //xmax
			l.weights[i++] = (center_y + box_height / 2.) / img_height; //ymax

			//output 1: square box of sqrt(min_size * max_size)
			if (l.max_size != l.min_size) {
				++n;
				i = 4 * (n * l.w * l.h + h * l.w + w);
				box_width = sqrt(l.max_size * l.min_size);
				box_height = box_width;

				l.weights[i++] = (center_x - box_width / 2.) / img_width; //xmin
				l.weights[i++] = (center_y - box_height / 2.) / img_height; //ymin
				l.weights[i++] = (center_x + box_width / 2.) / img_width; //xmax
				l.weights[i++] = (center_y + box_height / 2.) / img_height; //ymax
			}

			//output 2 ~ end:
			for (s = 0; s < l.nbiases; ++s) {
				++n;
				i = 4 * (n * l.w * l.h + h * l.w + w);
				box_width = l.min_size / sqrt(l.biases[s]);
				box_height = l.min_size * sqrt(l.biases[s]);

				l.weights[i++] = (center_x - box_width / 2.) / img_width; //xmin
				l.weights[i++] = (center_y - box_height / 2.) / img_height; //ymin
				l.weights[i++] = (center_x + box_width / 2.) / img_width; //xmax
				l.weights[i++] = (center_y + box_height / 2.) / img_height; //ymax
				
				if (l.prior_flip) {
					++n;
					i = 4 * (n * l.w * l.h + h * l.w + w);
					float tmp = box_width;
					box_width = box_height;
					box_height = tmp;

					l.weights[i++] = (center_x - box_width / 2.) / img_width; //xmin
					l.weights[i++] = (center_y - box_height / 2.) / img_height; //ymin
					l.weights[i++] = (center_x + box_width / 2.) / img_width; //xmax
					l.weights[i++] = (center_y + box_height / 2.) / img_height; //ymax
				}
			}
		}
	}

	if (l.prior_clip) {
		for (i = 0; i < l.nweights; ++i) {
			l.weights[i] = l.weights[i] < 0 ? 0 : (l.weights[i] > 1 ? 1 : l.weights[i]);
		}
	}
	for (i = 0; i < l.nweights; i = i + 4) {
		float xmin = l.weights[i+0];
		float ymin = l.weights[i+1];
		float xmax = l.weights[i+2];
		float ymax = l.weights[i+3];
		l.weights[i+0] = (xmin + xmax) / 2;
		l.weights[i+1] = (ymin + ymax) / 2;
		l.weights[i+2] = xmax - xmin;
		l.weights[i+3] = ymax - ymin;
	}
}

void resize_priorbox_layer(layer *l, int w, int h, int net_w, int net_h) {
	l->w = w;
	l->h = h;

    l->out_w = l->w * l->h * l->n;

    l->outputs = l->out_w * l->out_h * l->out_c;
    l->inputs = l->outputs;
    l->nweights = 4 * l->n * l->w * l->h;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));
    l->weights = realloc(l->weights, l->nweights*sizeof(float));

    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->weights_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
    create_priorbox(*l, net_w, net_h);
    l->weights_gpu =   cuda_make_array(l->weights, l->nweights);
}