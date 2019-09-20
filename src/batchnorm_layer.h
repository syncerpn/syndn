#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

layer make_batchnorm_layer(int batch, int w, int h, int c);
void resize_batchnorm_layer(layer *l, int w, int h);
void forward_batchnorm_layer_gpu(layer l, network net);
void backward_batchnorm_layer_gpu(layer l, network net);
void pull_batchnorm_layer(layer l);
void push_batchnorm_layer(layer l);

//nghiant
void make_scale_new_factor(float* factor, float *scale, float* var, int batch, int filters, fixed_point_scheme fps, float* scale_new);
void make_scale_new_scalar(float scalar, float *scale, float* var, int batch, int filters, fixed_point_scheme fps, float* scale_new);
void make_bias_new_factor(float* bias, float* factor, float *scale, float* var, int batch, int filters, fixed_point_scheme fps, float* bias_new);
//nghiant_end

#endif