#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "multibox_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
#include "box.h"
}

__device__ float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

__device__ float box_intersection(float px, float py, float pw, float ph, float tx, float ty, float tw, float th)
{
    float w = overlap(px, pw, tx, tw);
    float h = overlap(py, ph, ty, th);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

__device__ float box_union(float px, float py, float pw, float ph, float tx, float ty, float tw, float th)
{
    float i = box_intersection(px, py, pw, ph, tx, ty, tw, th);
    float u = pw*ph + tw*th - i;
    return u;
}

__device__ float box_iou_kernel(float px, float py, float pw, float ph, float tx, float ty, float tw, float th) {
    return box_intersection(px, py, pw, ph, tx, ty, tw, th) / box_union(px, py, pw, ph, tx, ty, tw, th);
}

__global__ void delta_multibox_a_kernel(float* output, float* weights, float* delta, float* truth, int ltruths,
    int batch, int lw, int lh, int max_boxes, int classes,
    float class_scale, float coord_scale, float object_scale) {

    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= batch) return;
    int b = index;
    int t, k, n;
    float dx, dy, dw, dh, px, py, pw, ph, tx, ty, tw, th;
    float best_iou;
    int best_n;
    int box_index, obj_index, cls_index;

    for (t = 0; t < max_boxes; ++t) {

        tx = truth[t * 5 + b * ltruths + 0];
        ty = truth[t * 5 + b * ltruths + 1];
        tw = truth[t * 5 + b * ltruths + 2];
        th = truth[t * 5 + b * ltruths + 3];

        best_iou = 0;
        best_n = -1;

        if (tx == 0) break;

        for (n = 0; n < lw; ++n) {
            dx = weights[n * 4 + 0];
            dy = weights[n * 4 + 1];
            dw = weights[n * 4 + 2];
            dh = weights[n * 4 + 3];

            float iou = box_iou_kernel(dx, dy, dw, dh, tx, ty, tw, th);

            if (iou > best_iou) {
                best_iou = iou;
                best_n = n;
            }
        }

        if (best_n >= 0) {
            box_index = b * lh * lw + best_n * lh + 0;
            obj_index = b * lh * lw + best_n * lh + 4;
            cls_index = b * lh * lw + best_n * lh + 5;

            px = output[box_index + 0];
            py = output[box_index + 1];
            pw = output[box_index + 2];
            ph = output[box_index + 3];

            delta[obj_index] += object_scale * (1 - output[obj_index]);
            int truth_class = (int)(truth[t * 5 + b * ltruths + 4]);

            for (k = 0; k < classes; ++k) delta[cls_index + k] += class_scale * (((k == truth_class) ? 1 : 0) - output[cls_index + k]);

            float d[4];
            d[0] = px - tx;
            d[1] = py - ty;
            d[2] = pw - tw;
            d[3] = ph - th;

            for (k = 0; k < 4; ++k) delta[box_index + k] += -coord_scale * 2 * d[k];
        }
    }
}

void delta_multibox_a_gpu(float* output, float* weights, float* delta, float* truth, int ltruths,
    int batch, int lw, int lh, int max_boxes, int classes,
    float class_scale, float coord_scale, float object_scale) {

    delta_multibox_a_kernel<<<cuda_gridsize(batch), BLOCK>>>(output, weights, delta, truth, ltruths, batch, lw, lh, max_boxes, classes, class_scale, coord_scale, object_scale);
    check_error(cudaPeekAtLastError());
}

//nghiant: seems ok now
__global__ void delta_multibox_b_kernel(float* output, float* weights, float* delta, float* truth, int ltruths,
    int batch, int lw, int lh, int max_boxes, int classes, float ignore_thresh, float truth_thresh,
    float class_scale, float coord_scale, float noobject_scale, float object_scale) {

	int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= batch * lw) return;

	int i = index % lw;
	int b = (index - i) / lw;
	int t, k;

    int box_index = b * lh * lw + i * lh + 0;
    int obj_index = b * lh * lw + i * lh + 4;
    int cls_index = b * lh * lw + i * lh + 5;

    if (delta[obj_index]) return; //already assigned something here
	
	float px = output[box_index + 0];
	float py = output[box_index + 1];
	float pw = output[box_index + 2];
	float ph = output[box_index + 3];
	
	float dx = weights[i * 4 + 0];
	float dy = weights[i * 4 + 1];
	float dw = weights[i * 4 + 2];
	float dh = weights[i * 4 + 3];
    
    float best_iou = 0;
    int best_t = -1;

    for (t = 0; t < max_boxes; ++t) {
        float tx = truth[t * 5 + b * ltruths + 0];
        float ty = truth[t * 5 + b * ltruths + 1];
        float tw = truth[t * 5 + b * ltruths + 2];
        float th = truth[t * 5 + b * ltruths + 3];
        
        if (tx == 0) break;

        float iou = box_iou_kernel(dx, dy, dw, dh, tx, ty, tw, th);
        if (iou > best_iou) {
            best_iou = iou;
            best_t = t;
        }
    }

    delta[obj_index] = noobject_scale * (0 - output[obj_index]);
    if (best_iou > ignore_thresh) delta[obj_index] = 0;
    if (best_iou > truth_thresh) {
        delta[obj_index] = object_scale * (1 - output[obj_index]);
        int truth_class = (int)(truth[best_t * 5 + b * ltruths + 4]);

        for (k = 0; k < classes; ++k) delta[cls_index + k] = class_scale * (((k == truth_class) ? 1 : 0) - output[cls_index + k]);

        float tx = truth[best_t * 5 + b * ltruths + 0];
        float ty = truth[best_t * 5 + b * ltruths + 1];
        float tw = truth[best_t * 5 + b * ltruths + 2];
        float th = truth[best_t * 5 + b * ltruths + 3];

        float d[4];
        d[0] = px - tx;
        d[1] = py - ty;
        d[2] = pw - tw;
        d[3] = ph - th;

		for (k = 0; k < 4; ++k) delta[box_index + k] = -coord_scale * 2 * d[k];
    }
}

void delta_multibox_b_gpu(float* output, float* weights, float* delta, float* truth, int ltruths,
    int batch, int lw, int lh, int max_boxes, int classes, float ignore_thresh, float truth_thresh,
    float class_scale, float coord_scale, float noobject_scale, float object_scale) {

    delta_multibox_b_kernel<<<cuda_gridsize(batch * lw), BLOCK>>>(output, weights, delta, truth, ltruths, batch, lw, lh, max_boxes, classes, ignore_thresh, truth_thresh, class_scale, coord_scale, noobject_scale, object_scale);
    check_error(cudaPeekAtLastError());
}

__global__ void softmax_multibox_kernel(float* input, float* output, int batch, int lw, int lh) {
	int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= batch * lw) return;
	int n = lh - 5;
	int i;
	float sum = 0;
	float largest = -INFINITY;
	for (i = 0; i < n; ++i) {
		largest = (input[index * lh + 5 + i] > largest) ? input[index * lh + 5 + i] : largest;
	}
	for (i = 0; i < n; ++i) {
		float e = expf(input[index * lh + 5 + i] - largest);
		sum += e;
		output[index * lh + 5 + i] = e;
	}
	for (i = 0; i < n; ++i) {
		output[index * lh + 5 + i] /= sum;
	}
}

void softmax_multibox_gpu(float* input, float* output, int batch, int lw, int lh) {
	softmax_multibox_kernel<<<cuda_gridsize(lw * batch), BLOCK>>>(input, output, batch, lw, lh);
	check_error(cudaPeekAtLastError());
}

extern "C" void forward_multibox_layer_gpu(const layer l, network net) {
	int i, b;
	for (b = 0; b < l.batch; ++b) {
		int idx = b * l.outputs;
		for (i = 0; i < l.n; ++i) {
			copy_gpu(l.input_sizes[i] * l.h, net.layers[l.input_layers[i]].output_gpu + b * net.layers[l.input_layers[i]].outputs, 1, l.output_gpu + idx, 1);
			idx += l.input_sizes[i] * l.out_h;
		}
	}
	softmax_multibox_gpu(l.output_gpu, l.output_gpu, l.batch, l.w, l.h);

    if (!net.train || l.onlyforward) {
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }
    
    *(l.cost) = 0;

	delta_multibox_a_gpu(l.output_gpu, l.weights_gpu, l.delta_gpu, net.truth_gpu, l.truths, l.batch, l.w, l.h, l.max_boxes, l.classes, l.class_scale, l.coord_scale, l.object_scale);
    delta_multibox_b_gpu(l.output_gpu, l.weights_gpu, l.delta_gpu, net.truth_gpu, l.truths, l.batch, l.w, l.h, l.max_boxes, l.classes, l.ignore_thresh, l.truth_thresh, l.class_scale, l.coord_scale, l.noobject_scale, l.object_scale);

    cuda_pull_array(l.delta_gpu, l.delta, l.batch * l.outputs);
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2) * l.impact;
}

extern "C" void backward_multibox_layer_gpu(layer l, network net) {
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *delta = net.layers[index].delta_gpu;
        int input_size = l.input_sizes[i] * l.h;
        for(j = 0; j < l.batch; ++j){
            axpy_gpu(input_size, l.impact, l.delta_gpu + offset + j*l.outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
}