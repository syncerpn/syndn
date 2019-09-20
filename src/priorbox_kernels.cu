#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "priorbox_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

__global__ void output_prior_to_pred_kernel(float* output, float* input, float* weights, float variance_center, float variance_size, int length, int lw, int lh, int ln, int entry_size) {
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= length) return;
	int iidx = index;
	int w = index % lw;
	index = (index - w) / lw;
	int h = index % lh;
	index = (index - h) / lh;
	int i = index % entry_size;
	index = (index - i) / entry_size;
	int n = index % ln;
	int b = (index - n) / ln;
	int oidx = b * entry_size * ln * lh * lw + n * entry_size * lh * lw + h * entry_size * lw + w * entry_size + i;
	int pidx = 4 * (n * lw * lh + h * lw + w);
	if (i == 0) {
		//xc
		output[oidx] = variance_center * input[iidx] * weights[pidx + 2] + weights[pidx + 0];
	} else if (i == 1) {
		//yc
		output[oidx] = variance_center * input[iidx] * weights[pidx + 3] + weights[pidx + 1];
	} else if (i == 2) {
		//w
		output[oidx] = exp(variance_size * input[iidx]) * weights[pidx + 2];
	} else if (i == 3) {
		//h
		output[oidx] = exp(variance_size * input[iidx]) * weights[pidx + 3];
	} else if (i == 4) {
		//obj
		output[oidx] = 1.f/(1.f + expf(-input[iidx])); //logistic activated
	} else {
		//others
		output[oidx] = input[iidx];
	}
}

void output_prior_to_pred_gpu(float* output, float* input, float* weights, float variance_center, float variance_size, int batch, int lw, int lh, int ln, int entry_size) {
    int length = batch * ln * entry_size * lh * lw;
	output_prior_to_pred_kernel<<<cuda_gridsize(length), BLOCK>>>(output, input, weights, variance_center, variance_size, length, lw, lh, ln, entry_size);
    check_error(cudaPeekAtLastError());
}

__global__ void delta_remap_output_kernel(float* net_delta, float* l_delta, float* output, float* weights, float variance_center, float variance_size, int length, int lw, int lh, int ln, int entry_size, float limpact) {
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= length) return;
	int iidx = index;
	int w = index % lw;
	index = (index - w) / lw;
	int h = index % lh;
	index = (index - h) / lh;
	int i = index % entry_size;
	index = (index - i) / entry_size;
	int n = index % ln;
	int b = (index - n) / ln;
	int oidx = b * entry_size * ln * lh * lw + n * entry_size * lh * lw + h * entry_size * lw + w * entry_size + i;
	int pidx = 4 * (n * lw * lh + h * lw + w);
	if (i == 0) {
		//xc
		net_delta[iidx] += limpact * l_delta[oidx] * variance_center * weights[pidx + 2];
	} else if (i == 1) {
		//yc
		net_delta[iidx] += limpact * l_delta[oidx] * variance_center * weights[pidx + 3];
	} else if (i == 2) {
		//w
		net_delta[iidx] += limpact * l_delta[oidx] * output[oidx] * variance_size;
	} else if (i == 3) {
		//h
		net_delta[iidx] += limpact * l_delta[oidx] * output[oidx] * variance_size;
	} else if (i == 4) {
		//obj
		net_delta[iidx] += limpact * l_delta[oidx] * output[oidx] * (1 - output[oidx]);
	} else {
		//others
		net_delta[iidx] += limpact * l_delta[oidx];
	}
}

void delta_remap_output_gpu(float* net_delta, float* l_delta, float* output, float* weights, float variance_center, float variance_size, int batch, int lw, int lh, int ln, int entry_size, float limpact) {
    int length = batch * ln * entry_size * lh * lw;
	delta_remap_output_kernel<<<cuda_gridsize(length), BLOCK>>>(net_delta, l_delta, output, weights, variance_center, variance_size, length, lw, lh, ln, entry_size, limpact);
    check_error(cudaPeekAtLastError());
}

void forward_priorbox_layer_gpu(const layer l, network net) {
	int entry_size = 4 + 1 + l.classes;
    output_prior_to_pred_gpu(l.output_gpu, net.input_gpu, l.weights_gpu, l.variance_center, l.variance_size, l.batch, l.w, l.h, l.n, entry_size);
}

void backward_priorbox_layer_gpu(layer l, network net) {
	int entry_size = 4 + 1 + l.classes;
    delta_remap_output_gpu(net.delta_gpu, l.delta_gpu, l.output_gpu, l.weights_gpu, l.variance_center, l.variance_size, l.batch, l.w, l.h, l.n, entry_size, l.impact);
}