#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "diff_layer.h"
#include "blas.h"
#include "utils.h"
#include "cuda.h"
}

__global__ void ignore_small_loss_kernel(int N, float* output, float* delta, float ignore_thresh) {
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;

    if (output[index] < ignore_thresh) {
        output[index] = 0;
        delta[index] = 0;
    }
}

void ignore_small_loss_gpu(int N, float* output, float* delta, float ignore_thresh) {
    ignore_small_loss_kernel<<<cuda_gridsize(N), BLOCK>>>(N, output, delta, ignore_thresh);
    check_error(cudaPeekAtLastError());
}

__global__ void diff_fill_skipped_channel_with_zero_kernel(int batch, int w, int h, int c, int* skip_list, int n, float* output, float* delta) {
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= batch * n * w * h) return;
    int iw = index % w;
    index = (index - iw) / w;
    int ih = index % h;
    index = (index - ih) / h;
    int ic = index % n;
    int ib = (index - ic) / n;

    int output_index = ib * w * h * c + skip_list[ic] * w * h + ih * w + iw;
    output[output_index] = 0;
    delta[output_index] = 0;
}

void diff_fill_skipped_channel_with_zero_gpu(int batch, int w, int h, int c, int* skip_list, int n, float* output, float* delta) {
    size_t N = batch * w * h * n;
    diff_fill_skipped_channel_with_zero_kernel<<<cuda_gridsize(N), BLOCK>>>(batch, w, h, c, skip_list, n, output, delta);
    check_error(cudaPeekAtLastError());
}

void forward_diff_layer_gpu(const layer l, network net) {
	*(l.cost) = 0;
    float* truth = net.layers[l.input_layers[0]].output_gpu;
    float* learn = net.layers[l.input_layers[1]].output_gpu;

    switch (l.cost_type) {
        case SMOOTH:
            smooth_l1_gpu(l.batch * l.outputs, learn, truth, l.delta_gpu, l.output_gpu);
            break;
        case L1:
            l1_gpu(l.batch * l.outputs, learn, truth, l.delta_gpu, l.output_gpu);
            break;
        case WGAN:
            wgan_gpu(l.batch * l.outputs, learn, truth, l.delta_gpu, l.output_gpu);
            break;
        case SSE:
            l2_gpu(l.batch * l.outputs, learn, truth, l.delta_gpu, l.output_gpu);
            break;
        case SYMEXP:
            symexp_gpu(l.batch * l.outputs, learn, truth, l.delta_gpu, l.output_gpu);
            break;
        case LOGCOSH:
            logcosh_gpu(l.batch * l.outputs, learn, truth, l.delta_gpu, l.output_gpu);
            break;
        case MASKED:
        case SEG:
        default:
            fprintf(stderr, "Warning: unsupported cost type; use SSE instead\n");
            l2_gpu(l.batch * l.outputs, learn, truth, l.delta_gpu, l.output_gpu);
            break;
    }
    // ignore_small_loss_gpu(l.batch * l.w * l.h * l.c, l.output_gpu, l.delta_gpu, l.ignore_thresh);
    if (l.n > 0) {
        diff_fill_skipped_channel_with_zero_gpu(l.batch, l.w, l.h, l.c, l.indexes_gpu, l.n, l.output_gpu, l.delta_gpu);
    }

    int i;
    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
    
    for (i = 0; i < l.outputs * l.batch; ++i) {
        *(l.cost) += l.output[i];
    }

	*(l.cost) /= (l.outputs * l.batch - l.batch * l.n * l.w * l.h);
    *(l.cost) *= l.impact;
}

void backward_diff_layer_gpu(layer l, network net) {

    int index = l.input_layers[1];
    float *delta = net.layers[index].delta_gpu;
	axpy_gpu(l.inputs * l.batch, l.impact, l.delta_gpu, 1, delta, 1);
}