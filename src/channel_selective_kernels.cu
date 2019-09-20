#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "activations.h"
#include "channel_selective_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
#include "box.h"
}

__global__ void selective_channel_mapping_kernel(int batch, int w, int h, int c, int* list, int list_length, float* input, float* output) {
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= batch * list_length * w * h) return;
    int output_index = index;
    int iw = index % w;
    index = (index - iw) / w;
    int ih = index % h;
    index = (index - ih) / h;
    int ic = index % list_length;
    int ib = (index - ic) / list_length;

    int input_index = ib * w * h * c + list[ic] * w * h + ih * w + iw;
    output[output_index] = input[input_index];
}

void selective_channel_mapping_gpu(int batch, int w, int h, int c, int* list, int list_length, float* input, float* output) {
	size_t N = batch * w * h * list_length;
	selective_channel_mapping_kernel<<<cuda_gridsize(N), BLOCK>>>(batch, w, h, c, list, list_length, input, output);
	check_error(cudaPeekAtLastError());
}

__global__ void selective_channel_reversing_kernel(int batch, int w, int h, int c, int* list, int list_length, float* input, float* output, float impact) {
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= batch * list_length * w * h) return;
    int output_index = index;
    int iw = index % w;
    index = (index - iw) / w;
    int ih = index % h;
    index = (index - ih) / h;
    int ic = index % list_length;
    int ib = (index - ic) / list_length;

    int input_index = ib * w * h * c + list[ic] * w * h + ih * w + iw;
    input[input_index] += output[output_index] * impact;
}

void selective_channel_reversing_gpu(int batch, int w, int h, int c, int* list, int list_length, float* input, float* output, float impact) {
	size_t N = batch * w * h * list_length;
	selective_channel_reversing_kernel<<<cuda_gridsize(N), BLOCK>>>(batch, w, h, c, list, list_length, input, output, impact);
	check_error(cudaPeekAtLastError());
}

extern "C" void forward_channel_selective_layer_gpu(const layer l, network net) {
	selective_channel_mapping_gpu(l.batch, l.out_w, l.out_h, l.c, l.indexes_gpu, l.out_c, net.input_gpu, l.output_gpu);
}

extern "C" void backward_channel_selective_layer_gpu(layer l, network net) {
	selective_channel_reversing_gpu(l.batch, l.out_w, l.out_h, l.c, l.indexes_gpu, l.out_c, net.delta_gpu, l.output_gpu, l.impact);
}