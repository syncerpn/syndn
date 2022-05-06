#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "partial_layer.h"
#include "blas.h"
#include "utils.h"
#include "cuda.h"
}

__global__ void fill_partial_kernel(int batch, int w, int h, int c, int offset_w, int offset_h, int offset_c, int out_w, int out_h, int out_c, float* input, float* output, int is_forward) {
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= batch * out_w * out_h * out_c) return;
    int ow = index % out_w;
    index = (index - ow) / out_w;
    int oh = index % out_h;
    index = (index - oh) / out_h;
    int oc = index % out_c;
    int ob = (index - oc) / out_c;
    
    int ib = ob;
    int iw = ow + offset_w;
    int ih = oh + offset_h;
    int ic = oc + offset_c;

    index = ob*out_w*out_h*out_c + oc*out_w*out_h + oh*out_w + ow;
    int input_index = ib*w*h*c + ic*w*h + ih*w + iw;
    if (is_forward) {
        output[index] = input[input_index];
    } else {
        input[input_index] += output[index];
    }
}

void fill_partial_gpu(int batch, int w, int h, int c, int offset_w, int offset_h, int offset_c, int out_w, int out_h, int out_c, float* input, float* output, int is_forward) {
    size_t N = batch * out_w * out_h * out_c;
    fill_partial_kernel<<<cuda_gridsize(N), BLOCK>>>(batch, w, h, c, offset_w, offset_h, offset_c, out_w, out_h, out_c, input, output, is_forward);
    check_error(cudaPeekAtLastError());
}

void forward_partial_layer_gpu(const layer l, network net) {
    float* input = net.layers[l.input_layers[0]].output_gpu;
    fill_partial_gpu(l.batch, l.w, l.h, l.c, l.offset_w, l.offset_h, l.offset_c, l.out_w, l.out_h, l.out_c, input, l.output_gpu, 1);
}

void backward_partial_layer_gpu(layer l, network net) {
    float* input_delta = net.layers[l.input_layers[0]].delta_gpu;
    scal_gpu(l.batch * l.outputs, l.impact, l.delta_gpu, 1);
    fill_partial_gpu(l.batch, l.w, l.h, l.c, l.offset_w, l.offset_h, l.offset_c, l.out_w, l.out_h, l.out_c, input_delta, l.delta_gpu, 0);
}