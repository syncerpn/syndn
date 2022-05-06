#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
}

__global__ void make_shifting_weights_max_kernel(float *weights, int n, int size, float *tran_weights, float* q_coeff, int n_coeff, int zero_center)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float max = 0;
    for(i = 0; i < size; ++i){
        max = max < fabsf(weights[f*size + i]) ? fabsf(weights[f*size + i]) : max;
    }
    
    for(i = 0; i < size; ++i){
        if (weights[f*size + i] > 0) {
            float expo_factor = roundf(log2f(weights[f*size + i] / max));
            if (expo_factor > q_coeff[0]) expo_factor = q_coeff[0];
            else if (expo_factor < q_coeff[n_coeff-1]) {
                if (zero_center) {
                    tran_weights[f*size + i] = 0;
                    continue;
                } else {
                    expo_factor = q_coeff[n_coeff-1];
                }
            }
            tran_weights[f*size + i] = max * powf(2.f, expo_factor);
        } else {
            float expo_factor = roundf(log2f(-weights[f*size + i] / max));
            if (expo_factor > q_coeff[0]) expo_factor = q_coeff[0];
            else if (expo_factor < q_coeff[n_coeff-1]) {
                if (zero_center) {
                    tran_weights[f*size + i] = 0;
                    continue;
                } else {
                    expo_factor = q_coeff[n_coeff-1];
                }
            }
            tran_weights[f*size + i] = -max * powf(2.f, expo_factor);
        }
    }
}

void make_shifting_weights_max_gpu(float *weights, int n, int size, float *tran_weights, float* q_coeff, int n_coeff, int zero_center)
{
    make_shifting_weights_max_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, tran_weights, q_coeff, n_coeff, zero_center);
    check_error(cudaPeekAtLastError());
}

__global__ void make_shifting_weights_mean_kernel(float *weights, int n, int size, float *tran_weights, float* q_coeff, int n_coeff, int zero_center)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
        mean += fabsf(weights[f*size + i]);
    }
    mean = mean / size;
    
    for(i = 0; i < size; ++i){
        if (weights[f*size + i] > 0) {
            float expo_factor = roundf(log2f(weights[f*size + i] / mean));
            if (expo_factor > q_coeff[0]) expo_factor = q_coeff[0];
            else if (expo_factor < q_coeff[n_coeff-1]) {
                if (zero_center) {
                    tran_weights[f*size + i] = 0;
                    continue;
                } else {
                    expo_factor = q_coeff[n_coeff-1];
                }
            }
            tran_weights[f*size + i] = mean * powf(2.f, expo_factor);
        } else {
            float expo_factor = roundf(log2f(-weights[f*size + i] / mean));
            if (expo_factor > q_coeff[0]) expo_factor = q_coeff[0];
            else if (expo_factor < q_coeff[n_coeff-1]) {
                if (zero_center) {
                    tran_weights[f*size + i] = 0;
                    continue;
                } else {
                    expo_factor = q_coeff[n_coeff-1];
                }
            }
            tran_weights[f*size + i] = -mean * powf(2.f, expo_factor);
        }
    }
}

void make_shifting_weights_mean_gpu(float *weights, int n, int size, float *tran_weights, float* q_coeff, int n_coeff, int zero_center)
{
    make_shifting_weights_mean_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, tran_weights, q_coeff, n_coeff, zero_center);
    check_error(cudaPeekAtLastError());
}

__global__ void make_shifting_weights_kernel(float *weights, int n, float *tran_weights, float* q_coeff, int n_coeff, int zero_center)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (weights[i] > 0) {
        float expo_factor = roundf(log2f(weights[i]));
        if (expo_factor > q_coeff[0]) expo_factor = q_coeff[0];
        else if (expo_factor < q_coeff[n_coeff-1]) {
            if (zero_center) {
                tran_weights[i] = 0;
                return;
            } else {
                expo_factor = q_coeff[n_coeff-1];
            }
        }
        tran_weights[i] = powf(2.f, expo_factor);
    } else {
        float expo_factor = roundf(log2f(-weights[i]));
        if (expo_factor > q_coeff[0]) expo_factor = q_coeff[0];
        else if (expo_factor < q_coeff[n_coeff-1]) {
            if (zero_center) {
                tran_weights[i] = 0;
                return;
            } else {
                expo_factor = q_coeff[n_coeff-1];
            }
        }
        tran_weights[i] = -powf(2.f, expo_factor);
    }
}

void make_shifting_weights_gpu(float *weights, int n, float *tran_weights, float* q_coeff, int n_coeff, int zero_center)
{
    make_shifting_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, tran_weights, q_coeff, n_coeff, zero_center);
    check_error(cudaPeekAtLastError());
}

__global__ void uniform_quantize_weights_kernel(float *weights, int n, float *tran_weights, float step_size, float* q_coeff, int n_coeff, int zero_center)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    tran_weights[i] = (round(weights[i]/step_size + 0.5 - (float)zero_center/2) - 0.5 + (float)zero_center/2) * step_size;
    tran_weights[i] = tran_weights[i] > q_coeff[n_coeff - 1] ? q_coeff[n_coeff - 1] : (tran_weights[i] < -q_coeff[n_coeff - 1] ? -q_coeff[n_coeff - 1] : tran_weights[i]);
}

void uniform_quantize_weights_gpu(float *weights, int n, float *tran_weights, float step_size, float* q_coeff, int n_coeff, int zero_center)
{
    uniform_quantize_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, tran_weights, step_size, q_coeff, n_coeff, zero_center);
    check_error(cudaPeekAtLastError());
}

__global__ void make_cycle_weights_kernel(float* weights, int n, float* tran_weights, int num_level, float step)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float pos_pole = powf(2.f, num_level - 1) - 1;
    float neg_pole = -powf(2.f, num_level - 1);

    float period = powf(2.f, num_level);
    tran_weights[i] = round(weights[i] / step);
    tran_weights[i] = tran_weights[i] - (int)(tran_weights[i] / period) * period;
    if (tran_weights[i] > pos_pole) {
        tran_weights[i] -= period;
    } else if (tran_weights[i] < neg_pole) {
        tran_weights[i] += period;
    }
    tran_weights[i] *= step;
}

void make_cycle_weights_gpu(float* weights, int n, float* tran_weights, int num_level, float step)
{
    make_cycle_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, tran_weights, num_level, step);
    check_error(cudaPeekAtLastError());
}