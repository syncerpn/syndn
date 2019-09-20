#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "batchnorm_layer.h"
#include "blas.h"
}
#include <stdio.h>

__device__ float quantize_kernel(float x, int nbit, int ibit, int sign)
{
    float max, min;
    float ifactor = powf(2,ibit);
    float qfactor = powf(2,nbit-ibit);
    if (sign) {
        max = ifactor/2 - 1/qfactor;
        min = max - ifactor + 1/qfactor;
    } else {
        max = ifactor - 1/qfactor;
        min = 0;
    }
    return (x > max) ? max : ((x < min) ? min : (round(x * qfactor)) / qfactor);
}

__global__ void make_scale_new_factor_kernel(float* factor, float *scale, float* var, int batch, int filters, fixed_point_scheme fps, float* scale_new)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= filters*batch) return;

    int k = index % filters;
    index /= filters;
    int f = index;
    int idx = f*filters + k;

    scale_new[idx] = factor[idx] * scale[idx] / sqrtf(var[idx] + .00001f);

    if (fps.type) {
        scale_new[idx] = quantize_kernel(scale_new[idx], fps.nbit, fps.ibit, 0);
    }
}

void make_scale_new_factor(float* factor, float *scale, float* var, int batch, int filters, fixed_point_scheme fps, float* scale_new)
{
    int num = filters * batch;
    make_scale_new_factor_kernel<<<cuda_gridsize(num), BLOCK>>>(factor, scale, var, batch, filters, fps, scale_new);
    check_error(cudaPeekAtLastError());
}

__global__ void make_scale_new_scalar_kernel(float scalar, float *scale, float* var, int batch, int filters, fixed_point_scheme fps, float* scale_new)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= filters*batch) return;

    int k = index % filters;
    index /= filters;
    int f = index;
    int idx = f*filters + k;

    scale_new[idx] = scalar * scale[idx] / sqrtf(var[idx] + .00001f);

    if (fps.type) {
        scale_new[idx] = quantize_kernel(scale_new[idx], fps.nbit, fps.ibit, 0);
    }
}

void make_scale_new_scalar(float scalar, float *scale, float* var, int batch, int filters, fixed_point_scheme fps, float* scale_new)
{
    int num = filters * batch;
    make_scale_new_scalar_kernel<<<cuda_gridsize(num), BLOCK>>>(scalar, scale, var, batch, filters, fps, scale_new);
    check_error(cudaPeekAtLastError());
}

__global__ void make_bias_new_factor_kernel(float* bias, float* factor, float *scale, float* var, int batch, int filters, fixed_point_scheme fps, float* bias_new)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= filters*batch) return;

    int k = index % filters;
    index /= filters;
    int f = index;
    int idx = f*filters + k;

    bias_new[idx] = bias[idx] - factor[idx] * scale[idx] / sqrtf(var[idx] + .00001f);

    if (fps.type) {
        bias_new[idx] = quantize_kernel(bias_new[idx], fps.nbit, fps.ibit, 1);
    }
}

void make_bias_new_factor(float* bias, float* factor, float *scale, float* var, int batch, int filters, fixed_point_scheme fps, float* bias_new)
{
    int num = filters * batch;
    make_bias_new_factor_kernel<<<cuda_gridsize(num), BLOCK>>>(bias, factor, scale, var, batch, filters, fps, bias_new);
    check_error(cudaPeekAtLastError());
}