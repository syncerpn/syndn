#ifndef CUDA_H
#define CUDA_H

#include "darknet.h"

void check_error(cudaError_t status);
void cuda_random(float *x_gpu, size_t n);
int *cuda_make_int_array(int *x, size_t n);
float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
cublasHandle_t blas_handle();
cudnnHandle_t cudnn_handle();
dim3 cuda_gridsize(size_t n);

#endif