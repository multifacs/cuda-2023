#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

__global__ void float_matrix_multiplication_kernel(const int m, const int n,
                                                   const int k, float* x,
                                                   float* y, float* z) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < k && row < m) {
    for (int i = 0; i < n; i++) {
      z[row * k + col] += x[row * n + i] * y[i * k + col];
    }
  }
}

__global__ void block_float_matrix_multiplication_kernel(const int m,
                                                         const int n,
                                                         const int k, float* x,
                                                         float* y, float* z) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  float res = 0;
  __shared__ float a_block[16 * 16];
  __shared__ float b_block[16 * 16];
  if (col < k && row < m) {
    for (int i = 0; i < n; i += blockDim.y) {
      a_block[threadIdx.y * blockDim.x + threadIdx.x] =
          x[(blockIdx.y * blockDim.y + threadIdx.y) * n + (i + threadIdx.x)];
      b_block[threadIdx.y * blockDim.y + threadIdx.x] =
          y[(i + threadIdx.y) * k + (blockIdx.x * blockDim.x + threadIdx.x)];
      __syncthreads();
      for (int j = 0; j < blockDim.x; j++) {
        res += a_block[threadIdx.y * blockDim.y + j] *
               b_block[j * blockDim.y + threadIdx.x];
      }
      __syncthreads();
    }
    z[row * k + col] += res;
  }
}

void float_matrix_multiplication_cuda(const int m, const int n, const int k,
                                      const float* x, const float* y, float* z,
                                      const dim3 dimGrid, const dim3 dimBlock) {
  cudaError_t cudaStatus;

  float *gpuX, *gpuY, *gpuZ;
  cudaStatus = cudaMalloc((void**)&gpuX, n * m * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc(gpuX) faild\n");
    return;
  }
  cudaStatus = cudaMalloc((void**)&gpuY, n * k * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc(gpuY) faild\n");
    return;
  }
  cudaStatus = cudaMalloc((void**)&gpuZ, m * k * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc(gpuZ) faild\n");
    return;
  }

  cudaStatus =
      cudaMemcpy(gpuX, x, n * m * sizeof(float), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy(gpuX) faild\n");
    return;
  }

  cudaStatus =
      cudaMemcpy(gpuY, y, n * k * sizeof(float), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy(gpuY) faild\n");
    return;
  }

  cudaStatus =
      cudaMemcpy(gpuZ, z, m * k * sizeof(float), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy(gpuZ) faild\n");
    return;
  }

  cudaEvent_t start, stop;
  float gpuTime = 0.0f;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  float_matrix_multiplication_kernel<<<dimGrid, dimBlock>>>(m, n, k, gpuX, gpuY,
                                                            gpuZ);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpuTime, start, stop);

  printf("CUDA: %dms", int(gpuTime));
  printf("\n");

  cudaStatus =
      cudaMemcpy(z, gpuZ, m * k * sizeof(float), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy(gpuZ) faild\n");
    return;
  }

  cudaFree(gpuX);
  cudaFree(gpuY);
  cudaFree(gpuZ);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return;
}

void block_float_matrix_multiplication_cuda(const int m, const int n,
                                            const int k, const float* x,
                                            const float* y, float* z,
                                            const dim3 dimGrid,
                                            const dim3 dimBlock) {
  cudaError_t cudaStatus;

  float *gpuX, *gpuY, *gpuZ;
  cudaStatus = cudaMalloc((void**)&gpuX, n * m * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc(gpuX) faild\n");
    return;
  }
  cudaStatus = cudaMalloc((void**)&gpuY, n * k * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc(gpuY) faild\n");
    return;
  }
  cudaStatus = cudaMalloc((void**)&gpuZ, m * k * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc(gpuZ) faild\n");
    return;
  }

  cudaStatus =
      cudaMemcpy(gpuX, x, n * m * sizeof(float), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy(gpuX) faild\n");
    return;
  }

  cudaStatus =
      cudaMemcpy(gpuY, y, n * k * sizeof(float), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy(gpuY) faild\n");
    return;
  }

  cudaStatus =
      cudaMemcpy(gpuZ, z, m * k * sizeof(float), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy(gpuZ) faild\n");
    return;
  }

  cudaEvent_t start, stop;
  float gpuTime = 0.0f;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  block_float_matrix_multiplication_kernel<<<dimGrid, dimBlock>>>(m, n, k, gpuX,
                                                                  gpuY, gpuZ);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpuTime, start, stop);

  printf("CUDA GEMM: %dms", int(gpuTime));
  printf("\n");

  cudaStatus =
      cudaMemcpy(z, gpuZ, m * k * sizeof(float), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy(gpuZ) faild\n");
    return;
  }

  cudaFree(gpuX);
  cudaFree(gpuY);
  cudaFree(gpuZ);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return;
}

void float_matrix_multiplication(const int m, const int n, const int k,
                                 float* x, float* y, float* z) {
  for (int i = 0; i < m; ++i)
    for (int p = 0; p < k; ++p)
      for (int j = 0; j < n; ++j) z[i * k + p] += x[i * n + j] * y[j * k + p];
}

void float_matrix_multiplication_omp(const int m, const int n, const int k,
                                     float* x, float* y, float* z) {
  int i, p, j;
  omp_set_num_threads(2);
#pragma omp parallel for private(i, p, j) shared(x, y, z)
  for (i = 0; i < m; ++i) {
    for (p = 0; p < n; ++j) {
      float sum = 0;
      for (j = 0; j < k; ++k) {
        sum += x[i * n + j] * y[j * k + p];
      }
      z[i * k + p] = sum;
    }
  }
}

int main() {
  const int N = 1024;

  float* x = new float[N * N];
  float* y = new float[N * N];
  float* z = new float[N * N];

  for (int i = 0; i < N * N; i++) {
    x[i] = 1.0;
  }
  for (int i = 0; i < N * N; i++) {
    y[i] = 1.0;
  }
  for (int i = 0; i < N * N; i++) {
    z[i] = 0.0;
  }

  float startTime = omp_get_wtime();
  float_matrix_multiplication(N, N, N, x, y, z);
  float endTime = omp_get_wtime();

  printf("Sequential: %dms", int((endTime - startTime) * 1000));
  printf("\n");

  startTime = omp_get_wtime();
  float_matrix_multiplication_omp(N, N, N, x, y, z);
  endTime = omp_get_wtime();

  printf("OpenMP: %dms\n\n", int((endTime - startTime) * 1000));

  dim3 dimBlockf(16, 16);
  dim3 dimGridf((N + dimBlockf.x - 1) / dimBlockf.x,
                (N + dimBlockf.y - 1) / dimBlockf.y);

  float_matrix_multiplication_cuda(N, N, N, x, y, z, dimGridf, dimBlockf);

  dim3 dimBlock_1f(16, 16);
  dim3 dimGrid_1f((N + dimBlock_1f.x - 1) / dimBlock_1f.x,
                  (N + dimBlock_1f.y - 1) / dimBlock_1f.y);

  block_float_matrix_multiplication_cuda(N, N, N, x, y, z, dimGrid_1f,
                                         dimBlock_1f);

  delete[] x, delete[] y, delete[] z;

  return 0;
}