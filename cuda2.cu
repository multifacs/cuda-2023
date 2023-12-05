#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <omp.h>
#include <stdio.h>

#include <iostream>

__global__ void saxpy_kernel(const int n, const float a, float *x,
                             const int incx, float *y, const int incy) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    y[i * incy] += a * x[i * incx];
  }
}

__global__ void daxpy_kernel(const int n, const double a, double *x,
                             const int incx, double *y, const int incy) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    y[i * incy] += a * x[i * incx];
  }
}

void saxpy_gpu(const int n, const float a, float *x, const int incx, float *y,
               const int incy, const int numBlocks, const int blocksSize) {
  cudaError_t cudaStatus;
  int sizeX = 1 + (n - 1) * abs(incx);
  int sizeY = 1 + (n - 1) * abs(incy);

  float *gpuX;
  cudaStatus = cudaMalloc((void **)&gpuX, sizeX * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc(gpuX) faild f1\n");
    return;
  }

  float *gpuY;
  cudaStatus = cudaMalloc((void **)&gpuY, sizeY * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc(gpuY) faild f2\n");
    return;
  }

  cudaStatus =
      cudaMemcpy(gpuX, x, sizeX * sizeof(float), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy(gpuX) faild\n");
    return;
  }

  cudaStatus =
      cudaMemcpy(gpuY, y, sizeY * sizeof(float), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy(gpuY) faild\n");
    return;
  }

  cudaEvent_t startTime, stopF;
  float gpuTimeF = 0.0f;

  cudaEventCreate(&startTime);
  cudaEventCreate(&stopF);
  cudaEventRecord(startTime, 0);

  saxpy_kernel<<<numBlocks, blocksSize>>>(n, a, gpuX, incx, gpuY, incy);

  cudaEventRecord(stopF, 0);
  cudaEventSynchronize(stopF);
  cudaEventElapsedTime(&gpuTimeF, startTime, stopF);

  printf("OpenGL: %dms  size: %d", int(gpuTimeF), blocksSize);

  cudaStatus =
      cudaMemcpy(y, gpuY, sizeY * sizeof(float), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy(gpuY) faild\n");
    return;
  }

  cudaFree(gpuX);
  cudaFree(gpuY);
  cudaEventDestroy(startTime);
  cudaEventDestroy(stopF);
  return;
}

void daxpy_gpu(const int n, const double a, double *x, const int incx,
               double *y, const int incy, const int numBlocks,
               const int blocksSize) {
  cudaError_t cudaStatus;
  int sizeX = 1 + (n - 1) * abs(incx);
  int sizeY = 1 + (n - 1) * abs(incy);

  double *gpuX;
  cudaStatus = cudaMalloc((void **)&gpuX, sizeX * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc(gpuX) faild d1\n");
    return;
  }

  double *gpuY;
  cudaStatus = cudaMalloc((void **)&gpuY, sizeY * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc(gpuY) faild d2\n");
    return;
  }

  cudaStatus =
      cudaMemcpy(gpuX, x, sizeX * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy(gpuX) faild\n");
    return;
  }

  cudaStatus =
      cudaMemcpy(gpuY, y, sizeY * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy(gpuY) faild\n");
    return;
  }

  cudaEvent_t startTime, stopD;
  float gpuTimeD = 0.0f;

  cudaEventCreate(&startTime);
  cudaEventCreate(&stopD);
  cudaEventRecord(startTime, 0);

  daxpy_kernel<<<numBlocks, blocksSize>>>(n, a, gpuX, incx, gpuY, incy);

  cudaEventRecord(stopD, 0);
  cudaEventSynchronize(stopD);
  cudaEventElapsedTime(&gpuTimeD, startTime, stopD);

  printf("OpenGL: %dms  size: %d", int(gpuTimeD), blocksSize);

  cudaStatus =
      cudaMemcpy(y, gpuY, sizeY * sizeof(double), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy(gpuY) faild\n");
    return;
  }

  cudaFree(gpuX);
  cudaFree(gpuY);
  cudaEventDestroy(startTime);
  cudaEventDestroy(stopD);
  return;
}

template <typename t>
bool comp(t *a1, t *a2, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (a1[i] != a2[i]) return false;
  }
  return true;
}

void saxpy(const int n, const float a, float *x, const int incx, float *y,
           const int incy) {
  const int biasx = incx < 0 ? (n - 1) * abs(incx) : 0;
  const int biasy = incy < 0 ? (n - 1) * abs(incy) : 0;

  for (size_t i = 0; i < n; i++) {
    y[biasy + i * incy] += a * x[biasx + i * incx];
  }
}

void daxpy(const int n, const double a, double *x, const int incx, double *y,
           const int incy) {
  const int biasx = incx < 0 ? (n - 1) * abs(incx) : 0;
  const int biasy = incy < 0 ? (n - 1) * abs(incy) : 0;

  for (size_t i = 0; i < n; i++) {
    y[biasy + i * incy] += a * x[biasx + i * incx];
  }
}

void saxpy_omp(const int n, const float a, float *x, const int incx, float *y,
               const int incy) {
  const int biasx = incx < 0 ? (n - 1) * abs(incx) : 0;
  const int biasy = incy < 0 ? (n - 1) * abs(incy) : 0;

#pragma omp parallel for num_threads(4)
  for (int i = 0; i < n; i++) {
    y[biasy + i * incy] += a * x[biasx + i * incx];
  }
}

void daxpy_omp(const int n, const double a, double *x, const int incx,
               double *y, const int incy) {
  const int biasx = incx < 0 ? (n - 1) * abs(incx) : 0;
  const int biasy = incy < 0 ? (n - 1) * abs(incy) : 0;

#pragma omp parallel for num_threads(4)
  for (int i = 0; i < n; i++) {
    y[biasy + i * incy] += a * x[biasx + i * incx];
  }
}

int main() {
  const int n = 50000000;  // 1e7;
  const int incx = 10;
  const int incy = 10;
  const int sizeX = 1 + (n - 1) * abs(incx);
  const int sizeY = 1 + (n - 1) * abs(incy);
  int block_size;
  int num_blocks;

  const float aFloat = 10.0f;
  float *xFloat = new float[sizeX];
  float *yFloat = new float[sizeY];
  for (int i = 0; i < n; ++i) {
    xFloat[i] = 5.0f;
    yFloat[i] = 1.0f;
  }

  double startTime = omp_get_wtime();
  saxpy(n, aFloat, xFloat, incx, yFloat, incy);
  double endTime = omp_get_wtime();

  printf("Saxpy Type Float\n");
  printf("Sequential: %dms", int((endTime - startTime) * 1000.0));
  printf("\n");

  delete[] xFloat;
  delete[] yFloat;

  xFloat = new float[sizeX];
  yFloat = new float[sizeY];
  for (int i = 0; i < n; ++i) {
    xFloat[i] = 5.0f;
    yFloat[i] = 1.0f;
  }

  startTime = omp_get_wtime();
  saxpy_omp(n, aFloat, xFloat, incx, yFloat, incy);
  endTime = omp_get_wtime();

  printf("OpenMP: %dms", int((endTime - startTime) * 1000.0));
  printf("\n");

  delete[] xFloat;
  delete[] yFloat;

  for (int i = 8; i <= 128; i *= 2) {
    block_size = i;
    num_blocks = (n + block_size - 1) / block_size;

    xFloat = new float[sizeX];
    yFloat = new float[sizeY];

    for (int i = 0; i < n; ++i) {
      xFloat[i] = 5.0;
      yFloat[i] = 1.0;
    }
    saxpy_gpu(n, aFloat, xFloat, incx, yFloat, incy, num_blocks, block_size);
    printf("\n");

    delete[] xFloat;
    delete[] yFloat;
  }

  const double aDouble = 10.0;
  double *xDouble = new double[sizeX];
  double *yDouble = new double[sizeY];
  for (int i = 0; i < n; ++i) {
    xDouble[i] = 5.0;
    yDouble[i] = 1.0;
  }

  startTime = omp_get_wtime();
  daxpy(n, aDouble, xDouble, incx, yDouble, incy);
  endTime = omp_get_wtime();

  printf("\n");
  printf("Daxpy Type Double\n");
  printf("Sequential: %dms", int((endTime - startTime) * 1000.0));
  printf("\n");

  delete[] xDouble;
  delete[] yDouble;

  xDouble = new double[sizeX];
  yDouble = new double[sizeY];
  for (int i = 0; i < n; ++i) {
    xDouble[i] = 5.0;
    yDouble[i] = 1.0;
  }

  startTime = omp_get_wtime();
  daxpy_omp(n, aDouble, xDouble, incx, yDouble, incy);
  endTime = omp_get_wtime();

  printf("OpenMP: %dms", int((endTime - startTime) * 1000));
  printf("\n");
  delete[] xDouble;
  delete[] yDouble;

  for (int i = 8; i <= 128; i *= 2) {
    block_size = i;
    num_blocks = (n + block_size - 1) / block_size;

    xDouble = new double[sizeX];
    yDouble = new double[sizeY];
    for (int i = 0; i < n; ++i) {
      xDouble[i] = 5.0;
      yDouble[i] = 1.0;
    }

    daxpy_gpu(n, aDouble, xDouble, incx, yDouble, incy, num_blocks, block_size);
    printf("\n");

    delete[] xDouble;
    delete[] yDouble;
  }

  return 0;
}