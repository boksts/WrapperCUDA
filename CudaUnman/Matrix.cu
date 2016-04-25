#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MiniWrapForCuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 8


//тансопнирование матрицы
__global__ void transpose(double* inputMatrix, double* outputMatrix, int width, int height){
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if( x < width &&  y < height)
			outputMatrix[x * height + y] = inputMatrix[y * width + x];
}

double *Transp_CUDA(double *a, int N, int M){

	double * a_t = new double[N*M];
	double* a_dev; //Исходная матрица 
	double* a_t_dev; //Транспонированная матрица 
	double* a_t1_dev;

	cudaMalloc((void**)&a_dev, N*M * sizeof(double));
	cudaMalloc((void**)&a_t_dev, N*M * sizeof(double));

	cudaMemcpy(a_dev, a, N*M * sizeof(double), cudaMemcpyHostToDevice);

	int gridSizeX = (M / BLOCK_SIZE) + ((M % BLOCK_SIZE) > 0 ? 1 : 0);
	int gridSizeY = (N / BLOCK_SIZE) + ((N % BLOCK_SIZE) > 0 ? 1 : 0);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(gridSizeX, gridSizeY);

	transpose << <blocks, threads >> >(a_dev, a_t_dev, M, N);
	cudaThreadSynchronize();
	
	cudaMemcpy(a_t, a_t_dev, N*M * sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaFree(a_dev);
	cudaFree(a_t_dev);

	return a_t;
	delete[] a_t;
}


//умножение матриц
__global__ void matMult(double *a, double *b, int M, int N, int Q, double * c)
{
	int   i = blockDim.x * blockIdx.x + threadIdx.x;
	int   j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < M && j < Q  ){
			c[i*Q + j] = 0;
			for (int k = 0; k < N; k++)
				c[i*Q + j] += a[i*N + k] * b[k*Q + j];
	}
}


double *Mult_CUDA(double *a, double *b, int M, int N, int Q)
{

	double * c = new double[M*Q];

	double * adev = NULL;
	double * bdev = NULL;
	double * cdev = NULL;

	cudaMalloc((void**)&adev, sizeof(double) *M*N);
	cudaMalloc((void**)&bdev, sizeof(double)*Q*N);
	cudaMalloc((void**)&cdev, sizeof(double)*M*Q);

	//определение размера грида
	int gridSizeX = (M / BLOCK_SIZE) + ((M % BLOCK_SIZE) > 0 ? 1 : 0);
	int gridSizeY = (Q / BLOCK_SIZE) + ((Q % BLOCK_SIZE) > 0 ? 1 : 0);

	//определение числа блоков и потоков
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(gridSizeX, gridSizeY);

	cudaMemcpy(adev, a, sizeof(double) *M*N, cudaMemcpyHostToDevice);
	cudaMemcpy(bdev, b, sizeof(double) *Q*N, cudaMemcpyHostToDevice);

	matMult <<<blocks, threads >>>(adev, bdev, M, N,Q, cdev);
	cudaThreadSynchronize();

	cudaMemcpy(c, cdev, sizeof(double) *M*Q, cudaMemcpyDeviceToHost);

	cudaFree(adev);
	cudaFree(bdev);
	cudaFree(cdev);


	return c;
	delete c;
}


//умножение матрицы на вектор
__global__ void matMultVector(double *a, double *b, int m, int n, double * c)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid<m)
	{
		float sum = 0;
		for (int i = 0; i<n; i++)
			sum += a[tid*n + i] * b[i];
		c[tid] = sum;
	}

}


double *MultVector_CUDA(double *a, double *b, int M, int N)
{
	double * c = new double[M];
	double * adev = NULL;
	double * bdev = NULL;
	double * cdev = NULL;

	cudaMalloc((void**)&adev, sizeof(double) * N * M);
	cudaMalloc((void**)&bdev, sizeof(double) * N);
	cudaMalloc((void**)&cdev, sizeof(double) * M);

	//определение размера грида
	int gridSizeX = (M / BLOCK_SIZE) + ((M % BLOCK_SIZE) > 0 ? 1 : 0);

	//определение числа блоков и потоков
	dim3 threads(BLOCK_SIZE);
	dim3 blocks(gridSizeX);

	cudaMemcpy(adev, a, sizeof(double) * N * M, cudaMemcpyHostToDevice);
	cudaMemcpy(bdev, b, sizeof(double) * N, cudaMemcpyHostToDevice);

	matMultVector << <blocks, threads >> >(adev, bdev, M, N, cdev);
	cudaThreadSynchronize();

	cudaMemcpy(c, cdev, sizeof(double)*M, cudaMemcpyDeviceToHost);

	cudaFree(adev);
	cudaFree(bdev);
	cudaFree(cdev);

	return c;
	delete c;
}