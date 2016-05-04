#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MiniWrapForCuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define BLOCK_SIZE 64

typedef  double(*FType)(float x);


//__device__  FType func;
__device__ double func(float x){
	return (x*x);
}






 __global__ void
SimpsonMethod_3_8(float *sum_Dev, float *cut_Dev, float a, float b, int n) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	float h = (b - a) / n;
	
	if (i == 0)
		sum_Dev[i] = ((3.0 / 8.0) * (func(a) + func(b)));
	if (i == 1)
		sum_Dev[i] = ((7.0 / 6.0) * (func(a + h) + func(b - h)));
	if (i == 3)
		sum_Dev[i] = ((23.0 / 24.0) * (func(a + 2 * h) + func(b - 2 * h)));
	if (i > 3)
		sum_Dev[i] = func(a + (i - 1)*h);
}


__global__ void SimpsonMethod(float *sum_Dev, float *cut_Dev, float a, float b, int n/*, FType func*/) {

	

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	float h = (b - a) / n;

	cut_Dev[i] = h*i;

	if (i != 0 && i % 2 == 0 && i != n - 1)
		sum_Dev[i] = 4 * func(a + cut_Dev[i]);
	if (i != 0 && i % 2 == 1 && i != n - 1)
		sum_Dev[i] = 2 * func(a + cut_Dev[i]);
	if (i == 0)
		sum_Dev[i] = func(a);
	if (i == n - 1)
		sum_Dev[i] = func(b);
}


double Compute(float a, float b, int n, void *Function, int method){
	FType F = (FType)(Function);
	FType F_Dev;
	//cudaMalloc((void**)&F_Dev, sizeof(FType));
	//cudaMemcpyToSymbol(F_Dev, F, sizeof(FType));
	///cudaMemcpy(F_Dev, F, sizeof(FType), cudaMemcpyHostToDevice);


	float *sum = new float[n];
	float *sum_Dev = NULL;
	float *cut_Dev = NULL;
	float h = (b - a) / n;

	cudaMalloc((void**)&sum_Dev, n*sizeof(float));
	cudaMalloc((void**)&cut_Dev, n*sizeof(float));

	int gridSizeX = (n / BLOCK_SIZE) + ((n % BLOCK_SIZE) > 0 ? 1 : 0);
	dim3 threads(BLOCK_SIZE, 1);
	dim3 blocks(gridSizeX, 1);

	switch (method){
	case 1: {
		SimpsonMethod << <blocks, threads >>>(sum_Dev, cut_Dev, a, b, n/*, F_Dev*/);
		break;
	}
	case 2: {
		SimpsonMethod_3_8 <<<blocks, threads >>>(sum_Dev, cut_Dev, a, b, n);
		break;
	}
	case 3: {
		break;
	}
	}

	cudaThreadSynchronize();

	cudaMemcpy(sum, sum_Dev, n*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(sum_Dev);
	cudaFree(cut_Dev);


	float result = 0;
	for (int j = 0; j < n; j++)
		result += sum[j];


	switch (method){
	case 1: {
		return (h / 3)*result;
	}
	case 2: {
		return h*result;
	}
	case 3: {
		break;
	}
	}


}

double Simpson_CUDA(float a, float b, int n, void *Function) {
	return Compute(a, b, n, Function, 1);
}

double Simpson_3_8_CUDA(float a, float b, int n, void *Function) {
	return Compute(a, b, n, Function, 2);
}