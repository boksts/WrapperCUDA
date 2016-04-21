#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MiniWrapForCuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

 __device__ float func(float x){
	return (x*x + 15);
}

typedef double(*FType)(float x);




__global__ void SimpsonMethod(float *sum_Dev, float *cut_Dev, float a, float b, int n) {

	int i = blockIdx.x*blockDim.x + threadIdx.x + 1;

	float h = (b - a) / n;

	cut_Dev[0] = a;
	for (int j = 1; j < n; j++)
		cut_Dev[j] = cut_Dev[j - 1] + h;

	if (i < n)
		sum_Dev[i] = func(cut_Dev[i]) + 4 * func(cut_Dev[i] + h / 2) + func(cut_Dev[i]+h);

}


double Simpson_CUDA(float a, float b, int n, void *Function) {
	float  result = 0;
	
FType F;
	F = (FType)(Function);

	float h = (b - a) / n;
	float *sum = new float[n];
	float *sum_Dev = NULL;
	float *cut_Dev = NULL;

	cudaMalloc((void**)&sum_Dev, n*sizeof(float));
	cudaMalloc((void**)&cut_Dev, n*sizeof(float));

	SimpsonMethod <<<1, n>>>(sum_Dev, cut_Dev, a, b, n);
	cudaThreadSynchronize();

	cudaMemcpy(sum, sum_Dev, n*sizeof(float), cudaMemcpyDeviceToHost);

	for (int j = 0; j < n; j++)
		result += sum[j];

	//printf("result = %f\n", (h/6)*result);
	//system("pause");
	return result/6.0*h;
}