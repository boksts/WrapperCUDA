#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MiniWrapForCuda.h"
#include <stdio.h>
#include <stdlib.h>

// RK4
#define n 3

__device__ double func(int i, double y1, double y2, double y3) {
	switch (i) {
	case 0: return -(55 + y3)*y1 + 65 * y2;
	case 1:	return 0.0785*(y1 - y2);
	case 2: return 0.1*y1;
	}

}

__global__ void RK4(double *y, int t, double tau) {
	int idx = threadIdx.x;
	int i = (t - 1)*n;
	__shared__ double r1[n], r2[n], r3[n], r4[n];
	r1[idx] = tau*func(idx, y[i], y[i + 1], y[i + 2]);
	__syncthreads();
	r2[idx] = tau*func(idx, y[i] + 0.5*r1[0], y[i + 1] + 0.5*r1[1], y[i + 2] + 0.5*r1[2]);
	__syncthreads();
	r3[idx] = tau*func(idx, y[i] + 0.5*r2[0], y[i + 1] + 0.5*r2[1], y[i + 2] + 0.5*r2[2]);
	__syncthreads();
	r4[idx] = tau*func(idx, y[i] + r3[0], y[i + 1] + r3[1], y[i + 2] + r3[2]);
	y[i + n + idx] = y[i + idx] + (r1[idx] + 2.0*r2[idx] + 2.0*r3[idx] + r4[idx]) / 6.0;
}

__global__ void Eiler(double *y, int t, double tau) {
	int idx = threadIdx.x;
	int i = (t - 1)*n;
	y[i + n + idx] = y[i + idx] + tau*func(idx, y[i], y[i + 1], y[i + 2]);
}

__global__ void RK2(double *y, int t, double tau) {
	int idx = threadIdx.x;
	int i = (t - 1)*n;
	__shared__ double r1[n], r2[n];
	r1[idx] = tau*func(idx, y[i], y[i + 1], y[i + 2]);
	__syncthreads();
	r2[idx] = tau*func(idx, y[i] + r1[0], y[i + 1] +r1[1], y[i + 2] + r1[2]);
	
	y[i + n + idx] = y[i + idx] + (r1[idx] + r2[idx]) * 0.5;
}

double *Compute(double t0, double tmax, double tau, int method){
	double time = 0.0;
	//число шагов
	int nn = (int)((tmax - t0) / tau);

	//массив значений
	double *y = new double[n*nn];

	//начальные значения
	y[0] = y[1] = 1.0, y[2] = 0.0;

	double *yDev = NULL;
	int size = n*nn*sizeof(double);
	cudaMalloc((void**)&yDev, size);

	cudaMemcpy(yDev, y, size, cudaMemcpyHostToDevice);

	 
	switch (method){
		case 1: {
			for (int t = 1; t<nn; t++)
				Eiler << <1, n >> >(yDev, t, tau);
			break;
		}
		case 2: {
			for (int t = 1; t<nn; t++)
				RK2 << <1, n >> >(yDev, t, tau);
			break;
		}
		case 3: {
			for (int t = 1; t<nn; t++)
				RK4 << <1, n >> >(yDev, t, tau);
			break;
		}
		
	}

	cudaMemcpy(y, yDev, size, cudaMemcpyDeviceToHost);

	FILE *f = fopen("result.txt", "w");
	for (int t = 1; t<nn; t++){
		time += tau;
		fprintf(f, "time=%f y=%f %f %f\n", time, y[t*n], y[1 + t*n], y[2 + t*n]);
	}

	cudaFree(yDev);

	fclose(f);
	return y;
	delete y;


}

double *Eiler_CUDA(double t0, double tmax, double tau){
	return Compute(t0, tmax, tau, 1);

}

double *RK2_CUDA(double t0, double tmax, double tau){
	return Compute(t0, tmax, tau, 2);

}

double *RK4_CUDA(double t0, double tmax, double tau) {
	return Compute(t0, tmax, tau, 3);
}
