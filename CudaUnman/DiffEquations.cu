#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MiniWrapForCuda.h"



//прототип функции с диф.ур., которая передается из C#
//typedef  double(*FType)(int, double, double, double);

//диф.ур.
__device__ double func(int i, double y1, double y2, double y3) {
	switch (i) {
	case 0: return -(55 + y3) * y1 + 65 * y2;
	case 1: return 0.0785 * (y1 - y2);
	case 2: return 0.1 * y1;
	}
}


__global__ void RK4(int n, double* y, double* r, int t, double tau) {
	int idx = threadIdx.x;
	int i = (t - 1) * n;
	//расчет 1 коэффициента
	r[idx] = tau * func(idx, y[i], y[i + 1], y[i + 2]);
	//расчет 2 коэффициента
	r[idx + n] = tau * func(idx, y[i] + 0.5 * r[0], y[i + 1] + 0.5 * r[1], y[i + 2] + 0.5 * r[2]);
	//расчет 3 коэффициента
	r[idx + n * 2] = tau * func(idx, y[i] + 0.5 * r[0 + n], y[i + 1] + 0.5 * r[1 + n], y[i + 2] + 0.5 * r[2 + n]);
	//расчет 4 коэффициента
	r[idx + n * 3] = tau * func(idx, y[i] + r[0 + n * 2], y[i + 1] + r[1 + n * 2], y[i + 2] + r[2 + n * 2]);

	y[i + n + idx] = y[i + idx] + (r[idx] + 2.0 * r[idx + n] + 2.0 * r[idx + n * 2] + r[idx + n * 3]) / 6.0;
}

__global__ void Eiler(int n, double* y, int t, double tau) {
	int idx = threadIdx.x;
	int i = (t - 1) * n;
	y[i + n + idx] = y[i + idx] + tau * func(idx, y[i], y[i + 1], y[i + 2]);
}

__global__ void RK2(int n, double* y, double* r, int t, double tau) {
	int idx = threadIdx.x;
	int i = (t - 1) * n;

	/*r[idx] = y[i + idx] + tau/2* func(idx, y[i], y[i + 1], y[i + 2]);
	y[i + n + idx] = y[i + idx] + tau * func(idx, r[0], r[1], r[2]);*/

	//расчет 1 коэффициента
	r[idx] = tau * func(idx, y[i], y[i + 1], y[i + 2]);
	//расчет 2 коэффициента
	r[idx + n] = tau * func(idx, y[i] + r[0], y[i + 1] + r[1], y[i + 2] + r[2]);

	y[i + n + idx] = y[i + idx] + (r[idx] + r[idx + n]) * 0.5;
}

//функция выделения памяти и вызова ядра
double* Compute(double t0, double tmax, double tau, int n, double* ynach, void* Function, int method) {

	//получаем указатель на функцию из C#
	//FType F = (FType)(Function);

	double time = 0.0;
	//число шагов
	int nn = (int)((tmax - t0) / tau);

	//массив значений
	double* y = new double[n * nn];

	//начальные значения
	for (int i = 0; i < n; i++)
		y[i] = ynach[i];

	double* yDev = NULL;
	int size = n * nn * sizeof(double);
	cudaMalloc((void**)&yDev, size);

	cudaMemcpy(yDev, y, size, cudaMemcpyHostToDevice);

	//массив коэффициентов приближения
	double* rk2 = NULL;
	double* rk4 = NULL;
	cudaMalloc((void**)&rk2, n * 2);
	cudaMalloc((void**)&rk4, n * 4);


	switch (method) {
	case 1:
		{
			for (int t = 1; t < nn; t++)
				Eiler << <1, n >> >(n, yDev, t, tau);
			break;
		}
	case 2:
		{
			for (int t = 1; t < nn; t++)
				RK2 << <1, n >> >(n, yDev, rk2, t, tau);
			break;
		}
	case 3:
		{
			for (int t = 1; t < nn; t++)
				RK4 << <1, n >> >(n, yDev, rk4, t, tau);
			break;
		}

	}

	cudaMemcpy(y, yDev, size, cudaMemcpyDeviceToHost);

	/*FILE *f = fopen("result.txt", "w");
	for (int t = 1; t<nn; t++){
		time += tau;
		fprintf(f, "time=%f y=%f %f %f\n", time, y[t*n], y[1 + t*n], y[2 + t*n]);
	}*/

	cudaFree(yDev);

	//fclose(f);
	return y;
	delete y;
}


//метод Эйлера 
//t0 - время начальное, tmax - время конечное, tau - шаг, n - число уравнений, *ynach - массив начальных значений, *Function - диф.ур.
double* Eiler_CUDA(double t0, double tmax, double tau, int n, double* ynach, void* Function) {
	return Compute(t0, tmax, tau, n, ynach, Function, 1);

}

//метод Рунге-Кутты 2
//t0 - время начальное, tmax - время конечное, tau - шаг, n - число уравнений, *ynach - массив начальных значений, *Function - диф.ур.
double* RK2_CUDA(double t0, double tmax, double tau, int n, double* ynach, void* Function) {
	return Compute(t0, tmax, tau, n, ynach, Function, 2);

}

//метод Рунге-Кутты 4
//t0 - время начальное, tmax - время конечное, tau - шаг, n - число уравнений, *ynach - массив начальных значений, *Function - диф.ур.
double* RK4_CUDA(double t0, double tmax, double tau, int n, double* ynach, void* Function) {
	return Compute(t0, tmax, tau, n, ynach, Function, 3);
}
