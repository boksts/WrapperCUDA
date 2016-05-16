#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MiniWrapForCuda.h"

#define BLOCK_SIZE 64

//прототип функции с подынтегральной функцией, которая передается из C#
//typedef  double(*FType)(float x);

//подынтегральная функция
__device__ double func(float x) {
	return (x * x);
}


//ядро для Симпсона 3/8
__global__ void SimpsonMethod_3_8(float* sum_Dev, float* cut_Dev, float a, float b, int n) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	float h = (b - a) / n;

	if (i == 0)
		//расчет значений на границах
		sum_Dev[i] = ((3.0 / 8.0) * (func(a) + func(b)));
	if (i == 1)
		//расчет смещенных значений на границах
		sum_Dev[i] = ((7.0 / 6.0) * (func(a + h) + func(b - h)));
	if (i == 3)
		//расчет смещенных значений на границах
		sum_Dev[i] = ((23.0 / 24.0) * (func(a + 2 * h) + func(b - 2 * h)));
	if (i > 3)
		//расчет внутренних значений
		sum_Dev[i] = func(a + (i - 1) * h);
}

//ядро для Симпсона
__global__ void SimpsonMethod(float* sum_Dev, float* cut_Dev, float a, float b, int n/*, FType func*/) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	float h = (b - a) / n;

	//расчет массива разбиений
	cut_Dev[i] = h * i;

	if (i != 0 && i % 2 == 0 && i != n - 1)
		//расчет четных внутренних значений
		sum_Dev[i] = 4 * func(a + cut_Dev[i]);
	if (i != 0 && i % 2 == 1 && i != n - 1)
		//расчет нечетных внутренних значений
		sum_Dev[i] = 2 * func(a + cut_Dev[i]);
	if (i == 0)
		//расчет значения левой границы
		sum_Dev[i] = func(a);
	if (i == n - 1)
		//расчет значения правой границы
		sum_Dev[i] = func(b);
}

//ядро для Гауса
__global__ void GaussMethod(float* sum_Dev, float* xm_Dev, float* cm_Dev, float a, float b, int n, int point) {

	int j = blockDim.x * blockIdx.x + threadIdx.x;

	float h = (b - a) / n;

	for (int i = 0; i < point; i++)
		//расчет значения интеграла по числу точек
		sum_Dev[j] += cm_Dev[i] * func(xm_Dev[i] * (h / 2) + a + j * h + h / 2);
}


//функция выделения памяти и вызова ядра для метода Симпсона и Симпсона 3/8
double Compute(float a, float b, int n, void* Function, int method) {

	//получаем указатель на функцию из C#
	//FType F = (FType)(Function);

	float* sum = new float[n];
	float* sum_Dev = NULL;
	float* cut_Dev = NULL;
	float h = (b - a) / n;

	cudaMalloc((void**)&sum_Dev, n * sizeof(float));
	cudaMalloc((void**)&cut_Dev, n * sizeof(float));

	int gridSizeX = (n / BLOCK_SIZE) + ((n % BLOCK_SIZE) > 0 ? 1 : 0);
	dim3 threads(BLOCK_SIZE, 1);
	dim3 blocks(gridSizeX, 1);

	switch (method) {
	case 1:
		{
			SimpsonMethod << <blocks, threads >>>(sum_Dev, cut_Dev, a, b, n);
			break;
		}
	case 2:
		{
			SimpsonMethod_3_8 << <blocks, threads >> >(sum_Dev, cut_Dev, a, b, n);
			break;
		}
	}

	cudaThreadSynchronize();

	cudaMemcpy(sum, sum_Dev, n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(sum_Dev);
	cudaFree(cut_Dev);

	//выполнение редукции результатов, полученных с device
	float result = 0;
	for (int j = 0; j < n; j++)
		result += sum[j];

	cudaFree(sum_Dev);
	cudaFree(cut_Dev);
	delete[] sum;

	switch (method) {
	case 1:
		{
			return (h / 3) * result;
		}
	case 2:
		{
			return h * result;
		}
	}
}

//перегруженная функция выделения памяти и вызова ядра для метода Гауса
double Compute(float a, float b, int n, void* Function, int method, int point) {
	float h = (b - a) / n;
	float* sum = new float[n];
	float* sum_Dev = NULL;
	float* xm_Dev = NULL;
	float* cm_Dev = NULL;

	float* xm = new float[point];
	float* cm = new float[point];

	//выбор значений коэффициентов для расчета (в зависимости от выбранного количества точек)
	switch (point) {
	case 2:
		xm[0] = -0.5773503;
		xm[1] = -0.5773503;
		cm[0] = 1;
		cm[1] = 1;
		break;
	case 3:
		xm[0] = -0.7745967;
		xm[1] = 0;
		xm[2] = 0.7745967;
		cm[0] = 0.5555556;
		cm[1] = 0.8888889;
		cm[2] = 0.5555556;
		break;
	case 4:
		xm[0] = -0.8611363;
		xm[1] = -0.3399810;
		xm[2] = 0.3399810;
		xm[3] = 0.8611363;
		cm[0] = 0.3478548;
		cm[1] = 0.6521451;
		cm[2] = 0.6521451;
		cm[3] = 0.3478548;
	}

	cudaMalloc((void**)&sum_Dev, n * sizeof(float));
	cudaMalloc((void**)&xm_Dev, point * sizeof(float));
	cudaMalloc((void**)&cm_Dev, point * sizeof(float));

	cudaMemcpy(xm_Dev, xm, point * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cm_Dev, cm, point * sizeof(float), cudaMemcpyHostToDevice);

	int gridSizeX = (n / BLOCK_SIZE) + ((n % BLOCK_SIZE) > 0 ? 1 : 0);
	dim3 threads(BLOCK_SIZE, 1);
	dim3 blocks(gridSizeX, 1);

	GaussMethod <<<blocks, threads >>>(sum_Dev, xm_Dev, cm_Dev, a, b, n, point);
	cudaThreadSynchronize();

	cudaMemcpy(sum, sum_Dev, n * sizeof(float), cudaMemcpyDeviceToHost);

	//выполнение редукции результатов, полученных с device
	float result = 0;
	for (int j = 0; j < n; j++)
		result += sum[j];

	cudaFree(sum_Dev);
	cudaFree(xm_Dev);
	cudaFree(cm_Dev);
	delete[] sum;
	delete[] xm;
	delete[] cm;

	return (h / 2) * result;
}

//метод Симпсона
//a b  - интервал, n - число разбиений, * Function - подынтегральная функция
double Simpson_CUDA(float a, float b, int n, void* Function) {
	return Compute(a, b, n, Function, 1);
}

//метод Симпсона 3/8
//a b  - интервал, n - число разбиений, * Function - подынтегральная функция
double Simpson_3_8_CUDA(float a, float b, int n, void* Function) {
	return Compute(a, b, n, Function, 2);
}

//метод Гауса
//a b  - интервал, n - число разбиений, * Function - подынтегральная функция, point - число точек в методе
double Gauss_CUDA(float a, float b, int n, void* Function, int point) {
	return Compute(a, b, n, Function, 3, point);
}
