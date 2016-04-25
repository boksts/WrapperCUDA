#include <stdio.h>
#include "MiniWrapForCuda.h"
#include <stdlib.h>
#include <math.h>



//тансопнирование матрицы

double *Transp_Seq(double *a, int N, int M){

	int tmp;
	for (int i = 0; i<M; i++)
		for (int j = i; j<N; j++){
		tmp = a[i*N+ j];
		a[i*N + j] = a[j*M + i];
		a[j*M + i] = tmp;
		}

	return a;
}


//умножение матриц
double *Mult_Seq(double *a, double *b, int M, int N, int Q)
{
	double * c = new double[M*Q];

	for (int i = 0; i < M; i++)
		for (int j = 0; j < Q; j++) {
		c[i*Q + j] = 0;
		for (int k = 0; k < N; k++)
			c[i*Q + j] += a[i*N + k] * b[k*Q + j];
		}

	return c;
	delete c;
}


//умножение матрицы на вектор
double *MultVector_Seq(double *a, double *b, int M, int N)
{
	double * c = new double[M];

	for (int j = 0; j<M; j++)
	{
		float sum = 0;
		for (int i = 0; i<N; i++)
			sum += a[j*N + i] * b[i];
		c[j] = sum;
	}

	return c;
	delete c;
}