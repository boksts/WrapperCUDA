#pragma once
#include "CudaMathFuncs.h"
#include "MiniWrapForCuda.h"

//диф.ур
double* MyCudaMathFuncs::DiffEquations::_Eiler(double t0, double tmax, double tau)
{
	return Eiler_CUDA(t0, tmax, tau);
}
double* MyCudaMathFuncs::DiffEquations::_RK2(double t0, double tmax, double tau)
{
	return RK2_CUDA(t0, tmax, tau);
}

double* MyCudaMathFuncs::DiffEquations::_RK4(double t0, double tmax, double tau)
{
	return RK4_CUDA(t0,tmax,tau);
}


//интегралы
double MyCudaMathFuncs::Integrals::_Simpson(float a, float b, int n, void* Function)
{
	
	return Simpson_CUDA(a, b, n, Function);
}


//матрицы
double *MyCudaMathFuncs::Matrix::_Mult(double *a, double *b, int M, int N, int Q)
{
	return Mult_CUDA(a, b, M,N,Q);
}

double *MyCudaMathFuncs::Matrix::_Transp(double *a, int N, int M)
{
	return Transp_CUDA(a, N, M);
}

double *MyCudaMathFuncs::Matrix::_MultVector(double *a, double *b, int M, int N)
{
	return MultVector_CUDA(a, b, M, N);
}