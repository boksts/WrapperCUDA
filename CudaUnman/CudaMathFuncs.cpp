#pragma once
#include "CudaMathFuncs.h"
#include "MiniWrapForCuda.h"

//==================================
//���������������� ���������

//����� ������
double* MyCudaMathFuncs::DiffEquations::_Eiler(double t0, double tmax, double tau, int n, double *ynach, void* Function)
{
	return Eiler_CUDA(t0, tmax, tau,n,ynach, Function);
}

//����� �����-����� 2
double* MyCudaMathFuncs::DiffEquations::_RK2(double t0, double tmax, double tau, int n, double *ynach, void* Function)
{
	return RK2_CUDA(t0, tmax, tau, n, ynach, Function);
}

//����� �����-����� 4
double* MyCudaMathFuncs::DiffEquations::_RK4(double t0, double tmax, double tau, int n, double *ynach, void* Function)
{
	return RK4_CUDA(t0, tmax, tau, n, ynach, Function);
}


//===================================
//���������

//����� ��������
double MyCudaMathFuncs::Integrals::_Simpson(float a, float b, int n, void* Function)
{
	return Simpson_CUDA(a, b, n, Function);
}

//����� �������� 3/8
double MyCudaMathFuncs::Integrals::_Simpson_3_8(float a, float b, int n, void* Function)
{
	return Simpson_3_8_CUDA(a, b, n, Function);
}

//����� �����
double MyCudaMathFuncs::Integrals::_Gauss(float a, float b, int n, void* Function, int point)
{
	return Gauss_CUDA(a, b, n, Function, point);
}


//======================================
//�������

//��������� ������
double *MyCudaMathFuncs::Matrix::_Mult(double *a, double *b, int M, int N, int Q)
{
	return Mult_CUDA(a, b, M,N,Q);
}

//����������������
double *MyCudaMathFuncs::Matrix::_Transp(double *a, int N, int M)
{
	return Transp_CUDA(a, N, M);
}

//��������� ������� �� ������
double *MyCudaMathFuncs::Matrix::_MultVector(double *a, double *b, int M, int N)
{
	return MultVector_CUDA(a, b, M, N);
}


//========================================
//������� ����������������
double *MyCudaMathFuncs::Matrix_Seq::_Mult(double *a, double *b, int M, int N, int Q)
{
	return Mult_Seq(a, b, M, N, Q);
}

double *MyCudaMathFuncs::Matrix_Seq::_Transp(double *a, int N, int M)
{
	return Transp_Seq(a, N, M);
}

double *MyCudaMathFuncs::Matrix_Seq::_MultVector(double *a, double *b, int M, int N)
{
	return MultVector_Seq(a, b, M, N);
}