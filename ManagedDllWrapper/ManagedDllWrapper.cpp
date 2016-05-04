// This is the main DLL file.

#include "stdafx.h"
#include "ManagedDllWrapper.h"

#include "..\CudaUnman\CudaMathFuncs.h"
#include "..\CudaUnman\CudaMathFuncs.cpp"

//���. ��.

array<double> ^CppWrapper::MathFuncsDiffEquations::Diffur(double t0, double tmax, double tau,int method){
	myCudaClass = new  MyCudaMathFuncs::DiffEquations();
	int nn = (int)((tmax - t0) / tau);
	array<Double>^ managedArray = gcnew array<Double>(nn * 3);
	double *mass;
	switch (method){
	case 1: mass = myCudaClass->_Eiler(t0, tmax, tau); break;
	case 2: mass = myCudaClass->_RK2(t0, tmax, tau); break;
	case 3: mass = myCudaClass->_RK4(t0, tmax, tau); break;

	}

	System::Runtime::InteropServices::Marshal::Copy((System::IntPtr)mass, managedArray, 0, nn * 3);
	return managedArray;
}

array<double> ^CppWrapper::MathFuncsDiffEquations::Eiler(double t0, double tmax, double tau)
{
	return Diffur(t0, tmax, tau, 1);
}

array<double> ^CppWrapper::MathFuncsDiffEquations::RK2(double t0, double tmax, double tau)
{
	return Diffur(t0, tmax, tau, 2);
}

array<double> ^CppWrapper::MathFuncsDiffEquations::RK4(double t0, double tmax, double tau)
{   
	return Diffur(t0, tmax, tau, 3);
}



//���������
double CppWrapper::MathFuncsIntegral::Simpson(float a, float b, int n, FDelegate ^ fdelegate){
	delegatePointer = (void*)Marshal::GetFunctionPointerForDelegate(fdelegate).ToPointer();
	myCudaClass = new MyCudaMathFuncs::Integrals();
	return myCudaClass->_Simpson(a, b, n, delegatePointer);
}
double CppWrapper::MathFuncsIntegral::Simpson_3_8(float a, float b, int n, FDelegate ^ fdelegate){
	delegatePointer = (void*)Marshal::GetFunctionPointerForDelegate(fdelegate).ToPointer();
	myCudaClass = new MyCudaMathFuncs::Integrals();
	return myCudaClass->_Simpson_3_8(a, b, n, delegatePointer);
}


//�������
array<double> ^CppWrapper::MathFuncsMatrix::Transp(array<double> ^a,  int N, int M){
	myCudaClass = new MyCudaMathFuncs::Matrix();
	array<Double>^ managedArray = gcnew array<Double>(N*M);
	double *_a, *_c;
	_a = new double[N*M];
	_c = new double[N*M];

	System::Runtime::InteropServices::Marshal::Copy(a, 0, (System::IntPtr)_a, N*M);
	_c = myCudaClass->_Transp(_a,  N, M);
	
	System::Runtime::InteropServices::Marshal::Copy((System::IntPtr)_c, managedArray, 0, N*M);
	return managedArray;
}

array<double> ^CppWrapper::MathFuncsMatrix::MultVector(array<double> ^a, array<double> ^b, int M, int N){
	myCudaClass = new MyCudaMathFuncs::Matrix();
	array<Double>^ managedArray = gcnew array<Double>(M);
	double *_a, *_b, *_c;
	_a = new double[N*M];
	_b = new double[N];
	_c = new double[M];
	System::Runtime::InteropServices::Marshal::Copy(a, 0, (System::IntPtr)_a, N*M);
	System::Runtime::InteropServices::Marshal::Copy(b, 0, (System::IntPtr)_b, N);

	_c = myCudaClass->_MultVector(_a, _b, M, N);
	System::Runtime::InteropServices::Marshal::Copy((System::IntPtr)_c, managedArray, 0, M);

	return managedArray;
}

array<double> ^CppWrapper::MathFuncsMatrix::Mult(array<double> ^a, array<double> ^b, int M,int N, int Q){
	myCudaClass = new MyCudaMathFuncs::Matrix();
	array<Double>^ managedArray = gcnew array<Double>(M*Q);
	double *_a, *_b, *_c;
	_a = new double[M*N];
	_b = new double[Q*N];
	_c = new double[M*Q];
	System::Runtime::InteropServices::Marshal::Copy(a, 0, (System::IntPtr)_a, N*M);
	System::Runtime::InteropServices::Marshal::Copy(b, 0, (System::IntPtr)_b, Q*N);

	_c=myCudaClass->_Mult(_a, _b, M,N,Q);
	System::Runtime::InteropServices::Marshal::Copy((System::IntPtr)_c, managedArray, 0, M*Q);

	return managedArray;
}

//������� ����������������
array<double> ^CppWrapper::MathFuncsMatrixSeq::Transp(array<double> ^a, int N, int M){
	myClass = new MyCudaMathFuncs::Matrix_Seq();
	array<Double>^ managedArray = gcnew array<Double>(N*M);
	double *_a, *_c;
	_a = new double[N*M];
	_c = new double[N*M];
	System::Runtime::InteropServices::Marshal::Copy(a, 0, (System::IntPtr)_a, N*M);
	_c = myClass->_Transp(_a, N, M);

	System::Runtime::InteropServices::Marshal::Copy((System::IntPtr)_c, managedArray, 0, N*M);
	return managedArray;
}

array<double> ^CppWrapper::MathFuncsMatrixSeq::MultVector(array<double> ^a, array<double> ^b, int M, int N){
	myClass = new MyCudaMathFuncs::Matrix_Seq();
	array<Double>^ managedArray = gcnew array<Double>(M);
	double *_a, *_b, *_c;
	_a = new double[N*M];
	_b = new double[N];
	_c = new double[M];
	System::Runtime::InteropServices::Marshal::Copy(a, 0, (System::IntPtr)_a, N*M);
	System::Runtime::InteropServices::Marshal::Copy(b, 0, (System::IntPtr)_b, N);

	_c = myClass->_MultVector(_a, _b, M, N);
	System::Runtime::InteropServices::Marshal::Copy((System::IntPtr)_c, managedArray, 0, M);

	return managedArray;
}

array<double> ^CppWrapper::MathFuncsMatrixSeq::Mult(array<double> ^a, array<double> ^b, int M, int N, int Q){
	myClass = new MyCudaMathFuncs::Matrix_Seq();
	array<Double>^ managedArray = gcnew array<Double>(M*Q);
	double *_a, *_b, *_c;
	_a = new double[M*N];
	_b = new double[Q*N];
	_c = new double[M*Q];
	System::Runtime::InteropServices::Marshal::Copy(a, 0, (System::IntPtr)_a, N*M);
	System::Runtime::InteropServices::Marshal::Copy(b, 0, (System::IntPtr)_b, Q*N);

	_c = myClass->_Mult(_a, _b, M, N, Q);
	System::Runtime::InteropServices::Marshal::Copy((System::IntPtr)_c, managedArray, 0, M*Q);

	return managedArray;
}

