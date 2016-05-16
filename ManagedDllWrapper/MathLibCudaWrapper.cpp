// This is the main DLL file.

#include "stdafx.h"
#include "MathLibCudaWrapper.h"

#include "..\CudaUnman\CudaMathFuncs.h"
#include "..\CudaUnman\CudaMathFuncs.cpp"


//дифференциальные уравнения

array<double> ^MathLibCUDA::MathFuncsDiffEquations::Diffur(double t0, double tmax, double tau, int n, array<double> ^ynach, FDelegate ^ fdelegate, int method){
	//получаем указатель на функцию с диф. ур.
	delegatePointer = (void*)Marshal::GetFunctionPointerForDelegate(fdelegate).ToPointer();

	myCudaClass = new MyCudaMathFuncs::DiffEquations();
	
	//число шагов
	int nn = (int)((tmax - t0) / tau);
	
	//массив для передачи результатов в управляемый код
	array<Double>^ managedArray = gcnew array<Double>(nn * n);

	//массив для результатов из неуправляемого кода
	double *mass;

	//начальные значения
	double *_ynach=new double[n];

	//копирование начальных значений из управляемого кода в неуправляемый
	System::Runtime::InteropServices::Marshal::Copy(ynach, 0, (System::IntPtr)_ynach, n);

	switch (method){
		case 1: mass = myCudaClass->_Eiler(t0, tmax, tau,n,_ynach, delegatePointer); break;
		case 2: mass = myCudaClass->_RK2(t0, tmax, tau, n, _ynach, delegatePointer); break;
		case 3: mass = myCudaClass->_RK4(t0, tmax, tau, n, _ynach, delegatePointer); break;

}
	//копирование результата из неуправляемого кода в управляемый
	System::Runtime::InteropServices::Marshal::Copy((System::IntPtr)mass, managedArray, 0, nn * n);
	return managedArray;
}


array<double> ^MathLibCUDA::MathFuncsDiffEquations::Eiler(double t0, double tmax, double tau, int n, array<double> ^ynach, FDelegate ^ fdelegate)
{
	return Diffur(t0, tmax, tau, n, ynach, fdelegate, 1);
}

array<double> ^MathLibCUDA::MathFuncsDiffEquations::RK2(double t0, double tmax, double tau, int n, array<double> ^ynach, FDelegate ^ fdelegate)
{
	return Diffur(t0, tmax, tau, n, ynach, fdelegate, 2);
}

array<double> ^MathLibCUDA::MathFuncsDiffEquations::RK4(double t0, double tmax, double tau, int n, array<double> ^ynach, FDelegate ^ fdelegate)
{
	return Diffur(t0, tmax, tau, n, ynach, fdelegate, 3);
}



//интегралы

double MathLibCUDA::MathFuncsIntegral::Simpson(float a, float b, int n, FDelegate ^ fdelegate){
	//получаем указатель на подынтегральную функцию
	delegatePointer = (void*)Marshal::GetFunctionPointerForDelegate(fdelegate).ToPointer();
	
	myCudaClass = new MyCudaMathFuncs::Integrals();

	return myCudaClass->_Simpson(a, b, n, delegatePointer);
}
double MathLibCUDA::MathFuncsIntegral::Simpson_3_8(float a, float b, int n, FDelegate ^ fdelegate){
	//получаем указатель на подынтегральную функцию
	delegatePointer = (void*)Marshal::GetFunctionPointerForDelegate(fdelegate).ToPointer();

	myCudaClass = new MyCudaMathFuncs::Integrals();

	return myCudaClass->_Simpson_3_8(a, b, n, delegatePointer);
}
double MathLibCUDA::MathFuncsIntegral::Gauss(float a, float b, int n, FDelegate ^ fdelegate, int point){
	//получаем указатель на подынтегральную функцию
	delegatePointer = (void*)Marshal::GetFunctionPointerForDelegate(fdelegate).ToPointer();

	myCudaClass = new MyCudaMathFuncs::Integrals();

	return myCudaClass->_Gauss(a, b, n, delegatePointer, point);
}


//матрицы

array<double> ^MathLibCUDA::MathFuncsMatrix::Transp(array<double> ^a, int N, int M){
	myCudaClass = new MyCudaMathFuncs::Matrix();

	//массив для передачи результатов в управляемый код
	array<Double>^ managedArray = gcnew array<Double>(N*M);

	double *_a, *_c;
	_a = new double[N*M];
	_c = new double[N*M];

	//копирование начальных данных из управляемого кода в неуправляемый
	System::Runtime::InteropServices::Marshal::Copy(a, 0, (System::IntPtr)_a, N*M);

	_c = myCudaClass->_Transp(_a, N, M);

	//копирование результата из неуправляемого кода в управляемый
	System::Runtime::InteropServices::Marshal::Copy((System::IntPtr)_c, managedArray, 0, N*M);
	return managedArray;
}

array<double> ^MathLibCUDA::MathFuncsMatrix::MultVector(array<double> ^a, array<double> ^b, int M, int N){
	myCudaClass = new MyCudaMathFuncs::Matrix();
	
	//массив для передачи результатов в управляемый код
	array<Double>^ managedArray = gcnew array<Double>(M);

	double *_a, *_b, *_c;
	_a = new double[N*M];
	_b = new double[N];
	_c = new double[M];

	//копирование начальных данных из управляемого кода в неуправляемый
	System::Runtime::InteropServices::Marshal::Copy(a, 0, (System::IntPtr)_a, N*M);
	System::Runtime::InteropServices::Marshal::Copy(b, 0, (System::IntPtr)_b, N);

	_c = myCudaClass->_MultVector(_a, _b, M, N);

	//копирование результата из неуправляемого кода в управляемый
	System::Runtime::InteropServices::Marshal::Copy((System::IntPtr)_c, managedArray, 0, M);

	return managedArray;
}

array<double> ^MathLibCUDA::MathFuncsMatrix::Mult(array<double> ^a, array<double> ^b, int M,int N, int Q){
	myCudaClass = new MyCudaMathFuncs::Matrix();
	
	//массив для передачи результатов в управляемый код
	array<Double>^ managedArray = gcnew array<Double>(M*Q);

	double *_a, *_b, *_c;
	_a = new double[M*N];
	_b = new double[Q*N];
	_c = new double[M*Q];

	//копирование начальных данных из управляемого кода в неуправляемый
	System::Runtime::InteropServices::Marshal::Copy(a, 0, (System::IntPtr)_a, N*M);
	System::Runtime::InteropServices::Marshal::Copy(b, 0, (System::IntPtr)_b, Q*N);

	_c=myCudaClass->_Mult(_a, _b, M,N,Q);

	//копирование результата из неуправляемого кода в управляемый
	System::Runtime::InteropServices::Marshal::Copy((System::IntPtr)_c, managedArray, 0, M*Q);

	return managedArray;
}

///матрицы последовательные

array<double> ^MathLibCUDA::MathFuncsMatrixSeq::Transp(array<double> ^a, int N, int M){
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

array<double> ^MathLibCUDA::MathFuncsMatrixSeq::MultVector(array<double> ^a, array<double> ^b, int M, int N){
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

array<double> ^MathLibCUDA::MathFuncsMatrixSeq::Mult(array<double> ^a, array<double> ^b, int M, int N, int Q){
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

