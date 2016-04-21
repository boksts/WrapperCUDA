// ManagedDllWrapper.h

#pragma once

#include "..\CudaUnman\CudaMathFuncs.h"
#include "..\CudaUnman\CudaMathFuncs.cpp"

using namespace System;
using namespace System::Runtime::InteropServices;


namespace CppWrapper {
    public  ref class MathFuncsDiffEquations
    {
    public:
		array<double> ^RK4(double t0, double tmax, double tau);
	

    private:
	   MyCudaMathFuncs::DiffEquations *myCudaClass; // an instance of class in C++ for CUDA
    };

	public ref class MathFuncsIntegral
	{
	public:
		Delegate^ F;
		void* delegatePointer;

		[UnmanagedFunctionPointerAttribute(CallingConvention::Cdecl)]
		delegate double FDelegate(float);

		double Simpson(float a, float b, int n, FDelegate ^ fdelegate);
		
	private:
		MyCudaMathFuncs::Integrals *myCudaClass; // an instance of class in C++ for CUDA
	};

	public ref class MathFuncsMatrix
	{
	public:
		array<double> ^Mult(array<double> ^a, array<double> ^b, int M,int N,int Q);
		array<double> ^Transp(array<double> ^a, int N, int M);
		array<double> ^MultVector(array<double> ^a, array<double> ^b, int M, int N);

	private:
		MyCudaMathFuncs::Matrix *myCudaClass; // an instance of class in C++ for CUDA
	};

}