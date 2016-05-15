// ManagedDllWrapper.h

#pragma once

#include "..\CudaUnman\CudaMathFuncs.h"
#include "..\CudaUnman\CudaMathFuncs.cpp"

using namespace System;
using namespace System::Runtime::InteropServices;


namespace CppWrapper {

	/// <summary>
	/// ������������� ����������� ������ ������� ���������������� ��������� ����� ��������</summary>	
	public ref class MathFuncsDiffEquations
	{
		Delegate^ F;
		void* delegatePointer;
	
		public:
		[UnmanagedFunctionPointerAttribute(CallingConvention::Cdecl)]
		delegate double FDelegate(int, double, double, double);

		/// <summary>
		/// ����� ������</summary>	
		/// <param name="t0"> ��������� �����</param> 	
		///<param name="tmax"> �������� �����</param> 
		///<param name="tau"> ���</param>
		/// <param name="n"> ����� ���������</param>
		/// <param name="ynach"> ������ ��������� ��������</param>
		/// <param name="fdelegate"> ���������������� ��������� 
		///<code> 
		///������ �������:
		///public double diffEqu(int i, double y1, double y2, double y3) {
		///	switch (i) {
		///	case 0:
		///		return -(55 + y3)*y1 + 65 * y2;
		///	case 1:
		///		return 0.0785*(y1 - y2);
		///	case 2:
		///		return 0.1*y1;
		///	}
		///	return 0;
		///}
		///</code></param>
		array<double> ^Eiler(double t0, double tmax, double tau, int n, array<double> ^ynach, FDelegate ^fdelegate);
		
		/// <summary>
		/// ����� �����-����� 2
		/// </summary>
		/// <param name="t0"> ��������� �����</param> 	
		///<param name="tmax"> �������� �����</param> 
		///<param name="tau"> ���</param>
		/// <param name="n"> ����� ���������</param>
		/// <param name="ynach"> ������ ��������� ��������</param>
		/// <param name="fdelegate"> ���������������� ���������
		///<code> 
		///������ �������:
		///public double diffEqu(int i, double y1, double y2, double y3) {
		///	switch (i) {
		///	case 0:
		///		return -(55 + y3)*y1 + 65 * y2;
		///	case 1:
		///		return 0.0785*(y1 - y2);
		///	case 2:
		///		return 0.1*y1;
		///	}
		///	return 0;
		///}
		///</code></param>
		array<double> ^RK2(double t0, double tmax, double tau, int n, array<double> ^ynach, FDelegate ^fdelegate);

		/// <summary>
		/// ����� �����-����� 4
		/// </summary>
		/// <param name="t0"> ��������� �����</param> 	
		///<param name="tmax"> �������� �����</param> 
		///<param name="tau"> ���</param>
		/// <param name="n"> ����� ���������</param>
		/// <param name="ynach"> ������ ��������� ��������</param>
		/// <param name="fdelegate"> ���������������� ���������
		///<code> 
		///������ �������:
		///public double diffEqu(int i, double y1, double y2, double y3) {
		///	switch (i) {
		///	case 0:
		///		return -(55 + y3)*y1 + 65 * y2;
		///	case 1:
		///		return 0.0785*(y1 - y2);
		///	case 2:
		///		return 0.1*y1;
		///	}
		///	return 0;
		///}
		///</code></param>
		array<double> ^RK4(double t0, double tmax, double tau, int n, array<double> ^ynach, FDelegate ^fdelegate);

		private:
		array<double> ^Diffur(double t0, double tmax, double tau, int n, array<double> ^ynach, FDelegate ^ fdelegate, int method);
		MyCudaMathFuncs::DiffEquations *myCudaClass;
	};


	/// <summary>
	/// ������������� ����������� �������� �������� ����� ��������</summary>	
	public ref class MathFuncsIntegral
	{
		Delegate^ F;
		void* delegatePointer;
	
		public:
		[UnmanagedFunctionPointerAttribute(CallingConvention::Cdecl)]
		delegate double FDelegate(float);

		/// <summary>
		/// ����� ��������
		/// </summary>
		/// <param name="a"> �������� ��</param> 	
		///<param name="b"> �������� ��</param> 
		/// <param name="n"> ����� ���������</param>
		/// <param name="fdelegate"> ��������������� �������
		///<code> 
		///������ �������:
		///public double F(float x) {
		///  	return (x*x);
		///}
		///</code></param>
		double Simpson(float a, float b, int n, FDelegate ^ fdelegate);

		/// <summary>
		/// ����� �������� 3/8
		/// </summary>
		/// <param name="a"> �������� ��</param> 	
		///<param name="b"> �������� ��</param> 
		/// <param name="n"> ����� ���������</param>
		/// <param name="fdelegate"> ��������������� �������
		///<code> 
		///������ �������:
		///public double F(float x) {
		///  	return (x*x);
		///}
		///</code></param>
		double Simpson_3_8(float a, float b, int n, FDelegate ^ fdelegate);

		/// <summary>
		/// ����� �����
		/// </summary>
		/// <param name="a"> �������� ��</param> 	
		/// <param name="b"> �������� ��</param> 
		/// <param name="n"> ����� ���������</param>
		/// <param name="fdelegate"> ��������������� �������</param>
		/// <param name="point"> ����� ����� � ������
		///<code> 
		///������ �������:
		///public double F(float x) {
		///  	return (x*x);
		///}
		///</code></param>
		double Gauss(float a, float b, int n, FDelegate ^ fdelegate, int point);

		private:
		MyCudaMathFuncs::Integrals *myCudaClass;
	};

	/// <summary>
	/// ������������� ����������� ��������������� �������, �������� ������� �� ������� � ������� �� ������</summary>	
	public ref class MathFuncsMatrix
	{
		public:

		/// <summary>
		/// ��������� ������
		/// </summary>
		/// <param name="a"> a[M][N] - 1 �������</param> 	
		///<param name="b"> b[M][Q]- 2 �������</param> 
		/// <param name="M"> �����������</param>
		/// <param name="N"> �����������</param>
		/// <param name="Q"> �����������</param>
		array<double> ^Mult(array<double> ^a, array<double> ^b, int M,int N,int Q);

		/// <summary>
		/// ���������������� �������
		/// </summary>
		/// <param name="a"> a[N][M] - �������� �������</param> 	
		/// <param name="N"> �����������</param>
		/// <param name="M"> �����������</param>
		array<double> ^Transp(array<double> ^a, int N, int M);

		/// <summary>
		/// ��������� ������� �� ������
		/// </summary>
		/// <param name="a"> a[M][N] - �������</param> 	
		///<param name="b"> b[N]- ������</param> 
		/// <param name="M"> �����������</param>
		/// <param name="N"> �����������</param>
		array<double> ^MultVector(array<double> ^a, array<double> ^b, int M, int N);

		private:
		MyCudaMathFuncs::Matrix *myCudaClass;
	};

	public ref class MathFuncsMatrixSeq
	{
		public:
		array<double> ^Mult(array<double> ^a, array<double> ^b, int M, int N, int Q);
		array<double> ^Transp(array<double> ^a, int N, int M);
		array<double> ^MultVector(array<double> ^a, array<double> ^b, int M, int N);

		private:
		MyCudaMathFuncs::Matrix_Seq *myClass; // an instance of class in C++ for CUDA
	};

}
