// ManagedDllWrapper.h

#pragma once

#include "..\CudaUnman\CudaMathFuncs.h"
#include "..\CudaUnman\CudaMathFuncs.cpp"

using namespace System;
using namespace System::Runtime::InteropServices;


namespace CppWrapper {

	/// <summary>
	/// Предоставляет возможность решать систему дифференциальных уравнений тремя методами</summary>	
	public ref class MathFuncsDiffEquations
	{
		Delegate^ F;
		void* delegatePointer;
	
		public:
		[UnmanagedFunctionPointerAttribute(CallingConvention::Cdecl)]
		delegate double FDelegate(int, double, double, double);

		/// <summary>
		/// Метод Эйлера</summary>	
		/// <param name="t0"> начальное время</param> 	
		///<param name="tmax"> конечное время</param> 
		///<param name="tau"> шаг</param>
		/// <param name="n"> число уравнений</param>
		/// <param name="ynach"> массив начальных значений</param>
		/// <param name="fdelegate"> дифференциальные уравнения 
		///<code> 
		///Пример функции:
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
		/// Метод Рунге-Кутта 2
		/// </summary>
		/// <param name="t0"> начальное время</param> 	
		///<param name="tmax"> конечное время</param> 
		///<param name="tau"> шаг</param>
		/// <param name="n"> число уравнений</param>
		/// <param name="ynach"> массив начальных значений</param>
		/// <param name="fdelegate"> дифференциальные уравнения
		///<code> 
		///Пример функции:
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
		/// Метод Рунге-Кутта 4
		/// </summary>
		/// <param name="t0"> начальное время</param> 	
		///<param name="tmax"> конечное время</param> 
		///<param name="tau"> шаг</param>
		/// <param name="n"> число уравнений</param>
		/// <param name="ynach"> массив начальных значений</param>
		/// <param name="fdelegate"> дифференциальные уравнения
		///<code> 
		///Пример функции:
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
	/// Предоставляет возможность находить интеграл тремя методами</summary>	
	public ref class MathFuncsIntegral
	{
		Delegate^ F;
		void* delegatePointer;
	
		public:
		[UnmanagedFunctionPointerAttribute(CallingConvention::Cdecl)]
		delegate double FDelegate(float);

		/// <summary>
		/// Метод Симпсона
		/// </summary>
		/// <param name="a"> интервал от</param> 	
		///<param name="b"> интервал до</param> 
		/// <param name="n"> число разбиений</param>
		/// <param name="fdelegate"> подынтегральная функция
		///<code> 
		///Пример функции:
		///public double F(float x) {
		///  	return (x*x);
		///}
		///</code></param>
		double Simpson(float a, float b, int n, FDelegate ^ fdelegate);

		/// <summary>
		/// Метод Симпсона 3/8
		/// </summary>
		/// <param name="a"> интервал от</param> 	
		///<param name="b"> интервал до</param> 
		/// <param name="n"> число разбиений</param>
		/// <param name="fdelegate"> подынтегральная функция
		///<code> 
		///Пример функции:
		///public double F(float x) {
		///  	return (x*x);
		///}
		///</code></param>
		double Simpson_3_8(float a, float b, int n, FDelegate ^ fdelegate);

		/// <summary>
		/// Метод Гауса
		/// </summary>
		/// <param name="a"> интервал от</param> 	
		/// <param name="b"> интервал до</param> 
		/// <param name="n"> число разбиений</param>
		/// <param name="fdelegate"> подынтегральная функция</param>
		/// <param name="point"> число точек в методе
		///<code> 
		///Пример функции:
		///public double F(float x) {
		///  	return (x*x);
		///}
		///</code></param>
		double Gauss(float a, float b, int n, FDelegate ^ fdelegate, int point);

		private:
		MyCudaMathFuncs::Integrals *myCudaClass;
	};

	/// <summary>
	/// Предоставляет возможность транспонировать матрицу, умножать матрицу на матрицу и матрицу на вектор</summary>	
	public ref class MathFuncsMatrix
	{
		public:

		/// <summary>
		/// Умножение матриц
		/// </summary>
		/// <param name="a"> a[M][N] - 1 матрица</param> 	
		///<param name="b"> b[M][Q]- 2 матрица</param> 
		/// <param name="M"> размерность</param>
		/// <param name="N"> размерность</param>
		/// <param name="Q"> размерность</param>
		array<double> ^Mult(array<double> ^a, array<double> ^b, int M,int N,int Q);

		/// <summary>
		/// Транспонирование матрицы
		/// </summary>
		/// <param name="a"> a[N][M] - исходная матрица</param> 	
		/// <param name="N"> размерность</param>
		/// <param name="M"> размерность</param>
		array<double> ^Transp(array<double> ^a, int N, int M);

		/// <summary>
		/// Умножение матрицы на вектор
		/// </summary>
		/// <param name="a"> a[M][N] - матрица</param> 	
		///<param name="b"> b[N]- вектор</param> 
		/// <param name="M"> размерность</param>
		/// <param name="N"> размерность</param>
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
