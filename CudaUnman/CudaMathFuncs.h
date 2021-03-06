#pragma once

#include <stdexcept>
using namespace std;

class MyCudaMathFuncs{
	public: class  DiffEquations{
		public:
			double *_RK2(double t0, double tmax, double tau, int n, double *ynach, void* Function);
			double *_RK4(double t0, double tmax, double tau, int n, double *ynach, void* Function);
			double *_Eiler(double t0, double tmax, double tau, int n, double *ynach, void* Function);

	};

	public:	class Integrals{
		public:
			double _Simpson(float a, float b, int n, void* Function);
			double _Simpson_3_8(float a, float b, int n, void* Function);
			double _Gauss(float a, float b, int n, void* Function, int point);
	};

	public: class Matrix{
		public:
			double *_Mult(double *a, double *b, int M, int N, int Q);
			double *_Transp(double *a, int N, int M);
			double *_MultVector(double *a, double *b, int M, int N);
	};

	public: class Matrix_Seq{
	public:
		double *_Mult(double *a, double *b, int M, int N, int Q);
		double *_Transp(double *a, int N, int M);
		double *_MultVector(double *a, double *b, int M, int N);
	};

};
