#pragma once

#include <stdexcept>
using namespace std;

class MyCudaMathFuncs{
	public: class  DiffEquations{
		public:
			double *_RK2(double t0, double tmax, double tau);
			double *_RK4(double t0, double tmax, double tau);
			double *_Eiler(double t0, double tmax, double tau);

	};

	public:	class Integrals{
		public:
			double _Simpson(float a, float b, int n, void* Function);

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
