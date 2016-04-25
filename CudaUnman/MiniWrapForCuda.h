#pragma once
double* Eiler_CUDA(double t0, double tmax, double tau);
double* RK2_CUDA(double t0, double tmax, double tau);
double* RK4_CUDA(double t0, double tmax, double tau);

double Simpson_CUDA(float a, float b, int n, void *Function);

double *Mult_CUDA(double *a, double *b, int M, int N, int Q);
double *Transp_CUDA(double *a, int N, int M);
double *MultVector_CUDA(double *a, double *b, int M, int N);


double *Mult_Seq(double *a, double *b, int M, int N, int Q);
double *Transp_Seq(double *a, int N, int M);
double *MultVector_Seq(double *a, double *b, int M, int N);