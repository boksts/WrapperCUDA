#pragma once
double* Eiler_CUDA(double t0, double tmax, double tau, int n, double *ynach, void *Function);
double* RK2_CUDA(double t0, double tmax, double tau, int n, double *ynach, void *Function);
double* RK4_CUDA(double t0, double tmax, double tau, int n, double *ynach, void *Function);

double Simpson_CUDA(float a, float b, int n, void *Function);
double Simpson_3_8_CUDA(float a, float b, int n, void *Function);
double Gauss_CUDA(float a, float b, int n, void *Function, int point);

double *Mult_CUDA(double *a, double *b, int M, int N, int Q);
double *Transp_CUDA(double *a, int N, int M);
double *MultVector_CUDA(double *a, double *b, int M, int N);


double *Mult_Seq(double *a, double *b, int M, int N, int Q);
double *Transp_Seq(double *a, int N, int M);
double *MultVector_Seq(double *a, double *b, int M, int N);