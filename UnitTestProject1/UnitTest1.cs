using System;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MathLibCUDA;
using CSharp;

namespace UnitTestProject1 {

    //тесты на производительность
    [TestClass]
    public class UnitTestPerformance {

        //Умножение матриц
        //Добавлен комментарий
         [TestMethod]
        public void MatrixMult() {
            MathFuncsMatrix mymatr1 = new MathFuncsMatrix();
            MathFuncsMatrixSeq mymatr2 = new MathFuncsMatrixSeq();

            int M = 1000;
            int N = 2000;
            int Q = 100;
            var A = new double[M * N];
            var B = new double[N * Q];

            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    A[i * N + j] = i + j;

            for (int i = 0; i < N; i++)
                for (int j = 0; j < Q; j++)
                    B[i * Q + j] = j * 3.1;

            Stopwatch sw1 = new Stopwatch();
            sw1.Start();
            mymatr1.Mult(A, B, M, N, Q);
            sw1.Stop();
            double secPar = sw1.ElapsedMilliseconds / 1000.0;


            Stopwatch sw2 = new Stopwatch();
            sw2.Start();
            mymatr2.Mult(A, B, M, N, Q);
            sw2.Stop();
            double secSeq = sw2.ElapsedMilliseconds / 1000.0; 

            Assert.IsTrue(secPar<secSeq, "последовательный вариант быстрее!"); 

        }

         //Матрица на вектор
         [TestMethod]
         public void MatrixVector() {
             MathFuncsMatrix mymatr1 = new MathFuncsMatrix();
             MathFuncsMatrixSeq mymatr2 = new MathFuncsMatrixSeq();
             int N = 1000;
             int M = 500;
             double[] a = new double[M * N];
             double[] b = new double[N];
             for (int i = 0; i < M; i++)
                 for (int j = 0; j < N; j++)
                     a[i * N + j] = i + j;

             for (int j = 0; j < N; j++)
                 b[j] = j;

             Stopwatch sw1 = new Stopwatch();
             sw1.Start();
             mymatr1.MultVector(a, b, M, N); 
             sw1.Stop();
             double secPar = sw1.ElapsedMilliseconds / 1000.0;


             Stopwatch sw2 = new Stopwatch();
             sw2.Start();
             mymatr2.MultVector(a, b, M, N); 
             sw2.Stop();
             double secSeq = sw2.ElapsedMilliseconds / 1000.0;

             Assert.IsTrue(secPar < secSeq, "последовательный вариант быстрее!");

         }

         //Транспонирование
         [TestMethod]
         public void MatrixTransp() {
             MathFuncsMatrix mymatr1 = new MathFuncsMatrix();
             MathFuncsMatrixSeq mymatr2 = new MathFuncsMatrixSeq();
             int N = 1000;
             int M = 500;
             double[] a = new double[N * M];
             for (int i = 0; i < N; i++)
                 for (int j = 0; j < M; j++) {
                     a[i * M + j] = i + j;
                 }

             Stopwatch sw1 = new Stopwatch();
             sw1.Start();
             mymatr1.Transp(a, N, M);
             sw1.Stop();
             double secPar = sw1.ElapsedMilliseconds / 1000.0;


             Stopwatch sw2 = new Stopwatch();
             sw2.Start();
             mymatr2.Transp(a, N, M);
             sw2.Stop();
             double secSeq = sw2.ElapsedMilliseconds / 1000.0;

             Assert.IsTrue(secPar < secSeq, "последовательный вариант быстрее!");

         }
       
    }

    //тесты на достоверность результатов
     [TestClass]
    public class UnitTestResults {
      

        //матрицы
        MathFuncsMatrix mymatr = new MathFuncsMatrix();

        //транспонирование
        [TestMethod]
        public void MatrixTransp() {
            
            int N = 4;
            int M = 5;
            double[] a = new double[N*M];
            double[] c;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < M; j++)
                {
                    a[i*M + j] = i+j;
                }

            c = mymatr.Transp(a, N, M);
            Assert.AreEqual(c[3*M+1], a[1*M+3]);
        }

        //Умножение матрицы на вектор
        [TestMethod]
        public void MatrixMultVector() {
            int N = 4;
            int M = 10;
            double[] a = new double[M * N];
            double[] b = new double[N];
            double[] c;
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    a[i * N + j] = i + j;

            for (int j = 0; j < N; j++)
                b[j] = j;

            c = mymatr.MultVector(a, b, M, N);
            Assert.AreEqual(c[5], 44);
        }

        //Умножение матрицы на матрицу
        [TestMethod]
        public void MatrixMultMatrix() {
            int M = 3;
            int N = 4;
            int Q = 2;
            var A = new double[M * N];
            var B = new double[N * Q];
            double[] C;

            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    A[i * N + j] = i + j;

            for (int i = 0; i < N; i++)
                for (int j = 0; j < Q; j++)
                    B[i * Q + j] = j * 3.1;

            C = mymatr.Mult(A, B, M, N, Q);
            Assert.AreEqual(C[5], 43.4);

        }


        //дифф. уравнения
         MathFuncsDiffEquations mydiff = new MathFuncsDiffEquations();
         public double diffEqu(int i, double y1, double y2, double y3) {
             switch (i) {
                 case 0:
                     return -(55 + y3) * y1 + 65 * y2;
                 case 1:
                     return 0.0785 * (y1 - y2);
                 case 2:
                     return 0.1 * y1;
             }
             return 0;
         }

         //Эйлер
         [TestMethod]
         public void DiffEiler() {
             double[] mass = { };
             double t0 = 0.0, tmax = 0.1, tau = 0.01;

             int n = 3;

             double[] ynach = new double[n];
             ynach[0] = 1.0;
             ynach[1] = 1.0;
             ynach[2] = 0.0;
             mass = mydiff.Eiler(t0, tmax, tau, n, ynach, diffEqu);
             int nn = (int)((tmax - t0) / tau) - 1;

             Assert.AreEqual(Math.Round(mass[nn * n], 6), 1.182412);
             Assert.AreEqual(Math.Round(mass[nn * n + 1], 6), 1.001024);
             Assert.AreEqual(Math.Round(mass[nn * n + 2], 6), 0.010308);
         }


        //РК-2
         [TestMethod]
        public void DiffRK2() {
             double[] mass = { };
             double t0 = 0.0, tmax = 0.1, tau = 0.01;
             int n = 3;

             double[] ynach = new double[n];
             ynach[0] = 1.0;
             ynach[1] = 1.0;
             ynach[2] = 0.0;
             mass = mydiff.RK2(t0, tmax, tau, n, ynach, diffEqu);
             int nn = (int)((tmax - t0) / tau)-1;

             Assert.AreEqual(Math.Round(mass[nn*n], 6), 1.180697);
             Assert.AreEqual(Math.Round(mass[nn*n + 1], 6), 1.001026);
             Assert.AreEqual(Math.Round(mass[nn*n + 2], 6), 0.010311);
         }

         //РК-4
         [TestMethod]
         public void DiffRK4() {
             double[] mass = { };
             double t0 = 0.0, tmax = 0.1, tau = 0.01;
             int n = 3;

             double[] ynach = new double[n];
             ynach[0] = 1.0;
             ynach[1] = 1.0;
             ynach[2] = 0.0;
             mass = mydiff.RK4(t0, tmax, tau,n,ynach,diffEqu);

             int nn = (int)((tmax - t0) / tau) - 1;
             Assert.AreEqual(Math.Round(mass[nn * n], 6), 1.181266);
             Assert.AreEqual(Math.Round(mass[nn * n + 1], 6), 1.001026);
             Assert.AreEqual(Math.Round(mass[nn * n + 2], 6), 0.010310);
         }

         //интегралы
         MathFuncsIntegral integr = new MathFuncsIntegral();

         public double integrFunc(float x)
         {
             return x*x;
         }

         //Симпсон
         [TestMethod]
         public void integrSimpson()
         {
             Assert.AreEqual(Math.Round(integr.Simpson(0, 10, 100000, integrFunc), 6), 333.323242);
         }

         //Симпсон_3_8
         [TestMethod]
         public void integrSimpson_3_8()
         {
             Assert.AreEqual(Math.Round(integr.Simpson_3_8(0, 10, 100000, integrFunc), 6), 333.343445);
         }

         //Гаусс 2 точки
         [TestMethod]
         public void Gauss2()
         {
             Assert.AreEqual(Math.Round(integr.Gauss(0, 10, 100000, integrFunc, 2), 6), 333.330688);
         }

         //Гаусс 3 точки
         [TestMethod]
         public void Gauss3()
         {
             Assert.AreEqual(Math.Round(integr.Gauss(0, 10, 100000, integrFunc, 3), 6), 333.333313);
         }

         //Гаусс 4 точки
         [TestMethod]
         public void Gauss4()
         {
             Assert.AreEqual(Math.Round(integr.Gauss(0, 10, 100000, integrFunc, 4), 6), 333.333282);
         }
    }
}
