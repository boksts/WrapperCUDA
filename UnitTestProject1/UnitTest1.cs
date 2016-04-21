using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CppWrapper;
using CSharp;

namespace UnitTestProject1 {
    [TestClass]
    public class UnitTest1 {
      

        [TestMethod]
        public void TestMethod1() {
            MathFuncsMatrix myfunc = new MathFuncsMatrix();
             int N = 4;
            int M = 5;
            double[] a = new double[N*M];
            double[] c;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < M; j++)
                {
                    a[i*M + j] = i+j;
                }

            c = myfunc.Transp(a, N, M);
            Assert.AreEqual(c[3*M+1], a[1*M+3]);
        }

         [TestMethod]
        public void TestMethod2() {
              MathFuncsDiffEquations myfunc = new MathFuncsDiffEquations();
             double[] mass = { };
             mass = myfunc.RK4(0, 0.11, 0.01);
             Assert.AreEqual(Math.Round(mass[3]),1.0);
         }

         [TestMethod]
         public void TestMethod3() {
             MathFuncsDiffEquations myfunc = new MathFuncsDiffEquations();
             double[] mass = myfunc.RK4(0, 0.11, 0.01);
             Assert.IsNotNull(mass);
          
         }

    }
}
