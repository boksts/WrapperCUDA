using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

using MathLibCUDA;

namespace CSharp {
    public partial class Form1 : Form {
        MathFuncsIntegral myfun = new MathFuncsIntegral();

        public Form1() {
            InitializeComponent();
        }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //диф.ур.
        public double diffEqu(int i, double y1, double y2, double y3) {
            switch (i) {
                case 0:
                    return -(55 + y3)*y1 + 65*y2;
                case 1:
                    return 0.0785*(y1 - y2);
                case 2:
                    return 0.1*y1;
            }
            return 0;
        }

        private void button1_Click(object sender, EventArgs e) {
            double t0, tmax, tau;
            t0 = Convert.ToDouble(tbT0.Text);
            tmax = Convert.ToDouble(tbTmax.Text);
            tau = Convert.ToDouble(tbTau.Text);
            double[] mass = {};
            double t = t0;
            int n = 3;

            double[] ynach = new double[n];
            ynach[0] = 1.0;
            ynach[1] = 1.0;
            ynach[2] = 0.0;

            MathFuncsDiffEquations myfunc = new MathFuncsDiffEquations();

            if (rbEiler.Checked) {
                richTextBox1.Clear();
                mass = myfunc.Eiler(t0, tmax, tau, n, ynach, diffEqu);
                

                for (int i = 1; i < (int) ((tmax - t0)/tau); i++) {
                    t += tau;
                    richTextBox1.Text += String.Format("time={0} y1={1:f6} y2={2:f6} y3={3:f6}", t, mass[i*3],
                        mass[1 + i*3],
                        mass[2 + i*3]) + Environment.NewLine;
                }
            }
           
            if (rbRK2.Checked) {
                richTextBox1.Clear();
                mass = myfunc.RK2(t0, tmax, tau, n, ynach, diffEqu);
                for (int i = 1; i < (int) ((tmax - t0)/tau); i++) {
                    t += tau;
                    richTextBox1.Text += String.Format("time={0} y1={1:f6} y2={2:f6} y3={3:f6}", t, mass[i*3],
                        mass[1 + i*3],
                        mass[2 + i*3]) + Environment.NewLine;
                }
            }

            if (rbRK4.Checked) {
                richTextBox1.Clear();
            
                mass = myfunc.RK4(t0, tmax, tau, n, ynach, diffEqu);
                for (int i = 1; i < (int) ((tmax - t0)/tau); i++) {
                    t += tau;
                    richTextBox1.Text += String.Format("time={0} y1={1:f6} y2={2:f6} y3={3:f6}", t, mass[i*3],
                        mass[1 + i*3],
                        mass[2 + i*3]) + Environment.NewLine;
                }
            }
        }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
       
        //интегралы

        //подыинтегральная функция для передачи в качестве параметра
        public double F(float x) {
            return (x*x);
        }


        private void button2_Click_1(object sender, EventArgs e) {
            double a, b;
            int n, point;
            a = Convert.ToDouble(tbA.Text);
            b = Convert.ToDouble(tbB.Text);
            n = Convert.ToInt32(tbN.Text);
            point = Convert.ToInt32(numericUpDown1.Value);
            MathFuncsIntegral myfunc = new MathFuncsIntegral();
            
            if (rbSimpson.Checked)
                tbResult.Text = myfunc.Simpson((float) a, (float) b, n,F).ToString();
            if (rbSimpson38.Checked)
                tbResult.Text = myfunc.Simpson_3_8((float) a, (float) b, n, F).ToString();
            if (rbGuassa.Checked)
                tbResult.Text = myfunc.Gauss((float) a, (float) b, n, F, point).ToString();
        }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        //матрицы

        //транспонирование
        private void btnMatrix_Click(object sender, EventArgs e) {
            int N = 4;
            int M = 5;
            double[] a = new double[N*M];
            double[] c;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < M; j++) {
                    a[i*M + j] = i + j;
                }
            
            MathFuncsMatrix myfunc = new MathFuncsMatrix();

            c = myfunc.Transp(a, N, M);

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < M; j++)
                    rtbMatrix.Text += String.Format("{0:f2} ", a[i*M + j]);
                rtbMatrix.Text += Environment.NewLine;
            }

            for (int j = 0; j < M; j++) {
                for (int i = 0; i < N; i++)
                    rtbResult.Text += String.Format("{0:f2} ", c[j*N + i]);
                rtbResult.Text += Environment.NewLine;
            }
        }

        //матрица на вектор
        private void button1_Click_1(object sender, EventArgs e) {
            int N = 4;
            int M = 10;
            double[] a = new double[M*N];
            double[] b = new double[N];
            double[] c;
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    a[i*N + j] = i + j;

            for (int j = 0; j < N; j++)
                b[j] = j;

            MathFuncsMatrix myfunc = new MathFuncsMatrix();

            c = myfunc.MultVector(a, b, M, N);
        }


        //матрица на матрицу
        private void button2_Click(object sender, EventArgs e) {
            int M = 3;
            int N = 4;
            int Q = 2;
            var A = new double[M*N];
            var B = new double[N*Q];
            double[] C;

            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    A[i*N + j] = i + j;

            for (int i = 0; i < N; i++)
                for (int j = 0; j < Q; j++)
                    B[i*Q + j] = j*3.1;

            MathFuncsMatrix myfunc = new MathFuncsMatrix();

            C = myfunc.Mult(A, B, M, N, Q);
        }
    }
}
