﻿namespace CSharp
{
    partial class Form1
    {
        /// <summary>
        /// Требуется переменная конструктора.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Освободить все используемые ресурсы.
        /// </summary>
        /// <param name="disposing">истинно, если управляемый ресурс должен быть удален; иначе ложно.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Код, автоматически созданный конструктором форм Windows

        /// <summary>
        /// Обязательный метод для поддержки конструктора - не изменяйте
        /// содержимое данного метода при помощи редактора кода.
        /// </summary>
        private void InitializeComponent()
        {
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.label13 = new System.Windows.Forms.Label();
            this.button2 = new System.Windows.Forms.Button();
            this.button1 = new System.Windows.Forms.Button();
            this.label12 = new System.Windows.Forms.Label();
            this.label11 = new System.Windows.Forms.Label();
            this.label10 = new System.Windows.Forms.Label();
            this.rtbResult = new System.Windows.Forms.RichTextBox();
            this.rtbMatrix = new System.Windows.Forms.RichTextBox();
            this.btnMatrix = new System.Windows.Forms.Button();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.tbN = new System.Windows.Forms.TextBox();
            this.tbB = new System.Windows.Forms.TextBox();
            this.tbA = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.tbResult = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.rbGuassa = new System.Windows.Forms.RadioButton();
            this.rbSimpson38 = new System.Windows.Forms.RadioButton();
            this.rbSimpson = new System.Windows.Forms.RadioButton();
            this.label2 = new System.Windows.Forms.Label();
            this.btnIntegral = new System.Windows.Forms.Button();
            this.tabPage3 = new System.Windows.Forms.TabPage();
            this.richTextBox1 = new System.Windows.Forms.RichTextBox();
            this.tbTmax = new System.Windows.Forms.TextBox();
            this.tbTau = new System.Windows.Forms.TextBox();
            this.tbT0 = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.rbRK4 = new System.Windows.Forms.RadioButton();
            this.rbRK2 = new System.Windows.Forms.RadioButton();
            this.rbEiler = new System.Windows.Forms.RadioButton();
            this.label1 = new System.Windows.Forms.Label();
            this.btnDiffEq = new System.Windows.Forms.Button();
            this.numericUpDown1 = new System.Windows.Forms.NumericUpDown();
            this.label14 = new System.Windows.Forms.Label();
            this.tabControl1.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.tabPage2.SuspendLayout();
            this.tabPage3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown1)).BeginInit();
            this.SuspendLayout();
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.tabPage1);
            this.tabControl1.Controls.Add(this.tabPage2);
            this.tabControl1.Controls.Add(this.tabPage3);
            this.tabControl1.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.tabControl1.Location = new System.Drawing.Point(12, 12);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(540, 422);
            this.tabControl1.TabIndex = 2;
            // 
            // tabPage1
            // 
            this.tabPage1.Controls.Add(this.label13);
            this.tabPage1.Controls.Add(this.button2);
            this.tabPage1.Controls.Add(this.button1);
            this.tabPage1.Controls.Add(this.label12);
            this.tabPage1.Controls.Add(this.label11);
            this.tabPage1.Controls.Add(this.label10);
            this.tabPage1.Controls.Add(this.rtbResult);
            this.tabPage1.Controls.Add(this.rtbMatrix);
            this.tabPage1.Controls.Add(this.btnMatrix);
            this.tabPage1.Location = new System.Drawing.Point(4, 29);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage1.Size = new System.Drawing.Size(532, 389);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "Матрицы";
            this.tabPage1.UseVisualStyleBackColor = true;
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.label13.Location = new System.Drawing.Point(6, 267);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(124, 25);
            this.label13.TabIndex = 11;
            this.label13.Text = "Умножение";
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(32, 305);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(186, 51);
            this.button2.TabIndex = 10;
            this.button2.Text = "Матрица на матрицу";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(271, 305);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(186, 51);
            this.button1.TabIndex = 9;
            this.button1.Text = "Матрица на вектор";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click_1);
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.label12.Location = new System.Drawing.Point(6, 8);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(185, 25);
            this.label12.TabIndex = 8;
            this.label12.Text = "Транспонирование";
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(336, 28);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(110, 20);
            this.label11.TabIndex = 4;
            this.label11.Text = "Полученная";
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(74, 28);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(89, 20);
            this.label10.TabIndex = 3;
            this.label10.Text = "Исходная";
            // 
            // rtbResult
            // 
            this.rtbResult.Location = new System.Drawing.Point(271, 51);
            this.rtbResult.Name = "rtbResult";
            this.rtbResult.Size = new System.Drawing.Size(229, 135);
            this.rtbResult.TabIndex = 2;
            this.rtbResult.Text = "";
            // 
            // rtbMatrix
            // 
            this.rtbMatrix.Location = new System.Drawing.Point(18, 51);
            this.rtbMatrix.Name = "rtbMatrix";
            this.rtbMatrix.Size = new System.Drawing.Size(229, 135);
            this.rtbMatrix.TabIndex = 1;
            this.rtbMatrix.Text = "";
            // 
            // btnMatrix
            // 
            this.btnMatrix.Location = new System.Drawing.Point(155, 202);
            this.btnMatrix.Name = "btnMatrix";
            this.btnMatrix.Size = new System.Drawing.Size(186, 51);
            this.btnMatrix.TabIndex = 0;
            this.btnMatrix.Text = "Транспонирование";
            this.btnMatrix.UseVisualStyleBackColor = true;
            this.btnMatrix.Click += new System.EventHandler(this.btnMatrix_Click);
            // 
            // tabPage2
            // 
            this.tabPage2.Controls.Add(this.label14);
            this.tabPage2.Controls.Add(this.numericUpDown1);
            this.tabPage2.Controls.Add(this.tbN);
            this.tabPage2.Controls.Add(this.tbB);
            this.tabPage2.Controls.Add(this.tbA);
            this.tabPage2.Controls.Add(this.label6);
            this.tabPage2.Controls.Add(this.label5);
            this.tabPage2.Controls.Add(this.label4);
            this.tabPage2.Controls.Add(this.tbResult);
            this.tabPage2.Controls.Add(this.label3);
            this.tabPage2.Controls.Add(this.rbGuassa);
            this.tabPage2.Controls.Add(this.rbSimpson38);
            this.tabPage2.Controls.Add(this.rbSimpson);
            this.tabPage2.Controls.Add(this.label2);
            this.tabPage2.Controls.Add(this.btnIntegral);
            this.tabPage2.Location = new System.Drawing.Point(4, 29);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage2.Size = new System.Drawing.Size(532, 389);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "Интегралы";
            this.tabPage2.UseVisualStyleBackColor = true;
            // 
            // tbN
            // 
            this.tbN.Location = new System.Drawing.Point(404, 79);
            this.tbN.Name = "tbN";
            this.tbN.Size = new System.Drawing.Size(100, 26);
            this.tbN.TabIndex = 18;
            this.tbN.Text = "100000";
            // 
            // tbB
            // 
            this.tbB.Location = new System.Drawing.Point(404, 47);
            this.tbB.Name = "tbB";
            this.tbB.Size = new System.Drawing.Size(100, 26);
            this.tbB.TabIndex = 17;
            this.tbB.Text = "10";
            // 
            // tbA
            // 
            this.tbA.Location = new System.Drawing.Point(404, 15);
            this.tbA.Name = "tbA";
            this.tbA.Size = new System.Drawing.Size(100, 26);
            this.tbA.TabIndex = 16;
            this.tbA.Text = "0";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(373, 80);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(21, 20);
            this.label6.TabIndex = 15;
            this.label6.Text = "N";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(376, 47);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(18, 20);
            this.label5.TabIndex = 14;
            this.label5.Text = "b";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(376, 15);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(18, 20);
            this.label4.TabIndex = 13;
            this.label4.Text = "a";
            // 
            // tbResult
            // 
            this.tbResult.Location = new System.Drawing.Point(226, 227);
            this.tbResult.Name = "tbResult";
            this.tbResult.ReadOnly = true;
            this.tbResult.Size = new System.Drawing.Size(161, 26);
            this.tbResult.TabIndex = 12;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(123, 230);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(97, 20);
            this.label3.TabIndex = 11;
            this.label3.Text = "Результат";
            // 
            // rbGuassa
            // 
            this.rbGuassa.AutoSize = true;
            this.rbGuassa.Font = new System.Drawing.Font("Microsoft Sans Serif", 11F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.rbGuassa.Location = new System.Drawing.Point(18, 126);
            this.rbGuassa.Name = "rbGuassa";
            this.rbGuassa.Size = new System.Drawing.Size(91, 28);
            this.rbGuassa.TabIndex = 10;
            this.rbGuassa.TabStop = true;
            this.rbGuassa.Text = "Гаусса";
            this.rbGuassa.UseVisualStyleBackColor = true;
            // 
            // rbSimpson38
            // 
            this.rbSimpson38.AutoSize = true;
            this.rbSimpson38.Font = new System.Drawing.Font("Microsoft Sans Serif", 11F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.rbSimpson38.Location = new System.Drawing.Point(18, 76);
            this.rbSimpson38.Name = "rbSimpson38";
            this.rbSimpson38.Size = new System.Drawing.Size(151, 28);
            this.rbSimpson38.TabIndex = 9;
            this.rbSimpson38.TabStop = true;
            this.rbSimpson38.Text = "Симпсона 3/8";
            this.rbSimpson38.UseVisualStyleBackColor = true;
            // 
            // rbSimpson
            // 
            this.rbSimpson.AutoSize = true;
            this.rbSimpson.Font = new System.Drawing.Font("Microsoft Sans Serif", 11F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.rbSimpson.Location = new System.Drawing.Point(18, 28);
            this.rbSimpson.Name = "rbSimpson";
            this.rbSimpson.Size = new System.Drawing.Size(121, 28);
            this.rbSimpson.TabIndex = 8;
            this.rbSimpson.TabStop = true;
            this.rbSimpson.Text = "Симпсона";
            this.rbSimpson.UseVisualStyleBackColor = true;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.label2.Location = new System.Drawing.Point(3, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(161, 25);
            this.label2.TabIndex = 7;
            this.label2.Text = "Метод решения";
            // 
            // btnIntegral
            // 
            this.btnIntegral.Font = new System.Drawing.Font("Microsoft Sans Serif", 15F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.btnIntegral.Location = new System.Drawing.Point(153, 306);
            this.btnIntegral.Name = "btnIntegral";
            this.btnIntegral.Size = new System.Drawing.Size(213, 62);
            this.btnIntegral.TabIndex = 6;
            this.btnIntegral.Text = "Решить";
            this.btnIntegral.UseVisualStyleBackColor = true;
            this.btnIntegral.Click += new System.EventHandler(this.button2_Click_1);
            // 
            // tabPage3
            // 
            this.tabPage3.Controls.Add(this.richTextBox1);
            this.tabPage3.Controls.Add(this.tbTmax);
            this.tabPage3.Controls.Add(this.tbTau);
            this.tabPage3.Controls.Add(this.tbT0);
            this.tabPage3.Controls.Add(this.label9);
            this.tabPage3.Controls.Add(this.label8);
            this.tabPage3.Controls.Add(this.label7);
            this.tabPage3.Controls.Add(this.rbRK4);
            this.tabPage3.Controls.Add(this.rbRK2);
            this.tabPage3.Controls.Add(this.rbEiler);
            this.tabPage3.Controls.Add(this.label1);
            this.tabPage3.Controls.Add(this.btnDiffEq);
            this.tabPage3.Location = new System.Drawing.Point(4, 29);
            this.tabPage3.Name = "tabPage3";
            this.tabPage3.Size = new System.Drawing.Size(532, 389);
            this.tabPage3.TabIndex = 2;
            this.tabPage3.Text = "Дифф. уравнения";
            this.tabPage3.UseVisualStyleBackColor = true;
            // 
            // richTextBox1
            // 
            this.richTextBox1.Location = new System.Drawing.Point(8, 132);
            this.richTextBox1.Name = "richTextBox1";
            this.richTextBox1.ReadOnly = true;
            this.richTextBox1.Size = new System.Drawing.Size(514, 176);
            this.richTextBox1.TabIndex = 13;
            this.richTextBox1.Text = "";
            // 
            // tbTmax
            // 
            this.tbTmax.Location = new System.Drawing.Point(345, 45);
            this.tbTmax.Name = "tbTmax";
            this.tbTmax.Size = new System.Drawing.Size(100, 26);
            this.tbTmax.TabIndex = 11;
            this.tbTmax.Text = "0,11";
            // 
            // tbTau
            // 
            this.tbTau.Location = new System.Drawing.Point(345, 81);
            this.tbTau.Name = "tbTau";
            this.tbTau.Size = new System.Drawing.Size(100, 26);
            this.tbTau.TabIndex = 10;
            this.tbTau.Text = "0,01";
            // 
            // tbT0
            // 
            this.tbT0.Location = new System.Drawing.Point(345, 10);
            this.tbT0.Name = "tbT0";
            this.tbT0.Size = new System.Drawing.Size(100, 26);
            this.tbT0.TabIndex = 9;
            this.tbT0.Text = "0";
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(288, 48);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(45, 20);
            this.label9.TabIndex = 8;
            this.label9.Text = "tmax";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(288, 81);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(32, 20);
            this.label8.TabIndex = 7;
            this.label8.Text = "tau";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(288, 16);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(23, 20);
            this.label7.TabIndex = 6;
            this.label7.Text = "t0";
            // 
            // rbRK4
            // 
            this.rbRK4.AutoSize = true;
            this.rbRK4.Font = new System.Drawing.Font("Microsoft Sans Serif", 11F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.rbRK4.Location = new System.Drawing.Point(18, 98);
            this.rbRK4.Name = "rbRK4";
            this.rbRK4.Size = new System.Drawing.Size(155, 28);
            this.rbRK4.TabIndex = 5;
            this.rbRK4.TabStop = true;
            this.rbRK4.Text = "Рунге-Кутта 4";
            this.rbRK4.UseVisualStyleBackColor = true;
            // 
            // rbRK2
            // 
            this.rbRK2.AutoSize = true;
            this.rbRK2.Font = new System.Drawing.Font("Microsoft Sans Serif", 11F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.rbRK2.Location = new System.Drawing.Point(18, 63);
            this.rbRK2.Name = "rbRK2";
            this.rbRK2.Size = new System.Drawing.Size(155, 28);
            this.rbRK2.TabIndex = 4;
            this.rbRK2.TabStop = true;
            this.rbRK2.Text = "Рунге-Кутта 2";
            this.rbRK2.UseVisualStyleBackColor = true;
            // 
            // rbEiler
            // 
            this.rbEiler.AutoSize = true;
            this.rbEiler.Font = new System.Drawing.Font("Microsoft Sans Serif", 11F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.rbEiler.Location = new System.Drawing.Point(18, 28);
            this.rbEiler.Name = "rbEiler";
            this.rbEiler.Size = new System.Drawing.Size(97, 28);
            this.rbEiler.TabIndex = 3;
            this.rbEiler.TabStop = true;
            this.rbEiler.Text = "Эйлера";
            this.rbEiler.UseVisualStyleBackColor = false;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.label1.Location = new System.Drawing.Point(3, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(161, 25);
            this.label1.TabIndex = 2;
            this.label1.Text = "Метод решения";
            // 
            // btnDiffEq
            // 
            this.btnDiffEq.Font = new System.Drawing.Font("Microsoft Sans Serif", 15F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.btnDiffEq.Location = new System.Drawing.Point(160, 314);
            this.btnDiffEq.Name = "btnDiffEq";
            this.btnDiffEq.Size = new System.Drawing.Size(213, 62);
            this.btnDiffEq.TabIndex = 1;
            this.btnDiffEq.Text = "Решить";
            this.btnDiffEq.UseVisualStyleBackColor = true;
            this.btnDiffEq.Click += new System.EventHandler(this.button1_Click);
            // 
            // numericUpDown1
            // 
            this.numericUpDown1.Location = new System.Drawing.Point(107, 162);
            this.numericUpDown1.Maximum = new decimal(new int[] {
            4,
            0,
            0,
            0});
            this.numericUpDown1.Minimum = new decimal(new int[] {
            2,
            0,
            0,
            0});
            this.numericUpDown1.Name = "numericUpDown1";
            this.numericUpDown1.Size = new System.Drawing.Size(120, 26);
            this.numericUpDown1.TabIndex = 19;
            this.numericUpDown1.Value = new decimal(new int[] {
            2,
            0,
            0,
            0});
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.Location = new System.Drawing.Point(18, 164);
            this.label14.Name = "label14";
            this.label14.RightToLeft = System.Windows.Forms.RightToLeft.Yes;
            this.label14.Size = new System.Drawing.Size(83, 20);
            this.label14.TabIndex = 20;
            this.label14.Text = "Порядок";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(564, 446);
            this.Controls.Add(this.tabControl1);
            this.Name = "Form1";
            this.Text = "Form1";
            this.tabControl1.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.tabPage1.PerformLayout();
            this.tabPage2.ResumeLayout(false);
            this.tabPage2.PerformLayout();
            this.tabPage3.ResumeLayout(false);
            this.tabPage3.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown1)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage tabPage1;
        private System.Windows.Forms.TabPage tabPage2;
        private System.Windows.Forms.TabPage tabPage3;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button btnDiffEq;
        private System.Windows.Forms.RadioButton rbRK4;
        private System.Windows.Forms.RadioButton rbRK2;
        private System.Windows.Forms.RadioButton rbEiler;
        private System.Windows.Forms.RadioButton rbGuassa;
        private System.Windows.Forms.RadioButton rbSimpson38;
        private System.Windows.Forms.RadioButton rbSimpson;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Button btnIntegral;
        private System.Windows.Forms.TextBox tbResult;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox tbN;
        private System.Windows.Forms.TextBox tbB;
        private System.Windows.Forms.TextBox tbA;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox tbTmax;
        private System.Windows.Forms.TextBox tbTau;
        private System.Windows.Forms.TextBox tbT0;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.RichTextBox richTextBox1;
        private System.Windows.Forms.Button btnMatrix;
        private System.Windows.Forms.RichTextBox rtbMatrix;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.RichTextBox rtbResult;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.NumericUpDown numericUpDown1;
    }
}

