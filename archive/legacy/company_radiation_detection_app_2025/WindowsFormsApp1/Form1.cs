using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace WindowsFormsApp1
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            serialPort1.PortName = comName;
            serialPort1.BaudRate = 9600;
            serialPort1.StopBits = System.IO.Ports.StopBits.One;
            serialPort1.DataBits = 8;
            serialPort1.Parity = System.IO.Ports.Parity.None;
            while (!serialPort1.IsOpen)
            {
                try
                {
                    serialPort1.Open();
                }
                catch
                {
                    continue;
                }
                Thread.Sleep(500);
            }
            button1.BackColor = Color.Lime;
        }

        private string comName = "COM3";

        private void button1_Click(object sender, EventArgs e)
        {
            timer1.Start();
        }
        private byte[] vs = { 0x01, 0x03, 0x00, 0x00, 0x00, 0x07, 0x04, 0x08 };

        private void timer1_Tick(object sender, EventArgs e)
        {
            serialPort1.Write(vs, 0, 8);
        }

        private byte[] bytes;
        private void serialPort1_DataReceived(object sender, System.IO.Ports.SerialDataReceivedEventArgs e)
        {
            try
            {
                this.Invoke((EventHandler)(delegate
                {
                    bytes = Encoding.Default.GetBytes(serialPort1.ReadExisting());
                    double num = (int)((bytes[6] & 0xFF)) + (int)((bytes[3] & 0xFF) << 24) + (int)((bytes[4] & 0xFF) << 16) + (int)((bytes[5] & 0xFF) << 8);
                    num = (num * 36) / 1000;
                    string str = num.ToString("0.000");
                    label2.Text=str+" "+"µSv/h";
                }));
            }
            catch (Exception ee)
            {
                
            }
        }
    }
}

