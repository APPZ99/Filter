'''
Author: APPZ99
Date: 2022-05-12 21:59:14
LastEditTime: 2022-05-13 16:17:15
LastEditors: APPZ99
Description: 常见数字滤波
'''

import numpy as np
import matplotlib.pyplot as plt

def DrawData():
    x = np.arange(0.0, 5.0, 0.01)
    y = np.cos(2 * np.pi * x) + np.random.normal(0, 0.1, 500)
    #y = np.random.normal(0, 1, 500)
    plt.plot(x, y, "r")
    plt.show()

class SinGenerater_c:

    def __init__(self, A, f, fs, phi, t):

        self.A = A      # 振幅
        self.f = f      # 信号频率
        self.fs = fs    # 采样频率
        self.phi = phi  # 相位
        self.t = t      # 采用时间

        self.Ts = 1 / self.fs    # 采样周期
        self.n = np.arange(self.t / self.Ts)    # 
        
        self.y = self.A * np.sin(2 * np.pi * self.f * self.n * self.Ts + self.phi * (np.pi / 180))


    def Noise(self, exp, var, fre):

        self.noise = np.random.normal(exp, var, fre)

        return self.y, self.y + self.noise








if __name__ == "__main__":

    fs = 2000
    sin= SinGenerater_c(2, 1, fs, 0, 10.0)
    _, y= sin.Noise(0, 0.1, fs * 10)
    print(np.shape(y)[0])
    x = np.arange(0.0, 10, 1/fs)
    plt.plot(x, y, "r")
    plt.show()