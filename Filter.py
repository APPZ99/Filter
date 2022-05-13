'''
Author: APPZ99
Date: 2022-05-12 21:59:14
LastEditTime: 2022-05-13 18:08:38
LastEditors: APPZ99
Description: 常见数字滤波
'''

import numpy as np
import matplotlib.pyplot as plt
from queue import Queue # 实现滑动滤波

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

class NormalFilter_c:

    def __init__(self, inputs, per):

        self.inputs = inputs
        self.per = per

    # 算数平均滤波
    def ArithmeticAverage(self):
        # 不满足整组的数据以最后一个数据作为补充
        if ((np.shape(self.inputs)[0] % self.per) != 0):
            #groups = np.shape(self.inputs)[0] / self.per
            for _ in range(self.per):
                self.inputs = np.append(self.inputs, self.inputs[np.shape(self.inputs)[0] - 1])

        self.inputs = self.inputs.reshape((-1, self.per))
        mean = []
        for temp in self.inputs:
            mean.append(temp.mean())

        return mean

    # 滑动平均滤波
    def SlidingAverage(self):
        # 不满足整组的数据以最后一个数据作为补充
        if ((np.shape(self.inputs)[0] % self.per) != 0):
            #groups = np.shape(self.inputs)[0] / self.per
            for _ in range(self.per):
                self.inputs = np.append(self.inputs, self.inputs[np.shape(self.inputs)[0] - 1])

        slider_mean = []
        for i, _ in enumerate(self.inputs):
            slider_mean.append(self.inputs[i : i + self.per].mean())

        return slider_mean

    # 中位值滤波
    def Median(self):
        # 不满足整组的数据以最后一个数据作为补充
        if ((np.shape(self.inputs)[0] % self.per) != 0):
            #groups = np.shape(self.inputs)[0] / self.per
            for _ in range(self.per):
                self.inputs = np.append(self.inputs, self.inputs[np.shape(self.inputs)[0] - 1])

        self.inputs = self.inputs.reshape((-1, self.per))
        median = []
        for temp in self.inputs:
            median.append(np.median(temp))

        return median

    # 带一阶滞后的滑动平均滤波
    def SlidingAverageWithOneOrderLay(self, percent):
            # 不满足整组的数据以最后一个数据作为补充
            if ((np.shape(self.inputs)[0] % self.per) != 0):
                #groups = np.shape(self.inputs)[0] / self.per
                for _ in range(self.per):
                    self.inputs = np.append(self.inputs, self.inputs[np.shape(self.inputs)[0] - 1])

            slider_mean = []
            for i, _ in enumerate(self.inputs):
                now_mean = self.inputs[i : i + self.per].mean()
                if i != 0:
                    true_mean = (1 - percent) * now_mean + percent * slider_mean[i - 1]
                    slider_mean.append(true_mean)
                else:
                    slider_mean.append(now_mean)

            return slider_mean



if __name__ == "__main__":

    fs = 2000
    sin = SinGenerater_c(2, 1, fs, 0, 10.0)
    _, y = sin.Noise(0, 0.1, fs * 10)
    #print(np.shape(y)[0])
    x = np.arange(0.0, 10, 1/fs)

    per = 20
    filter = NormalFilter_c(y, per)
    z1 = filter.SlidingAverage()
    x1 = np.arange(0.0, 10, 1 / fs)

    z2 = filter.ArithmeticAverage()
    x2 = np.arange(0.0, 10, 1 / fs * per)

    z3 = filter.Median()
    x3 = np.arange(0.0, 10, 1 / fs * per)

    z4 = filter.SlidingAverageWithOneOrderLay(0.2)
    x4 = np.arange(0.0, 10, 1 / fs * per)

    plt.figure(figsize=(6,6), dpi=80)
    plt.figure(1)
    ax1 = plt.subplot(221)
    plt.plot(x, y, "r")
    plt.plot(x1, z1, "g")
    ax2 = plt.subplot(222)
    plt.plot(x, y, "r")
    plt.plot(x2, z2, "b")
    ax3 = plt.subplot(223)
    plt.plot(x, y, "r")
    plt.plot(x3, z3, "y")
    ax4 = plt.subplot(224)
    plt.plot(x, y, "r")
    plt.plot(x4, z4, "k")
    plt.show()