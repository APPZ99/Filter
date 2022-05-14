'''
Author: APPZ99
Date: 2022-05-14 17:06:25
LastEditTime: 2022-05-14 20:41:41
LastEditors: APPZ99
Description: 多变量卡尔曼滤波实现
'''

import numpy as np
import matplotlib.pyplot as plt
import math

class KF_Filter_c:

    def __init__(self):

        # 初始化参数
        self.time_step = 1.0  # 时间步
        self.acc = 0.1  # 加速度
        self.K = 0.1    # 卡尔曼增益
        self.F = np.mat([[1.0, self.time_step],
                        [0.0, 1.0]])  # 状态转移矩阵
        self.Q = np.mat([[0.01, 0],
                        [0, 0.01]])      # 状态转移协方差矩阵
        self.U = np.mat([[(self.time_step ** 2) / 2],
                        [self.time_step]])      # 控制矩阵

        self.H = np.mat([[-1.0, 0],
                        [0, -1.0]])     # 观测矩阵
        self.measure_noise = np.mat(np.random.randn(2, 100))  # 观测噪声
        self.R = np.mat([[1.0, 0],
                        [0, 1.0]])      # 观测噪声协方差

    def draw_real_track(self):

        '''
        description: 绘制真实给定位移及速度曲线
        param {*}
        return {*}
        '''        
        real_status = np.mat(np.zeros((2, 100)))
        real_status[:, 0] = np.mat([[0.0],
                                    [1.0]])
        for i in range(99):
            real_status[:, i + 1] = self.F * real_status[:, i] + self.U * self.acc
        real_status = np.array(real_status)

        plt.figure(figsize=(6,6), dpi=80)
        plt.figure(1)
        ax1 = plt.subplot(211)
        plt.grid()
        plt.plot(real_status[0, :])
        ax2 = plt.subplot(212)
        plt.grid()
        plt.plot(real_status[1, :])
        plt.show()
        return real_status

    def draw_measure_track(self):

        '''
        description: 绘制测量位移及速度曲线
        param {*}
        return {*}
        '''        
        real_status = np.mat(self.draw_real_track())
        measure_track = np.mat(np.zeros((2, 100)))
        for i in range(100):
            measure_track[:, i] = self.H * real_status[:, i] + self.measure_noise[:, i]
        measure_track = np.array(measure_track)

        plt.figure(figsize=(6,6), dpi=80)
        plt.figure(1)
        ax1 = plt.subplot(211)
        plt.grid()
        plt.plot(measure_track[0, :])
        ax2 = plt.subplot(212)
        plt.grid()
        plt.plot(measure_track[1, :])
        plt.show()

    def KF(self):

        '''
        description: 卡尔曼滤波实现
        param {*}
        return {*}
        '''        
        est_status = np.mat(np.zeros((2, 100))) # 估计值
        upd_status = np.mat(np.zeros((2, 100))) # 真实值
        est_predict = np.zeros((100, 2, 2))     # 估计误差
        upd_predict = np.zeros((100, 2, 2))     # 真实误差
        
        est_predict[0, : , :] = np.mat([[1.0, 0.0], 
                                        [0.0, 1.0]])
        upd_predict[0, : , :] = np.mat([[1.0, 0.0], 
                                        [0.0, 1.0]])

        real_status = np.mat(np.zeros((2, 100)))
        real_status[:, 0] = np.mat([[0.0],
                                    [1.0]])

        for i in range(99):
            real_status[:, i + 1] = self.F * real_status[:, i] + self.U * self.acc

        measure_track = np.mat(np.zeros((2, 100)))
        for i in range(100):
            measure_track[:, i] = self.H * real_status[:, i] + self.measure_noise[:, i]

        for i in range(99):
            # 预测部分 两条公式
            est_status[:, i + 1] = self.F * upd_status[:, i] + self.U * self.acc
            ep = self.F * np.mat(upd_predict[i, : , :]) * self.F.T + self.Q
            est_predict[i + 1, : , :] = ep
            # 更新部分 三条公式
            self.K = ep * self.H.T * np.linalg.inv(self.H * ep * self.H.T + self.R)
            up = ep - self.K * self.H * ep
            upd_predict[i + 1, : , :] = up
            upd_status[:, i + 1] = est_status[:, i + 1] + self.K * (measure_track[:, i + 1] - self.H * est_status[:, i + 1])

        upd_status = np.array(upd_status)
        est_status = np.array(est_status)
        measure_track = np.array(measure_track)
        real_status = np.array(real_status)

        plt.figure(figsize=(6,6), dpi=80)
        plt.figure(1)
        ax1 = plt.subplot(321)
        plt.grid()
        plt.plot(real_status[0, :])
        ax2 = plt.subplot(322)
        plt.grid()
        plt.plot(real_status[1, :])

        ax3 = plt.subplot(323)
        plt.grid()
        plt.plot(measure_track[0, :])
        ax4 = plt.subplot(324)
        plt.grid()
        plt.plot(measure_track[1, :])

        ax5 = plt.subplot(325)
        plt.grid()
        plt.plot(upd_status[0, :], "r")
        plt.plot(est_status[0, :], "b")
        ax6 = plt.subplot(326)
        plt.grid()
        plt.plot(upd_status[1, :], "r")
        plt.plot(est_status[1, :], "b")
        plt.show()


if __name__ == "__main__":

    kf_filter = KF_Filter_c()
    #kf_filter.draw_real_track()
    #kf_filter.draw_measure_track()
    kf_filter.KF()




