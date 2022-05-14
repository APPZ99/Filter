'''
Author: APPZ99
Date: 2022-05-11 12:31:28
LastEditTime: 2022-05-14 17:07:54
LastEditors: APPZ99
Description: 卡尔曼滤波器单变量实现
'''

import numpy as np
import matplotlib.pyplot as plt

class KF_Filter:

    def __init__(self, KP, KI, KD, exp_alt, true_alt):

        self.kp = KP
        self.ki = KI
        self.kd = KD

        self.true_alt = true_alt
        self.exp_alt = exp_alt
        self.pre_alt = self.true_alt + np.random.normal(0, 0.1)
        self.pre_error = 0.1 + np.random.normal(0, 0.1)

        self.now_error = 0.0
        self.last_error = 0.0
        self.inter = 0.0
        self.diff = 0.0

        self.acc = 0.0
        self.vel = 0.0
        self.aim_alt =0.0
        self.time_step = 0.1
        self.measure = 0.0
        self.KG = 0.8

    def PID_Controller(self):

        self.last_error = self.now_error
        self.now_error = self.exp_alt - self.pre_alt
        self.inter += self.now_error
        self.diff = self.now_error - self.last_error
        self.aim_alt = self.kp * self.now_error + self.ki * self.inter + self.kd * self.diff
        self.acc = 2 * (self.aim_alt - self.vel * self.time_step) / (self.time_step ** 2)
        if self.acc > 20:
            self.acc = 20
        elif self.acc < -20:
            self.acc = -20
        return self.acc, self.aim_alt

    def KF(self):
        self.pre_alt = self.pre_alt + self.vel *self.time_step + 0.5 * self.acc * (self.time_step ** 2)
        self.true_alt = self.true_alt + self.vel * self.time_step + 0.5 * self.acc * (self.time_step ** 2)
        self.pre_error = self.pre_error + np.random.normal(0, 0.1)
        self.vel = self.vel + self.acc * self.time_step
        if self.vel >= 20:
            self.vel = 20
        if self.vel <= -20:
            self.vel = -20
        self.true_alt = self.true_alt + np.random.normal(0,0.1)
        self.measure = self.true_alt + np.random.normal(0,0.1)
        self.KG = self.pre_error / (self.pre_error + 0.1)
        self.pre_alt = self.pre_alt + self.KG * (self.measure - self.pre_alt)
        self.pre_error = (1 - self.KG) * self.pre_error
        return self.pre_alt, self.true_alt, self.vel, self.KG, self.pre_error


pre = []
true = []
vel = []
kg = []
error = []
altitude = KF_Filter(0.2, 0.00005, 0.05, 100, 0)
for i in range(1,200):
    acc, aim_alt = altitude.PID_Controller()
    pre_alt, true_alt, now_vel, k, pre_error = altitude.KF()
    pre.append(pre_alt)
    true.append(true_alt)
    vel.append(now_vel)
    kg.append(k)
    error.append(pre_error)

time = list(range(1, 200))
plt.figure(figsize=(6,6), dpi=80)
plt.figure(1)
ax1 = plt.subplot(221)
plt.plot(time, pre, color='blue', label='pre')
plt.plot(time, true, color='red',label='true')
ax2 = plt.subplot(222)
plt.plot(time, vel, "g")
ax3 = plt.subplot(223)
plt.plot(time, kg, "k")
ax4 = plt.subplot(224)
plt.plot(time, error, "y")
plt.show()



