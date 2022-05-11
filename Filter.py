'''
    @ author: APPZ99
    @ description: A PID controller based on python
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

        self.now_error = 0.0
        self.last_error = 0.0
        self.inter = 0.0
        self.diff = 0.0

        self.acc = 0.0
        self.vel = 0.0
        self.aim_alt =0.0
        self.time_step = 0.1
        self.meas = 0.0
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
        self.vel = self.vel + self.acc * self.time_step
        if self.vel >= 20:
            self.vel = 20
        if self.vel <= -20:
            self.vel = -20
        self.true_alt = self.true_alt + np.random.normal(0,0.1)
        self.meas = self.true_alt + np.random.normal(0,0.1)
        self.pre_alt = self.pre_alt + self.KG * (self.meas - self.pre_alt)
        return self.pre_alt, self.true_alt, self.vel


pre = []
true = []
vel = []
altitude = KF_Filter(0.2, 0.00005, 0.05, 100, 0)
for i in range(1,200):
    acc, aim_alt = altitude.PID_Controller()
    pre_alt, true_alt, now_vel = altitude.KF()
    pre.append(pre_alt)
    true.append(true_alt)
    vel.append(now_vel)

time = list(range(1, 200))
plt.plot(time, pre, color='blue', label='pre')
plt.plot(time, true, color='red',label='true')
plt.show()



