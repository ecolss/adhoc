#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import pylab as pl
pl.style.use('ggplot')

class RTPlot:
    def __init__(self, plots=[1,1], pause=0.01):
        pl.ion()
        fig, ax = pl.subplots(*plots)
        if plots[0]*plots[1] == 1:
            ax = np.array([[ax]])
        ax = ax.reshape(-1)
        self.fax = [fig, ax]

        self.pause = pause
        self.plots = plots

        self.xy = [([],[]) for i in range(ax.size)]
        self.data = [self.fax[1][i].plot(*self.xy[i])[0] for i in range(ax.size)]
        self.fax[0].canvas.draw()

    def update(self, xya):
        sz = min(len(xya), self.fax[1].size)
        for i in range(sz):
            if xya[i][-1]:
                self.xy[i][0].append(xya[i][0])
                self.xy[i][1].append(xya[i][1])
            else:
                self.xy[i] = xya[i][:-1]
        for i in range(sz):
            self.data[i].set_data(*self.xy[i])
            self.fax[1][i].relim()
            self.fax[1][i].autoscale_view(True,True,True)
        self.fax[0].canvas.draw()
        pl.pause(self.pause)

    def close(self):
        pl.waitforbuttonpress()

    def __del__(self):
        self.close()

class AdaGrad:
    def __init__(self, vns, init=0.1, max_norm=0, decay_step=100, decay_rate=0.95, min_lr=0.01):
        self.G = {vn:init for vn in vns}
        self.max_norm = max_norm
        self.n_step = 0
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.min_lr = min_lr

    def update(self, vgs, lrate=0.1):
        lrate = max(lrate*pow(self.decay_rate, int(self.n_step/self.decay_step)), self.min_lr)
        for vn,v,g in vgs:
            v = v.data
            g = g.data
            if self.max_norm>0:
                norm = g.norm()
                if norm>self.max_norm:
                    g = g*self.max_norm/norm
            v += lrate*g/np.sqrt(self.G[vn])
            self.G[vn] += g**2
        self.n_step += 1


if __name__ == '__main__':
    rtp = RTPlot([2,1], pause=0.002)

    for i in range(10):
        rtp.update([(i,i), (i,i)], 1)
