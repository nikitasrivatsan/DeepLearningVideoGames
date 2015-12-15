#!/usr/bin/python

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.image as mpimg
import pylab as plt
import numpy as np



DPI = 96;
STEP_SIZE = 1;

class plot_state:
    def __init__(self):        
        self.X = np.zeros(1)
        self.reward = np.zeros(1)
        self.max_q = np.zeros(1)
        self.rholder = np.zeros(STEP_SIZE)
        self.qholder = np.zeros(STEP_SIZE)

        plt.ion()
        
        fig1 = plt.figure("Performance",figsize=(12, 2.7), dpi=DPI)
        self.r = fig1.add_subplot(1,2,1)
        self.q = fig1.add_subplot(1,2,2)
        plt.get_current_fig_manager().window.wm_geometry("+20+40")
        self.graph_q = self.q.plot(self.X,self.max_q)[0]
        self.graph_r = self.r.plot(self.X,self.reward)[0]
        
        plt.draw()

        plt.ion()
        
        fig2 = plt.figure("Network State",figsize=(9, 5), dpi=DPI)
        self.l1 = fig2.add_subplot(1,4,1)
        self.l2 = fig2.add_subplot(1,4,2)
        self.l3 = fig2.add_subplot(1,4,3)
        self.l4 = fig2.add_subplot(1,4,4)
        self.l1.set_axis_off()
        self.l2.set_axis_off()
        self.l3.set_axis_off()
        self.l4.set_axis_off()
        plt.get_current_fig_manager().window.wm_geometry("+680+350")
        plt.tight_layout()
        
        plt.draw()

        plt.figure("Performance")
        plt.draw()
        
        self.counter = 0
        

    def update(self,r,q,c1,c2,c3):
        #Conv1 layer is 20x20x32
        #Conv2 layer is 5x5x64
        #Conv3 layer is 3x3x64
        
        self.rholder[self.counter] = r
        self.qholder[self.counter] = q
        self.counter += STEP_SIZE
        
        if self.counter == STEP_SIZE:
            self.counter = 0

            
            plt.figure("Network State")
            self.l1.imshow(s[:,:,0],aspect = 2.5)
            self.l2.imshow(s[:,:,1],aspect = 2.5)
            self.l3.imshow(s[:,:,2],aspect = 2.5)
            self.l4.imshow(s[:,:,3],aspect = 2.5)
            plt.draw()

            plt.figure("Performance")
            self.X = np.append(self.X,np.amax(self.X) + STEP_SIZE)

            self.max_q = np.append(self.max_q,np.mean(self.qholder))
            self.reward = np.append(self.reward,np.mean(self.rholder))
            
            self.qholder = np.zeros(STEP_SIZE)
            self.rholder = np.zeros(STEP_SIZE)
            
            self.graph_q.set_xdata(self.X)
            self.graph_q.set_ydata(self.max_q)

            self.graph_r.set_xdata(self.X)
            self.graph_r.set_ydata(self.reward)
            
            self.q.set_xlim([0,np.amax(self.X)])
            self.q.set_ylim([np.amin(self.max_q),np.amax(self.max_q)])

            plt.draw()

            self.r.set_xlim([0,np.amax(self.X)])
            self.r.set_ylim([np.amin(self.reward),np.amax(self.reward)])

            plt.draw()
            
            return self.max_q, self.reward
            

