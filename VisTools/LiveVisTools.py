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
        
        self.fig1 = plt.figure("Performance",figsize=(6.5, 2.7), dpi=DPI)
        self.q = self.fig1.add_subplot(1,1,1)
        plt.get_current_fig_manager().window.wm_geometry("+15+40")
        self.graph_q = self.q.plot(self.X,self.max_q)[0]
        plt.xlabel('Frames')
        plt.ylabel('Max Q Value')
        plt.tight_layout()
        
        plt.draw()

        plt.ion()
        
        self.fig2 = plt.figure("Network State",figsize=(9.2, 8.16), dpi=DPI)
        self.l1 = self.fig2.add_subplot(3,1,1)
        self.l2 = self.fig2.add_subplot(3,1,2)
        self.l3 = self.fig2.add_subplot(3,1,3)
        self.l1.set_title("First Convolutional Layer")
        self.l2.set_title("Second Convolutional Layer")
        self.l3.set_title("Third Convolutional Layer")
        self.l1.set_axis_off()
        self.l2.set_axis_off()
        self.l3.set_axis_off()
        self.l1.imshow(np.reshape(np.zeros(20*20*32),(20,20*32)),aspect = 6, animated=True)
        self.l2.imshow(np.reshape(np.zeros(5*5*64),(5,5*64)),aspect = 12)
        self.l3.imshow(np.reshape(np.zeros(3*3*64),(3,3*64)),aspect = 12)
        plt.get_current_fig_manager().window.wm_geometry("+680+40")
        plt.tight_layout()
        
        plt.draw()

        plt.figure("Performance")
        plt.draw()
        
        self.counter = 0
        self.startflag = True;

    def update(self,r,q,c1,c2,c3):
        #Conv1 layer is 20x20x32
        #Conv2 layer is 5x5x64
        #Conv3 layer is 3x3x64
        
        self.qholder[self.counter] = q
        self.counter += STEP_SIZE

        if self.counter == STEP_SIZE:
            self.counter = 0

            if self.startflag:
                plt.figure("Network State")
                self.startflag = False
                self.graphl1 = self.l1.imshow(np.reshape(np.rollaxis(c1, 2, 1),(20,20*32)),aspect = 6)
                self.graphl2 = self.l2.imshow(np.reshape(np.rollaxis(c2, 2, 1),(5,5*64)),aspect = 12)
                self.graphl3 = self.l3.imshow(np.reshape(np.rollaxis(c3, 2, 1),(3,3*64)),aspect = 12)

            
            plt.figure("Network State")

            self.graphl1.set_data(np.reshape(np.rollaxis(c1, 2, 1),(20,20*32)))
            self.graphl2.set_data(np.reshape(np.rollaxis(c2, 2, 1),(5,5*64)))
            self.graphl3.set_data(np.reshape(np.rollaxis(c3, 2, 1),(3,3*64)))
            
            plt.figure("Performance")
            self.X = np.append(self.X,np.amax(self.X) + STEP_SIZE)

            self.max_q = np.append(self.max_q,np.mean(self.qholder))
            
            self.qholder = np.zeros(STEP_SIZE)
            
            self.graph_q.set_xdata(self.X)
            self.graph_q.set_ydata(self.max_q)
            
            self.q.set_xlim([0,np.amax(self.X)])
            self.q.set_ylim([np.amin(self.max_q),np.amax(self.max_q)])

            self.fig1.canvas.draw()
            self.fig2.canvas.draw()
            #self.fig1.canvas.flush_events()
            
            return self.max_q
        
            

