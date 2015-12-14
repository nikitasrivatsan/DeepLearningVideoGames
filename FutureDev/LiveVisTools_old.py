#!/usr/bin/python

import pylab as plt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation

import Tkinter as tk
import ttk

sys.path.append("Raw Game Code/")
import pong

LARGE_FONT= ("Verdana", 12)
DPI = 96


class NetworkStateContainer(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Deep Q-Learning Visualization")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        frame = RewardPlot(container, self)
        self.frames = RewardPlot(container, self)
        frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(RewardPlot)
        
    def show_frame(self,cont):
        frame = self.frames
        #frame.tkraise()
        

class RewardPlot(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
       
        self.X = np.zeros(1)
        self.Y = np.zeros(1)
        self.holder = np.zeros(100)
        #plt.figure(figsize=(640/DPI / 0.9,200/DPI),dpi = DPI)
        self.f = plt.figure(1)
        self.graph = plt.plot(self.X,self.Y)[0]
        #mgr = plt.get_current_fig_manager()
        #mgr.full_screen_toggle()  # primitive but works to get screen size
        #py = mgr.canvas.height()
        #px = mgr.canvas.width()
        d = 10  # width of the window border in pixels
        
        self.counter = 0
        plt.draw()
        
        self.canvas = FigureCanvasTkAgg(self.f, self)
        self.canvas.show()
        #canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        #toolbar = NavigationToolbar2TkAgg(canvas, self)
        #toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def plot_reward(self,val):
        self.holder[self.counter] = val
        self.counter += 1
        if self.counter == 1:
            print('hi')
            self.counter = 0
            #self.Y = np.roll(self.Y,-1)
            self.Y = np.append(self.Y,np.mean(self.holder))
            self.X = np.append(self.X,np.amax(self.X) + 100)
            self.holder = np.zeros(100)
            self.graph.set_xdata(self.X)
            self.graph.set_ydata(self.Y)
            plt.ylim([np.amin(self.Y),np.amax(self.Y)])
            plt.xlim([0,np.amax(self.X)])
            plt.draw()
            self.canvas.draw()
            return self.Y



app = NetworkStateContainer()


    
