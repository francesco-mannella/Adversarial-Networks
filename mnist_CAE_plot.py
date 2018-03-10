# -*- coding: utf-8 -*-
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#---------------------------------------------------------------------------  
#---------------------------------------------------------------------------  
#---------------------------------------------------------------------------  
# Plot
class Plotter(object):  
    def __init__(self, epochs):   
        self.t = 0

        self.fig = plt.figure(figsize=(12,8))
        self.h = 10
        self.w = 4
        self.s = 20
        img_side = 28
        
        h , w, s = (self.h, self.w, self.s)
        gs = gridspec.GridSpec(s + h, s*2 + w)
        
        self.losses_ax = self.fig.add_subplot(gs[1:h-4, 2:s*2+w-2])
        self.losses_ax.set_title("Reconstruction error epochs:{}".format(self.t))
        self.losses_lines = []
        line, = self.losses_ax.plot(0,0)   
        self.losses_lines.append(line)
        self.labels = ["reconstruction"]  
        self.losses_ax.legend(self.losses_lines, self.labels)  
        self.losses_ax.grid(color='b', linestyle='--', linewidth=0.5)
        self.losses_ax.set_yticks(np.linspace(0.1,0.3, 11))
        self.losses_ax.set_xlim([0, epochs]) 
        self.losses_ax.set_ylim([0.1, 0.24])
                   
        self.data_pattern_axes = []
        self.data_pattern_imgs = []
        for x in range(s):
            for y in range(s):
                ax = self.fig.add_subplot(gs[h + s - x - 2,  w/2 + y])
                im = ax.imshow(np.zeros([img_side, img_side]), 
                               vmin=-1, vmax=1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                self.data_pattern_axes.append(ax)
                self.data_pattern_imgs.append(im)
        
        plt.subplots_adjust(top=1.0, bottom=0.0, 
                            left=0.0, right=1.0, 
                            hspace=0.0, wspace=0.0)
        if not os.path.exists("imgs"):
            os.makedirs("imgs")  

        self.pattern_axes = []
        self.pattern_imgs = []
        for x in range(s):
            for y in range(s):
                ax = self.fig.add_subplot(gs[h + s - x - 2, s + w/2 + y])
                im = ax.imshow(np.zeros([img_side, img_side]), 
                               vmin=-1, vmax=1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                self.pattern_axes.append(ax)
                self.pattern_imgs.append(im)
        
        plt.subplots_adjust(top=1.0, bottom=0.0, 
                            left=0.0, right=1.0, 
                            hspace=0.0, wspace=0.0)
        if not os.path.exists("imgs"):
            os.makedirs("imgs")  
            
    def plot(self, R_loss, data_patterns, patterns):
        
        img_side = 28
        losses = [R_loss]   
        t = len(R_loss)
        self.losses_lines[0].set_data(np.arange(t), R_loss)
        self.losses_ax.set_title("Reconstruction error epochs:{}".format(self.t))
            
        for x in range(self.s):
            for y in range(self.s):
                k = x*self.s + y     
                im = self.pattern_imgs[k]
                im.set_data(patterns[k].reshape(img_side, img_side))
                im = self.data_pattern_imgs[k]
                im.set_data(data_patterns[k].reshape(img_side, img_side))
        self.fig.canvas.draw()
        self.fig.savefig("imgs/cae.png".format(self.t))
        self.fig.savefig("imgs/cae-{:03d}.png".format(self.t))
        self.t += 1
        
