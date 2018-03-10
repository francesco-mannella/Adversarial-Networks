# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#---------------------------------------------------------------------------  
#---------------------------------------------------------------------------  
#---------------------------------------------------------------------------  
# Plot

# Plot
class Plotter(object):  
    def __init__(self, epochs):
        
        self.t = 0
        
        self.fig = plt.figure(figsize=(14,7))
        self.h = 2
        self.w = 3
        self.s = 5
        self.hg = 2
        self.wg = 3
        img_side = 28
        
        h , w, s, hg, wg = (self.h, self.w, self.s, self.hg, self.wg)
        gs = gridspec.GridSpec(s + h, s + wg + w)
        

        self.changes_ax = self.fig.add_subplot(gs[1:(1+hg), 1:(2+wg)])
        self.changes_ax.set_title("Weight change")
        self.cd_line, = self.changes_ax.plot(0,0)
        self.cg_line, = self.changes_ax.plot(0,0)
        self.changes_ax.legend([self.cd_line, self.cg_line], 
                              ["discriminator", "generator"])  
        self.changes_ax.set_xlim([0, epochs]) 
        self.changes_ax.set_ylim([0, 3.5])
        self.changes_ax.grid(color='b', linestyle='--', linewidth=0.5)
        self.changes_ax.set_yticks(np.linspace(0,3, 7))
          
        self.losses_ax = self.fig.add_subplot(gs[(2+hg):(2+2*hg), 1:(2+wg)])
        self.losses_ax.set_title("Losses")
        self.ld_line, = self.losses_ax.plot(0,0)
        self.lg_line, = self.losses_ax.plot(0,0)
        self.losses_ax.legend([self.ld_line, self.lg_line], 
                ["discriminator: log(D(x)) + log(1 - D(G(z)))", 
                    "generator: log(D(G(z)))"])  
        self.losses_ax.set_xlim([0, epochs]) 
        self.losses_ax.set_ylim([0, -2.8])
        self.losses_ax.grid(color='b', linestyle='--', linewidth=0.5)
        self.losses_ax.set_yticks(np.linspace(0,-6, 16))
          
        self.pattern_axes = []
        self.pattern_imgs = []
        for x in range(s):
            for y in range(s):
                ax = self.fig.add_subplot(gs[1+x, (2+wg)+y])
                im = ax.imshow(np.zeros([img_side, img_side]), 
                               vmin=-1, vmax=1, aspect="auto")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                self.pattern_axes.append(ax)
                self.pattern_imgs.append(im)
        
        plt.subplots_adjust(top=1.0, bottom=0.0, 
                            left=0.0, right=1.0, 
                            hspace=0.0, wspace=0.0)
        plt.tight_layout()

        if not os.path.exists("imgs"):
            os.makedirs("imgs")  
            

    def plot(self, G_losses, D_losses, G_changes, D_changes, patterns):
                 
        img_side = 28
        t = range(len(G_losses))

        self.ld_line.set_data(t, D_losses)
        self.lg_line.set_data(t, G_losses)
        
        self.cd_line.set_data(t, D_changes)
        self.cg_line.set_data(t, G_changes)
                                    
        l = len(patterns)
        for x in range(5):
            for y in range(5):
                k = x*5 + y     
                im = self.pattern_imgs[k]
                if k<l:
                    im.set_data(patterns[k].reshape(img_side, img_side))
        plt.tight_layout(pad=0.1)
        self.fig.canvas.draw()
        self.fig.savefig("imgs/gan-last.png")
        self.fig.savefig("imgs/gan-{:03d}.png".format(self.t))
        self.t += 1
