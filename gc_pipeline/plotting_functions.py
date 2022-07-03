#!/usr/bin/env python
# coding: utf-8

# # Common

# In[ ]:

import math
import numpy as np
import matplotlib.pyplot as plt


r_left = 57./255.;
g_left = 87./255.;
b_left = 225./225.;
rgb_left = (r_left, g_left, b_left);

r_right = 255./255.;
g_right = 138./255.;
b_right = 0./255.;
rgb_right = (r_right, g_right, b_right);


def pval_to_star(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


# # Motoneurons

# In[ ]:


def get_label_and_color_lists(mid, tot):    
    
    label_list = [2*i + 1 for i in range(mid)]
    label_list.extend([2*i + 2 for i in range(tot-mid)])
    
    color_list = [rgb_left for i in range(mid)]
    color_list.extend([rgb_right for i in range(int(tot-mid))])
        
    return label_list, color_list




def plot_gc_matrix(gc, mid):
    n_cells = len(gc)    
        
    gc[gc==0] = np.nan
    n_sig = len([i  for j in gc for i in j if i>0])
    plt.imshow(gc, cmap='YlOrRd')
    # plt.colorbar()
    
    ax = plt.gca()
    ax.set_xticks(np.arange(n_cells))
    ax.set_yticks(np.arange(n_cells))
    
    label_list, color_list = get_label_and_color_lists(mid, n_cells)
    
    ax.set_xticklabels(label_list)#, size=15) 
    ax.set_yticklabels(label_list)#, size=15)
    
    for color, tick in zip(color_list, ax.xaxis.get_major_ticks()):
        tick.label1.set_color(color) 
    for color, tick in zip(color_list, ax.yaxis.get_major_ticks()):
        tick.label1.set_color(color) 


    plt.xlabel("to neuron", size=20)
    plt.ylabel("from neuron", size=20)
    
    plt.tight_layout()


def get_coords(mid, tot):
    """ Calcualte the coordinates of points on a circle. """
    left = get_coords_left(mid)
    right = get_coords_right(tot - mid)
    
    return np.concatenate([left.T, right.T])


def get_coords_left(n_points):
    """ Calculate the coordinates of points on the left semicircle. """
    x = np.linspace(-1/int(n_points/2), -1, int(n_points/2))
    
    if n_points%2 == 1:
        x = np.concatenate((x,[-1.1]))
        
    x = np.concatenate((x, np.linspace(-1, -1/int(n_points/2), int(n_points/2))))    
    y = np.linspace(1, -1, n_points)
            
    return np.array([x, y])



def get_coords_right(n_points):
    """ Calculate the coordinates of points on the right semicircle. """
    x = np.linspace(1/int(n_points/2), 1, int(n_points/2))
    
    if n_points%2 == 1:
        x = np.concatenate((x,[1.1]))
        
    x = np.concatenate((x, np.linspace(1, 1/int(n_points/2), int(n_points/2))))
    y = np.linspace(1, -1, n_points)
            
    return np.array([x, y])


def plot_directed_graph(gc, mid, hide_digits=False): # for motoneurons
    """ Draw the network: points positions on the circle + GC arrows.
    Circle size proportional to the ipsilateral delta (GC_in - GC_out) 
        
        --> use whole delta instead of ipsi? /!\ if so need to adjust circle size parameter s
        now: s=np.abs(circle_size[i]*1000+(100*is_multiv))
    """
    textsize = 35
    cmap = plt.cm.Greys
    n_cells = len(gc)
    
    gc_ipsi_left = gc[:mid,:mid]
    gc_ipsi_right = gc[mid:, mid:]

    gc_contra_from_left = gc[:mid, mid:]
    gc_contra_from_right = gc[mid:, :mid]

    d_in = np.nansum(gc, axis=0)
    d_out = np.nansum(gc, axis=1)
    
    d_in_ipsi_left = np.nansum(gc_ipsi_left, axis=0)
    d_out_ipsi_left = np.nansum(gc_ipsi_left, axis=1)

    d_in_ipsi_right = np.nansum(gc_ipsi_right, axis=0)
    d_out_ipsi_right = np.nansum(gc_ipsi_right, axis=1)

    d_in_contra_right = np.nansum(gc_contra_from_left, axis=0) # left to right, in right
    d_out_contra_left = np.nansum(gc_contra_from_left, axis=1) # left to right, out left

    d_in_contra_left = np.nansum(gc_contra_from_right, axis=0) # right to left, in left
    d_out_contra_right = np.nansum(gc_contra_from_right, axis=1) # right to left, out right


    # Node centrality (or importance)
    delta = d_in - d_out
    delta_ipsi_left = d_out_ipsi_left - d_in_ipsi_left
    delta_ipsi_right = d_out_ipsi_right - d_in_ipsi_right
    delta_contra_left = d_out_contra_left - d_in_contra_left
    delta_contra_right = d_out_contra_right - d_in_contra_right

    circle_centers = get_coords(mid, n_cells)
    circle_size = np.concatenate([delta_ipsi_left, delta_ipsi_right])
    
    max_circle_size = np.max(np.abs(circle_size))
    circle_size = circle_size / max_circle_size
     
        
    for i, center in enumerate(circle_centers):
        if circle_size[i] < 0: 
            color = 'navy'
        elif circle_size[i] == 0:
            color = 'purple'
        else:
            color = 'firebrick'
        plt.scatter(center[0], center[1], s=100+np.abs(circle_size[i])*400, c=color)

    plt.axis("equal")

    gc_flat = gc.flatten()
    gc_flat[np.isnan(gc_flat)] = 0
    cells_sorted = np.argsort(gc_flat)
    
    gc_max = gc_flat[cells_sorted[-1]]
    gc_min = gc_flat[cells_sorted[0]]
    
    for cell in cells_sorted:
        gc_cell = gc_flat[cell]
        if gc_cell > 0:
            prop = (gc_cell-gc_min) / (gc_max-gc_min)
            color = cmap(prop) 
            width = prop * 0.05 
            cell_from = int(cell/len(circle_centers))
            cell_to = cell % len(circle_centers) 
            plt.arrow(circle_centers[cell_from,0], circle_centers[cell_from,1], 
                      circle_centers[cell_to,0]-circle_centers[cell_from,0], 
                      circle_centers[cell_to,1]-circle_centers[cell_from,1], 
                      color=color, width=width, length_includes_head=True) 
            
            
    label_list, color_list = get_label_and_color_lists(mid, n_cells)
    
    if not hide_digits:
        for label, color, pos in zip(label_list, color_list, circle_centers):
            plt.text(pos[0] + np.sign(pos[0])*0.4, pos[1], label, va='center', ha='center', color=color, size=textsize, fontweight='bold')

    plt.axis('off')
    plt.tight_layout()

        


# # Hindbrain

# In[1]:


# PLOT CELLS ON BRAIN BACKGROUND
def plot_background_and_cells(cell_centers, subset_neurons, background, subplots=False, invert=False):
    """ Plot the brain image of the plane and corresponding cells (of a fish-trial pair).
    To make two subplots (for info flow plots) use subplots=True.
    To only draw background without cells, use scatter=False (for emit/receive plots).
    For fish 2, 5, 6, 8: use invert=True
    cmap=plt.cm.gist_yarg: to have a background more white than black.
    vmax=np.max(background)/2: to increase contrast in background.
    """

    if subplots:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
        ax1.imshow(background, cmap=plt.cm.gist_yarg)
        ax1.set_title('Rostral to caudal', fontsize=25)
        ax2.imshow(background, cmap=plt.cm.gist_yarg)
        ax2.set_title('Caudal to rostral', fontsize=25)
        if invert:
            ax1.scatter(512 - cell_centers[:,0], cell_centers[:,1], color='silver', edgecolor='black', s=100)
            ax2.scatter(512 - cell_centers[:,0], cell_centers[:,1], color='silver', edgecolor='black', s=100)
            ax1.scatter(512 - cell_centers[subset_neurons,0], cell_centers[subset_neurons,1], color='crimson', edgecolor='black', s=100)
            ax2.scatter(512 - cell_centers[subset_neurons,0], cell_centers[subset_neurons,1], color='crimson', edgecolor='black', s=100)
        else:
            ax1.scatter(cell_centers[:,0], cell_centers[:,1], color='silver', edgecolor='black', s=100)
            ax2.scatter(cell_centers[:,0], cell_centers[:,1], color='silver', edgecolor='black', s=100)
            ax1.scatter(cell_centers[subset_neurons,0], cell_centers[subset_neurons,1], color='crimson', edgecolor='black', s=100)
            ax2.scatter(cell_centers[subset_neurons,0], cell_centers[subset_neurons,1], color='crimson', edgecolor='black', s=100)
        
        res = fig, (ax1, ax2)
    else:
        fig = plt.figure(figsize=(10,6))
        plt.imshow(background, cmap=plt.cm.gist_yarg, aspect='equal', vmax=np.max(background)/2)
        if invert:
            plt.scatter(512 - cell_centers[:,0], cell_centers[:,1], color='silver', edgecolor='black', s=100)
            plt.scatter(512 - cell_centers[subset_neurons,0], cell_centers[subset_neurons,1], color='crimson', edgecolor='black', s=100)
        else:
            plt.scatter(cell_centers[:,0], cell_centers[:,1], color='silver', edgecolor='black', s=100)
            plt.scatter(cell_centers[subset_neurons,0], cell_centers[subset_neurons,1], color='crimson', edgecolor='black', s=100)
        res = fig
    plt.axis('off')
    plt.tight_layout()
    
    return res

def plot_gc_matrix_HB(gc_motorneurons, loc, gc_type='dff_8', savepath=None, figname_suffix=None, title=False, save=False):
    n_lags = gc_motorneurons.loc[loc].n_lags
    n_cells = int(gc_motorneurons.loc[loc].n_cells)
    is_multiv = gc_motorneurons.loc[loc].multivariate
    
    
    if gc_type == 'raw':
        gc = gc_motorneurons.loc[loc].GC_sig_raw
    elif gc_type == 'dt':
        gc = gc_motorneurons.loc[loc].GC_sig_dt
    elif gc_type == 'f_smooth':
        gc = gc_motorneurons.loc[loc].GC_sig_f_smooth
    elif gc_type == 'dfdt_smooth':
        gc = gc_motorneurons.loc[loc].GC_sig_dfdt_smooth
    elif gc_type == 'disc_f':
        gc = gc_motorneurons.loc[loc].GC_sig_disc_f
    else:
        if gc_type != 'dff_8':
            print('Unknown \'gc_type\' param: GC on DF/F is returned.')
        gc = gc_motorneurons.loc[loc].GC_sig
        
        
    plt.figure(figsize=(5, 5))
    gc[gc==0] = np.nan
    n_sig = len([i  for j in gc for i in j if i>0])
    plt.imshow(gc, cmap='YlOrRd')
    
    if title: 
        fish = int(gc_motorneurons.loc[loc].Fish)
        trace = int(gc_motorneurons.loc[loc].Trace)
        if is_multiv:
            plt.suptitle(f"Fish {fish}, trace {trace} - Multivariate GC n_lags={n_lags}, {n_sig} significant links", size=20)
        else:
            plt.suptitle(f"Fish {fish}, trace {trace} - Bivariate GC n_lags={n_lags}, {n_sig} significant links", size=20)
    
    plt.xlabel("to neuron", size=25)
    plt.ylabel("from neuron", size=25)
    
    plt.tight_layout()
    
    if save:
        figname = f'figures/{savepath}/gc_mat_f{fish}t{trace}{figname_suffix}.svg'
        plt.savefig(figname)
        
    plt.show()

def plot_drive(cell_centers, subset_neurons, drive, background, default_color='silver', cst=1, invert=False):
    """ Plot the brain image of the plane and corresponding cells (of a fish-trial pair).
    Cell size proportional to its drive.
    """

    fig = plt.figure(figsize=(10,6))
    plt.imshow(background, cmap=plt.cm.gist_yarg, aspect='equal', vmax=np.max(background)/2)
    next_sub_idx = 0
        
    if invert:
        for j in range(len(cell_centers)):
            if next_sub_idx < len(subset_neurons) and j == subset_neurons[next_sub_idx]:
                color = 'crimson'
                next_sub_idx = next_sub_idx+1
            else:
                color = default_color
            plt.scatter(512 - cell_centers[j,0], cell_centers[j,1], color=color, edgecolor='black', s=100*(int(drive[j]*cst)+1), alpha=0.8)
    else:
        for j in range(len(cell_centers)):
            if next_sub_idx < len(subset_neurons) and j == subset_neurons[next_sub_idx]:
                color = 'crimson'
                next_sub_idx = next_sub_idx+1
            else:
                color = default_color
            plt.scatter(cell_centers[j,0], cell_centers[j,1], color=color, edgecolor='black', s=100*(int(drive[j]*cst)+1), alpha=0.8)
    plt.axis('off')
    
    return fig



def plot_background_and_cells_drive(cell_centers, background, drive, cst, color='crimson', subplots=False, invert=False):
    """ Plot the brain image of the plane and corresponding cells (of a fish-trial pair).
    To make two subplots (for info flow plots) use subplots=True.
    To only draw background without cells, use scatter=False (for emit/receive plots).
    For fish 2, 5, 6, 8: use invert=True
    cmap=plt.cm.gist_yarg: to have a background more white than black.
    vmax=np.max(background)/2: to increase contrast in background.
    Size proportional to drive.
    """
    ori_color = color
    
    if subplots:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
        ax1.imshow(background, cmap=plt.cm.gist_yarg)
        ax1.set_title('Rostral to caudal', fontsize=18)
        ax2.imshow(background, cmap=plt.cm.gist_yarg)
        ax2.set_title('Caudal to rostral', fontsize=18)
        if invert:
            for j in range(len(cell_centers)):
                s = 10*(int(drive[j]*cst) + np.sign(drive[j]))
                if s < 0:
                    s = np.abs(s)
                    color = 'navy'
                elif s == 0:
                    s = 100
                    color='purple'
                ax1.scatter(512 - cell_centers[j,0], cell_centers[j,1], color=color, edgecolor='black', s=s)
                ax2.scatter(512 - cell_centers[j,0], cell_centers[j,1], color=color, edgecolor='black', s=s)
                color = ori_color
        else:
            for j in range(len(cell_centers)):
                s = 100*(int(drive[j]*cst) + np.sign(drive[j]))
                if s < 0:
                    s = np.abs(s)
                    color = 'navy'
                elif s == 0:
                    s = 100
                    color='purple'
                ax1.scatter(cell_centers[j,0], cell_centers[j,1], color=color, edgecolor='black', s=s)
                ax2.scatter(cell_centers[j,0], cell_centers[j,1], color=color, edgecolor='black', s=s)
                color = ori_color
        
        res = fig, (ax1, ax2)
    else:
        fig = plt.figure(figsize=(10,6))
        plt.imshow(background, cmap=plt.cm.gist_yarg, aspect='equal', vmax=np.max(background)/2)
        if invert:
            for j in range(len(cell_centers)):
                plt.scatter(512 - cell_centers[j,0], cell_centers[j,1], color=color, edgecolor='black', s=10*(int(drive[j]*cst)+1))
        else:
            for j in range(len(cell_centers)):
                plt.scatter(cell_centers[j,0], cell_centers[j,1], color=color, edgecolor='black', s=10*(int(drive[j]*cst)+1))
        res = fig
    plt.axis('off')
    
    return res



# In[ ]:


def plot_info_flow_HB(gc, cell_centers, background, fish, drive_type='sum_out', zoom=True, xlims=[210,360], ylims=[85,165], cst=100):
    """ Arrow size proportional to GC value (strength) """

    if fish in [2, 5, 6, 8]:
        invert = True
    else:
        invert = False
        
    if drive_type == 'sum_out':
        drive = np.nansum(gc, axis=1)
    elif drive_type == 'difference':
        drive = np.nansum(gc, axis=1) - np.nansum(gc_medial, axis=0)
    elif drive_type == 'sum_in':
        drive = np.nansum(gc, axis=0)
    else:
        print('unknown drive_type. using sum out.')
        drive = np.nansum(gc, axis=1)
    
    fig, (ax1, ax2) = plot_background_and_cells_drive(cell_centers, background, drive, cst, subplots=True, invert=invert)
    
    gc_max = np.nanmax(gc)
    gc_min = np.nanmin(gc)
    
    for i_from in range(len(cell_centers)):
        for i_to in range(len(cell_centers)):
            if gc[i_from, i_to] > 0:
                
                x_from = cell_centers[i_from, 0]
                y_from = cell_centers[i_from, 1]

                x_to = cell_centers[i_to, 0]
                y_to = cell_centers[i_to, 1]

                if invert:
                    x_from = 512 - x_from
                    x_to = 512 - x_to
                
                width = 2*(gc[i_from, i_to] - gc_min) / (gc_max - gc_min)
                
                if x_from <= x_to: # rostral to caudal
                    ax1.arrow(x_from, y_from, x_to-x_from, y_to-y_from, color='crimson', length_includes_head=True, width=width, alpha=0.6)
                if x_from >= x_to: # caudal to rostral
                    ax2.arrow(x_from, y_from, x_to-x_from, y_to-y_from, color='crimson', length_includes_head=True, width=width, alpha=0.6) 
                

    ax1.axis('off')
    ax2.axis('off')
    if zoom:
        ax1.set_xlim(xlims)
        ax1.set_ylim(ylims)
        ax2.set_xlim(xlims)
        ax2.set_ylim(ylims)
    # plt.suptitle(title + ' neurons only', size=20)
    return fig, (ax1, ax2)

