#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:19:34 2024

@author: jesusglezs97
"""
import tensorflow as tf
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rc('axes', axisbelow=True)
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

width = 6.53278

def myCustomLoss(y_true, y_pred): #(batch, timesteps, Ncells)

    red_sum_num = tf.reduce_sum(tf.abs(y_true-y_pred), axis=(-1,-2)) #(batch,)
    red_sum_den = tf.reduce_sum(tf.abs(y_true), axis=(-1,-2)) #(batch,)
    
    red_mean = tf.reduce_mean(red_sum_num/(red_sum_den+1e-8))
    result = red_mean * 100
    
    return result

def save_data(ground_truth, pred_nn, pred_cl, folder='result_data', boundary=False):
    os.makedirs(folder+'/data_truth/', exist_ok=True)
    os.makedirs(folder+'/prediction_nn/', exist_ok=True)
    os.makedirs(folder+'/prediction_cl/', exist_ok=True)
    for k in ground_truth.keys():
        np.save(folder+'/data_truth/'+f'{k}.npy', ground_truth[k].numpy())
        np.save(folder+'/prediction_nn/'+f'{k}.npy', pred_nn[k].numpy())
        np.save(folder+'/prediction_cl/'+f'{k}.npy', pred_cl[k].numpy())
    print('Data saved successfully!')
    return

def load_data(folder, fields = {'k', 'y_velocity', 'p', 'omega', 'x_velocity', 'nut'}):
    ground_truth = {}
    pred_nn = {}
    pred_cl = {}
    
    for k in fields:
        ground_truth[k] = tf.convert_to_tensor(np.load(folder+f'/data_truth/{k}.npy'), dtype=tf.float32)
        pred_nn[k] = tf.convert_to_tensor(np.load(folder+f'/prediction_nn/{k}.npy'), dtype=tf.float32)
        pred_cl[k] = tf.convert_to_tensor(np.load(folder+f'/prediction_cl/{k}.npy'), dtype=tf.float32)
    print('Data loaded successfully!')
    return (ground_truth, pred_nn, pred_cl)
    
def plotter1_ux(ground_truth, pred_nn, pred_cl, t=None):
    ''' Multi-step mean percetage error of u^x vs timesteps simulated'''
    
    if t == None:
        _, ts, _ = ground_truth['x_velocity'].shape
    else: ts = t
  
    err_cl_x = np.zeros(ts+1)
    err_nn_x = np.zeros(ts+1)
    err_cl_x[0] = np.nan
    err_nn_x[0] = np.nan

    for j in range(ts):
        err_cl_x[j+1] = myCustomLoss(ground_truth['x_velocity'][:,0:j+1,:], pred_cl['x_velocity'][:,0:j+1,:])
        err_nn_x[j+1] = myCustomLoss(ground_truth['x_velocity'][:,0:j+1,:], pred_nn['x_velocity'][:,0:j+1,:])
        
    fig, ax = plt.subplots(figsize=(width*0.49,2.3))
    ax.plot(np.arange(ts+1), err_cl_x, c='tab:blue', label = 'Baseline')
    ax.plot(np.arange(ts+1), err_nn_x, c='tab:red', label = 'DC') 
    # ax.set_title('u_x')
    ax.set_xlabel('Timestep')
    ax.set_ylabel(r'$\Psi(\bar{u}^x)$')
    ax.legend()
    ax.grid()

    # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[0:2]]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # fig.legend(lines, labels, loc="lower center", ncol=4)
    # plt.grid()
    # fig.subplots_adjust(bottom=0.05, top=0.95)
    # fig.tight_layout()
    
    return fig

def plotter1_uy(ground_truth, pred_nn, pred_cl, t=None):
    ''' Multi-step mean percetage error of u^y vs timesteps simulated'''
    
    if t == None:
        _, ts, _ = ground_truth['y_velocity'].shape
    else: ts = t
  
    err_cl_x = np.zeros(ts+1)
    err_nn_x = np.zeros(ts+1)
    err_cl_x[0] = np.nan
    err_nn_x[0] = np.nan

    for j in range(ts):
        err_cl_x[j+1] = myCustomLoss(ground_truth['y_velocity'][:,0:j+1,:], pred_cl['y_velocity'][:,0:j+1,:])
        err_nn_x[j+1] = myCustomLoss(ground_truth['y_velocity'][:,0:j+1,:], pred_nn['y_velocity'][:,0:j+1,:])
        
    fig, ax = plt.subplots(figsize=(width*0.49,2.3))
    ax.plot(np.arange(ts+1), err_cl_x, c='tab:blue', label = 'Baseline')
    ax.plot(np.arange(ts+1), err_nn_x, c='tab:red', label = 'DC') 
    # ax.set_title('u_y')
    ax.set_xlabel('Timestep')
    ax.set_ylabel(r'$\Psi(\bar{u}^y)$')
    ax.legend()
    ax.grid()

    # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[0:2]]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # fig.legend(lines, labels, loc="lower center", ncol=4)
    # plt.grid()
    # fig.subplots_adjust(bottom=0.05, top=0.95)
    # fig.tight_layout()
    
    return fig

def plotter1_other(ground_truth, pred_nn, pred_classic, field, name_field, t=None):
    ''' Multi-step mean percetage error of field vs timesteps simulated'''
    
    if t == None:
        _, ts, _ = ground_truth[field].shape
    else: ts = t
  
    err_cl_x = np.zeros(ts+1)
    err_nn_x = np.zeros(ts+1)
    err_cl_x[0] = np.nan
    err_nn_x[0] = np.nan
    
    for j in range(ts):
        err_cl_x[j+1] = myCustomLoss(ground_truth[field][:,j:j+1,:], pred_classic[field][:,j:j+1,:])
        err_nn_x[j+1] = myCustomLoss(ground_truth[field][:,j:j+1,:], pred_nn[field][:,j:j+1,:])
        
    fig, ax = plt.subplots(figsize=(width*0.49,2.3))
    ax.plot(np.arange(ts+1), err_cl_x, c='tab:blue', label = 'Baseline')
    ax.plot(np.arange(ts+1), err_nn_x, c='tab:red', label = 'DC') 
    # ax.set_title('u_y')
    ax.set_xlabel('Timesteps simulated')
    if name_field == 'p':
        ax.set_ylabel(r'$\Psi(\bar{p})$')
    elif name_field == 'nut':
        ax.set_ylabel(r'$\Psi(\nu_t)$')
    elif name_field == 'omega':
        ax.set_ylabel(r'$\Psi(\omega)$')
    elif name_field == 'k':
        ax.set_ylabel(r'$\Psi(k)$')
        
    ax.legend()
    ax.grid()

    # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[0:2]]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # fig.legend(lines, labels, loc="lower center", ncol=4)
    # plt.grid()
    # fig.subplots_adjust(bottom=0.05, top=0.95)
    fig.tight_layout()
    
    return fig

def plotter1_paper_1(ground_truth, pred_nn, pred_cl, t=None, startTime=4, deltaT=0.04):
    ''' Multi-step mean percetage error of velocity vs time simulated'''
    
    if t == None:
        _, ts, _ = ground_truth['x_velocity'].shape
    else: ts = t
  
    err_cl_x = np.zeros(ts+1)
    err_nn_x = np.zeros(ts+1)
    err_cl_x[0] = np.nan
    err_nn_x[0] = np.nan
    err_cl_y = np.zeros(ts+1)
    err_nn_y = np.zeros(ts+1)
    err_cl_y[0] = np.nan
    err_nn_y[0] = np.nan

    for j in range(ts):
        err_cl_x[j+1] = myCustomLoss(ground_truth['x_velocity'][:,j:j+1,:], pred_cl['x_velocity'][:,j:j+1,:])
        err_nn_x[j+1] = myCustomLoss(ground_truth['x_velocity'][:,j:j+1,:], pred_nn['x_velocity'][:,j:j+1,:])
        err_cl_y[j+1] = myCustomLoss(ground_truth['y_velocity'][:,j:j+1,:], pred_cl['y_velocity'][:,j:j+1,:])
        err_nn_y[j+1] = myCustomLoss(ground_truth['y_velocity'][:,j:j+1,:], pred_nn['y_velocity'][:,j:j+1,:])
        
    fig, ax = plt.subplots(1,2,figsize=(width,2.3))

    ax[0].plot(np.arange(startTime,deltaT*(ts+1)+startTime, deltaT), err_cl_x, c='tab:blue', label = 'Baseline')
    ax[0].plot(np.arange(startTime,deltaT*(ts+1)+startTime, deltaT), err_nn_x, c='tab:red', label = 'DC') 
    ax[0].set_xticks([4,8,12,16,20])
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Error '+r'$\bar{u}^x$'+' (%)')
    ax[0].grid()
    ax[1].plot(np.arange(startTime,deltaT*(ts+1)+startTime, deltaT), err_cl_y, c='tab:blue', label = 'Baseline')
    ax[1].plot(np.arange(startTime,deltaT*(ts+1)+startTime, deltaT), err_nn_y, c='tab:red', label = 'DC') 
    ax[1].set_xticks([4,8,12,16,20])
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Error '+r'$\bar{u}^y$'+' (%)', labelpad=-1.)
    ax[1].grid()

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[0:1]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5,1.1))
    fig.tight_layout()
    # fig.subplots_adjust(bottom=0.05, top=0.84, wspace=0.3)
    
    return fig

def plotter1_paper_2(ground_truth, pred_nn, pred_cl, t=None, startTime=4, deltaT=0.04):
    ''' Multi-step mean percetage error of turbulent variables vs time simulated'''
    
    if t == None:
        _, ts, _ = ground_truth['x_velocity'].shape
    else: ts = t
  
    err_cl_x = np.zeros(ts+1)
    err_nn_x = np.zeros(ts+1)
    err_cl_x[0] = np.nan
    err_nn_x[0] = np.nan
    err_cl_y = np.zeros(ts+1)
    err_nn_y = np.zeros(ts+1)
    err_cl_y[0] = np.nan
    err_nn_y[0] = np.nan

    for j in range(ts):
        err_cl_x[j+1] = myCustomLoss(ground_truth['k'][:,j:j+1,:], pred_cl['k'][:,j:j+1,:])
        err_nn_x[j+1] = myCustomLoss(ground_truth['k'][:,j:j+1,:], pred_nn['k'][:,j:j+1,:])
        err_cl_y[j+1] = myCustomLoss(ground_truth['omega'][:,j:j+1,:], pred_cl['omega'][:,j:j+1,:])
        err_nn_y[j+1] = myCustomLoss(ground_truth['omega'][:,j:j+1,:], pred_nn['omega'][:,j:j+1,:])
        
    fig, ax = plt.subplots(1,2,figsize=(width,2.3))

    ax[0].plot(np.arange(startTime,deltaT*(ts+1)+startTime, deltaT), err_cl_x, c='tab:blue', label = 'Baseline')
    ax[0].plot(np.arange(startTime,deltaT*(ts+1)+startTime, deltaT), err_nn_x, c='tab:red', label = 'DC') 
    # ax[0].set_xticks([4,8,12,16,20])
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel(' Error k (%)')
    ax[0].grid()
    ax[1].plot(np.arange(startTime,deltaT*(ts+1)+startTime, deltaT), err_cl_y, c='tab:blue', label = 'Baseline')
    ax[1].plot(np.arange(startTime,deltaT*(ts+1)+startTime, deltaT), err_nn_y, c='tab:red', label = 'DC') 
    # ax[1].set_xticks([4,8,12,16,20])
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Error '+r'$\omega$'+' (%)')#, labelpad=5)
    ax[1].grid()

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[0:1]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5,1.1))
    # fig.subplots_adjust(bottom=0.05, top=0.83, wspace=0.3)
    fig.tight_layout()
    
    return fig

def plotter_snapshot_p (vector_data, case, t = -1, limits = (None, None), colormap = 'jet'):
        '''Plots the data of the square column case in vector form for video generation'''
        
        field = case.vmc(tf.squeeze(vector_data))[t,:,:].numpy().transpose()[:,::-1]
        field[16:16+8] = 101325
        if limits[0] == None: lim1 = tf.reduce_min(vector_data).numpy()
        else: lim1 = limits[0]
        if limits[1] == None: lim2 = tf.reduce_max(vector_data).numpy()
        else: lim2 = limits[1]        
        grid_x, grid_y = case.grid_coarse.get_mesh()
        
        fig, ax = plt.subplots(figsize=(width,width*0.49))
        # plot = plt.figure(figsize=(width*0.49,width*0.49*0.33))
        ax.pcolor(grid_x, grid_y, field, cmap=colormap, vmin=lim1, vmax=lim2)
        ax.set_xticks([])
        ax.set_yticks([])
        # plt.xticks(np.linspace(0,15,4))
        # plt.ylabel('y [m]', fontsize=size)
        # plt.yticks(np.linspace(0,5,3))
        # if legend == True:
        #     a = plt.colorbar(orientation='vertical')
        #     a.set_label(label=text,size=size)
        #     a.ax.tick_params(labelsize=size)
        x = [2-0.03125, 3-0.03125, 3-0.03125, 2-0.03125, 2-0.03125]
        y = [2, 2, 3, 3, 2]
        ax.fill(x, y, color='grey')

        return ax
    
def plotter_snapshot (subplot, vector_data, case, t = -1, limits = (None, None), colormap = 'jet'):
        '''Plots the data of the square column case in vector form for video generation'''
        
        field = case.vmc(tf.squeeze(vector_data))[t,:,:].numpy().transpose()[:,::-1]
        if limits[0] == None: lim1 = tf.reduce_min(vector_data).numpy()
        else: lim1 = limits[0]
        if limits[1] == None: lim2 = tf.reduce_max(vector_data).numpy()
        else: lim2 = limits[1]        
        grid_x, grid_y = case.grid_coarse.get_mesh()
        
        subplot.pcolor(grid_x, grid_y, field, cmap=colormap, vmin=lim1, vmax=lim2)
        
        subplot.set_xticks([])
        subplot.set_yticks([])

        x = [2-0.03125, 3-0.03125, 3-0.03125, 2-0.03125, 2-0.03125]
        y = [2, 2, 3, 3, 2]
        subplot.fill(x, y, color='grey')
        
        return plt.pcolor(grid_x, grid_y, field, cmap=colormap, vmin=lim1, vmax=lim2)

def plotter2(ground_truth, pred_nn, pred_classic, case, t=-1):
    '''Comparison of real fields'''
    fig, axes = plt.subplots(4, 2, figsize=[width, 2/3*width], gridspec_kw={"height_ratios":[1,1,1,0.05]}) 
    
    fields =['x_velocity', 'y_velocity']
    
    ims = []
    
    for i in range(len(fields)):
        lim_max = tf.reduce_max(ground_truth[fields[i]][0,-1,:]).numpy()
        # lim_min = tf.reduce_min(ground_truth[fields[i]][0,-1,:]).numpy()   
        lim_min = -lim_max
        plotter_snapshot(axes[0][i], ground_truth[fields[i]], case, t, limits = (lim_min, lim_max))
        plotter_snapshot(axes[1][i], pred_nn[fields[i]], case, t, limits = (lim_min, lim_max))
        im=plotter_snapshot(axes[2][i], pred_classic[fields[i]], case, t, limits = (lim_min, lim_max))
        ims.append(im)
        
    fig.colorbar(ims[0], cax=axes[3][0], location='bottom', extend = 'both', label = r'$\bar{u}^x$ [m/s]')
    fig.colorbar(ims[1], cax=axes[3][1], location='bottom', extend = 'both', label = r'$\bar{u}^y$ [m/s]')
    
    rows = ['Ground truth', 'DC model', 'Baseline model']
    cols = [r'$\bar{u}^x$', r'$\bar{u}^y$']
    pad = 5 # Separation of titles
    
    # for ax, col in zip(axes[0], cols):
    #     ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
    #                 xycoords='axes fraction', textcoords='offset points',
    #                 ha='center', va='baseline')

    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    rotation=90, xycoords=ax.yaxis.label, textcoords='offset points',
                    ha='right', va='center')
    
    # fig.tight_layout()
    fig.subplots_adjust(wspace = 0.03, hspace = 0.08)
    
    return fig
    

def plotter3(ground_truth, pred_nn, pred_classic, timestep=0):
    '''Comparison of the error in L1-norm'''
    
    fields =['x_velocity', 'y_velocity', 'p']
    text = ['Error u_x [m/s]', 'Error u_y [m/s]', 'Error p [Pa]']
    error_cl = {k: tf.abs(ground_truth[k][0,timestep,:]-pred_classic[k][0,timestep,:]) for k in fields}
    error_nn = {k: tf.abs(ground_truth[k][0,timestep,:]-pred_nn[k][0,timestep,:]) for k in fields}
    
    fig, axes = plt.subplots(3, 2, figsize=[55, 30]) 
    
    count = 1
    for i in range(len(fields)):
        lim_max = np.max([tf.reduce_max(error_nn[fields[i]]).numpy(), tf.reduce_max(error_cl[fields[i]]).numpy()])
        plt.subplot(3, 2, count)           
        case.plotter_video(error_nn[fields[i]], limits = (0., lim_max), magnitude = text[i], legend=True, colormap='Blues')
        plt.subplot(3, 2, count+1)
        case.plotter_video(error_cl[fields[i]], limits = (0., lim_max), magnitude = text[i], legend=True, colormap='Blues')
        count += 2
        
    cols = ['Learned model', 'Baseline model']
    rows = ['u_x', 'u_y', 'p']
    pad = 20 # Separation of titles
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size=50, ha='center', va='baseline')

    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=50, ha='right', va='center')
    
    fig.tight_layout()
    
    
    return fig

def plotter4_ux(ground_truth, pred_nn, pred_cl, case, t=0, y_coord_plane=20): #t=50, plane=30
    '''Predicted u^x along a longitudinal plane'''
    
    centerline_x_nn = case.vmc(tf.squeeze(pred_nn['x_velocity']))[t,y_coord_plane,:].numpy()
    centerline_x_nn[16:16+8] = 0.
    centerline_x_cl = case.vmc(tf.squeeze(pred_cl['x_velocity']))[t,y_coord_plane,:].numpy()
    centerline_x_cl[16:16+8] = 0.
    centerline_x_tr = case.vmc(tf.squeeze(ground_truth['x_velocity']))[t,y_coord_plane,:].numpy()
    centerline_x_tr[16:16+8] = 0.
    
    cell_size = case.mesh_dict_coarse['B1']['cell_size'][0]
    fig, ax = plt.subplots(figsize=(width,2.))
    ax.plot(centerline_x_tr, label = 'Ground true', c='black') 
    ax.plot(centerline_x_cl, '--', label = 'Baseline', c='tab:blue')
    ax.plot(centerline_x_nn, '--',label = 'DC', c='tab:red') 
    ax.set_xlabel('x/D')
    ax.set_ylabel(r'$\bar{u}^x$ [m/s]')
    ax.legend()
    ax.grid()
    ax.set_xticklabels(ax.get_xticks()*cell_size)
    
    return fig

def plotter4_uy(ground_truth, pred_nn, pred_cl, case, t=0, y_coord_plane=20): #t=50, plane=30
    '''Predicted u^y along a longitudinal plane'''
    
    centerline_y_nn = case.vmc(tf.squeeze(pred_nn['y_velocity']))[t,y_coord_plane,:].numpy()
    centerline_y_nn[16:16+8] = 0.
    centerline_y_cl = case.vmc(tf.squeeze(pred_cl['y_velocity']))[t,y_coord_plane,:].numpy()
    centerline_y_cl[16:16+8] = 0.
    centerline_y_tr = case.vmc(tf.squeeze(ground_truth['y_velocity']))[t,y_coord_plane,:].numpy()
    centerline_y_tr[16:16+8] = 0.
    
    cell_size = case.mesh_dict_coarse['B1']['cell_size'][0]
    fig, ax = plt.subplots(figsize=(width,2.))
    ax.plot(centerline_y_tr, label = 'Ground true',  c='black') 
    ax.plot(centerline_y_cl, '--', label = 'Baseline', c='tab:blue')
    ax.plot(centerline_y_nn, '--', label = 'DC', c='tab:red') 
    ax.set_xlabel('x/D')
    ax.set_ylabel(r'$\bar{u}^y$ [m/s]')
    ax.legend()
    ax.grid()
    ax.set_xticklabels(ax.get_xticks()*cell_size)
    
    return fig

def plotter4_other(ground_truth, pred_nn, pred_cl, case, field, name_field, 
                   t=0, y_coord_plane=20): #t=50, plane=30
    '''Predicted field along a longitudinal plane'''
    
    centerline_x_nn = case.vmc(tf.squeeze(pred_nn[field]))[t,y_coord_plane,:].numpy()
    centerline_x_nn[16:16+8] = 0.
    centerline_x_cl = case.vmc(tf.squeeze(pred_cl[field]))[t,y_coord_plane,:].numpy()
    centerline_x_cl[16:16+8] = 0.
    centerline_x_tr = case.vmc(tf.squeeze(ground_truth[field]))[t,y_coord_plane,:].numpy()
    centerline_x_tr[16:16+8] = 0.
    
    cell_size = case.mesh_dict_coarse['B1']['cell_size'][0]
    fig, ax = plt.subplots(figsize=(width,2.))
    ax.plot(centerline_x_tr, label = 'Ground true', c='black') 
    ax.plot(centerline_x_cl, '--', label = 'Baseline', c='tab:blue')
    ax.plot(centerline_x_nn, '--',label = 'DC', c='tab:red')  
    
    ax.set_xlabel('x/D')
    if name_field == 'p':
        ax.set_ylabel(r'$\bar{p})$ $[Pa]$')
    elif name_field == 'nut':
        ax.set_ylabel(r'$\nu_t$ $[m²/s]$')
    elif name_field == 'omega':
        ax.set_ylabel(r'$\omega$ $[s^{-1}]$')
    elif name_field == 'k':
        ax.set_ylabel(r'$k$ $[m²/s²]$')

    ax.legend()
    ax.grid()
    ax.set_xticklabels(ax.get_xticks()*cell_size)
    
    return fig

def plotter4_paper(ground_truth, pred_nn, pred_cl, case, t=0, y_coord_plane=20):
    ''' Multi-step mean percetage error of turbulent variables vs time simulated'''
    
    centerline_x_nn = case.vmc(tf.squeeze(pred_nn['x_velocity']))[t,y_coord_plane,:].numpy()
    centerline_x_nn[16:16+8] = 0.
    centerline_x_cl = case.vmc(tf.squeeze(pred_cl['x_velocity']))[t,y_coord_plane,:].numpy()
    centerline_x_cl[16:16+8] = 0.
    centerline_x_tr = case.vmc(tf.squeeze(ground_truth['x_velocity']))[t,y_coord_plane,:].numpy()
    centerline_x_tr[16:16+8] = 0.
    
    centerline_y_nn = case.vmc(tf.squeeze(pred_nn['y_velocity']))[t,y_coord_plane,:].numpy()
    centerline_y_nn[16:16+8] = 0.
    centerline_y_cl = case.vmc(tf.squeeze(pred_cl['y_velocity']))[t,y_coord_plane,:].numpy()
    centerline_y_cl[16:16+8] = 0.
    centerline_y_tr = case.vmc(tf.squeeze(ground_truth['y_velocity']))[t,y_coord_plane,:].numpy()
    centerline_y_tr[16:16+8] = 0.
    
    centerline_nut_nn = case.vmc(tf.squeeze(pred_nn['nut']))[t,y_coord_plane,:].numpy()
    centerline_nut_nn[16:16+8] = 0.
    centerline_nut_cl = case.vmc(tf.squeeze(pred_cl['nut']))[t,y_coord_plane,:].numpy()
    centerline_nut_cl[16:16+8] = 0.
    centerline_nut_tr = case.vmc(tf.squeeze(ground_truth['nut']))[t,y_coord_plane,:].numpy()
    centerline_nut_tr[16:16+8] = 0.
    
    cell_size = case.mesh_dict_coarse['B1']['cell_size'][0]
    
    
    fig, ax = plt.subplots(3,1,figsize=(width,2.7))
    ax[0].plot(centerline_x_tr, label = 'Ground true', c='black') 
    ax[0].plot(centerline_x_cl, '--', label = 'Baseline', c='tab:blue')
    ax[0].plot(centerline_x_nn, '--',label = 'DC', c='tab:red') 
    ax[0].set_ylabel(r'$\bar{u}^x$ [m/s]')
    ax[0].grid()
    ax[0].set_xticklabels([])
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    ax[1].plot(centerline_y_tr, label = 'Ground true', c='black') 
    ax[1].plot(centerline_y_cl, '--', label = 'Baseline', c='tab:blue')
    ax[1].plot(centerline_y_nn, '--',label = 'DC', c='tab:red') 
    ax[1].set_ylabel(r'$\bar{u}^y$ [m/s]', labelpad=0.3)
    ax[1].grid()
    ax[1].set_xticklabels([])
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    ax[2].plot(centerline_nut_tr, label = 'Ground true', c='black') 
    ax[2].plot(centerline_nut_cl, '--', label = 'Baseline', c='tab:blue')
    ax[2].plot(centerline_nut_nn, '--',label = 'DC', c='tab:red') 
    ax[2].set_xlabel('x/D')
    ax[2].set_ylabel(r'$\nu_t$ $[m²/s]$')
    ax[2].grid()
    ax[2].set_xticklabels(ax[2].get_xticks()*cell_size)
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[0:1]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5,1.1))
    # fig.subplots_adjust(bottom=0.05, top=0.87, wspace=0.3)
    
    rows = [r'\textbf{(a)}', r'\textbf{(b)}', r'\textbf{(c)}']
    pad = [5, 8.5, 5] # Separation of titles
    
    for ax, row, p in zip(ax, rows, pad):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - p, 18),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    ha='right', va='center')
    fig.tight_layout()
        
    return fig

def plotter5_x(faces, velocities, inter_weights, case, t=0):
    '''Comparison modified interpolation weights'''

    # Find shared face from two cells
    def find_face(id_1, id_2, case):
        
        faces_1 = set(case.FoamMesh.cell_faces[id_1])
        faces_2 = set(case.FoamMesh.cell_faces[id_2])
        
        return list(faces_1.intersection(faces_2))[0]
    
    faces = [find_face(item[0],item[1],case) for item in faces]
    field = 'x_velocity_edge'
    fig, axs = plt.subplots(3, 4, figsize=[width*0.49, width*0.55*0.63], gridspec_kw={"height_ratios":[1,1,0.1]}) 
    for ax in axs[-1,:]:
        ax.remove() 
    gs = axs[1, 2].get_gridspec()
    axbig = fig.add_subplot(gs[-1, :])
    x = np.linspace(1, 4, 4)
    y = np.linspace(1, 3, 3)
    xv, yv = np.meshgrid(x, y)
    axs[0,0].streamplot(xv, yv, velocities['x_velocity_edge'][t,faces[0],:,:].numpy(), velocities['y_velocity_edge'][t,faces[0],:,:].numpy(), density=[0.2, 0.2], broken_streamlines=False)
    axs[0,1].streamplot(xv, yv, velocities['x_velocity_edge'][t,faces[1],:,:].numpy(), velocities['y_velocity_edge'][t,faces[1],:,:].numpy(), density=[0.2, 0.2], broken_streamlines=False)
    axs[0,2].streamplot(xv, yv, velocities['x_velocity_edge'][t,faces[2],:,:].numpy(), velocities['y_velocity_edge'][t,faces[2],:,:].numpy(), density=[0.2, 0.2], broken_streamlines=False)
    axs[0,3].streamplot(xv, yv, velocities['x_velocity_edge'][t,faces[3],:,:].numpy(), velocities['y_velocity_edge'][t,faces[3],:,:].numpy(), density=[0.2, 0.2], broken_streamlines=False)
    
    axs[1,0].pcolor(xv, yv, inter_weights[field][t,0,faces[0],:,:].numpy(), cmap='seismic', vmin=-1, vmax=1)
    axs[1,0].axvline(x = 2.5, ymin = 0.34, ymax = 0.65, color ='yellow', linestyle="--")
    axs[1,1].pcolor(xv, yv, inter_weights[field][t,0,faces[1],:,:].numpy(), cmap='seismic', vmin=-1, vmax=1)
    axs[1,1].axvline(x = 2.5, ymin = 0.34, ymax = 0.65, color ='yellow', linestyle="--")
    axs[1,2].pcolor(xv, yv, inter_weights[field][t,0,faces[2],:,:].numpy(), cmap='seismic', vmin=-1, vmax=1)
    axs[1,2].axvline(x = 2.5, ymin = 0.34, ymax = 0.65, color ='yellow', linestyle="--")
    l=axs[1,3].pcolor(xv, yv, inter_weights[field][t,0,faces[3],:,:].numpy(), cmap='seismic', vmin=-1, vmax=1)
    axs[1,3].axvline(x = 2.5, ymin = 0.34, ymax = 0.65, color ='yellow', linestyle="--")
    
    plt.figtext(0.5, 1.02, "Velocity streamlines", va="top", ha="center")
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        
    # fig.colorbar(l, ax=axs.ravel().tolist(), orientation='horizontal',  pad=0.05, label=r'Modified interpolation weights $\tilde{w}$')
    fig.colorbar(l, cax=axbig, location='bottom', extend = 'both', label = r'Modified interpolation weights $\tilde{w}^x$')
    fig.tight_layout()
    
    return fig

def plotter5_y(faces, velocities, inter_weights, case, t=0):
    '''Comparison modified interpolation weights'''

    # Find shared face from two cells
    def find_face(id_1, id_2, case):
        
        faces_1 = set(case.FoamMesh.cell_faces[id_1])
        faces_2 = set(case.FoamMesh.cell_faces[id_2])
        
        return list(faces_1.intersection(faces_2))[0]
    
    faces = [find_face(item[0],item[1],case) for item in faces]
    field = 'y_velocity_edge'
    fig, axs = plt.subplots(3, 4, figsize=[width*0.49, width*0.55*0.63], gridspec_kw={"height_ratios":[1,1,0.1]}) 
    for ax in axs[-1,:]:
        ax.remove() 
    gs = axs[1, 2].get_gridspec()
    axbig = fig.add_subplot(gs[-1, :])
    x = np.linspace(1, 4, 4)
    y = np.linspace(1, 3, 3)
    xv, yv = np.meshgrid(y, x)
    axs[0,0].streamplot(xv, yv, np.rot90(velocities['x_velocity_edge'][t,faces[0],:,:].numpy(), k=-1), np.rot90(velocities['y_velocity_edge'][t,faces[0],:,:].numpy(), k=-1), density=[0.2, 0.2], broken_streamlines=False)
    axs[0,1].streamplot(xv, yv, np.rot90(velocities['x_velocity_edge'][t,faces[1],:,:].numpy(), k=-1), np.rot90(velocities['y_velocity_edge'][t,faces[1],:,:].numpy(), k=-1), density=[0.2, 0.2], broken_streamlines=False)
    axs[0,2].streamplot(xv, yv, np.rot90(velocities['x_velocity_edge'][t,faces[2],:,:].numpy(), k=-1), np.rot90(velocities['y_velocity_edge'][t,faces[2],:,:].numpy(), k=-1), density=[0.2, 0.2], broken_streamlines=False)
    axs[0,3].streamplot(xv, yv, np.rot90(velocities['x_velocity_edge'][t,faces[3],:,:].numpy(), k=-1), np.rot90(velocities['y_velocity_edge'][t,faces[3],:,:].numpy(), k=-1), density=[0.2, 0.2], broken_streamlines=False)
    
    axs[1,0].pcolor(xv, yv, np.rot90(inter_weights[field][t,0,faces[0],:,:].numpy(), k=-1), cmap='seismic', vmin=-1, vmax=1)
    axs[1,0].axhline(y = 2.5, xmin = 0.34, xmax = 0.65, color ='yellow', linestyle="--")
    axs[1,1].pcolor(xv, yv, np.rot90(inter_weights[field][t,0,faces[1],:,:].numpy(), k=-1), cmap='seismic', vmin=-1, vmax=1)
    axs[1,1].axhline(y = 2.5, xmin = 0.34, xmax = 0.65, color ='yellow', linestyle="--")
    axs[1,2].pcolor(xv, yv, np.rot90(inter_weights[field][t,0,faces[2],:,:].numpy(), k=-1), cmap='seismic', vmin=-1, vmax=1)
    axs[1,2].axhline(y = 2.5, xmin = 0.34, xmax = 0.65, color ='yellow', linestyle="--")
    l=axs[1,3].pcolor(xv, yv, np.rot90(inter_weights[field][t,0,faces[3],:,:].numpy(), k=-1), cmap='seismic', vmin=-1, vmax=1)
    axs[1,3].axhline(y = 2.5, xmin = 0.34, xmax = 0.65, color ='yellow', linestyle="--")

    plt.figtext(0.5, 1.02, "Velocity streamlines", va="top", ha="center")
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        
    # fig.colorbar(l, ax=axs.ravel().tolist(), orientation='horizontal',  pad=0.05, label=r'Modified interpolation weights $\tilde{w}$')
    fig.colorbar(l, cax=axbig, location='bottom', extend = 'both', label = r'Modified interpolation weights $\tilde{w}^y$')
    fig.tight_layout()
    
    return fig

def plotter5_xy(faces, velocities, inter_weights, case, t=0):
    '''Comparison modified interpolation weights'''

    # Find shared face from two cells
    def find_face(id_1, id_2, case):
        
        faces_1 = set(case.FoamMesh.cell_faces[id_1])
        faces_2 = set(case.FoamMesh.cell_faces[id_2])
        
        return list(faces_1.intersection(faces_2))[0]
    
    faces = [find_face(item[0],item[1],case) for item in faces]
    field = 'x_velocity_edge'
    fig, axs = plt.subplots(3, 4, figsize=[width, width*0.63], gridspec_kw={"height_ratios":[1,1,0.05]}) 
    for ax in axs[-1,:]:
        ax.remove() 
    gs = axs[1, 2].get_gridspec()
    axbig = fig.add_subplot(gs[-1, :])
    x = np.linspace(1, 4, 4)
    y = np.linspace(1, 3, 3)
    xv1, yv1 = np.meshgrid(x, y)
    xv2, yv2 = np.meshgrid(y, x)
    axs[0,0].streamplot(xv1, yv1, velocities['x_velocity_edge'][t,faces[0],:,:].numpy(), velocities['y_velocity_edge'][t,faces[0],:,:].numpy(), density=[0.2, 0.2], broken_streamlines=False)
    axs[0,1].streamplot(xv1, yv1, velocities['x_velocity_edge'][t,faces[1],:,:].numpy(), velocities['y_velocity_edge'][t,faces[1],:,:].numpy(), density=[0.2, 0.2], broken_streamlines=False)
    axs[0,2].streamplot(xv2, yv2, np.rot90(velocities['y_velocity_edge'][t,faces[2],:,:].numpy(), k=-1), np.rot90(velocities['x_velocity_edge'][t,faces[2],:,:].numpy(), k=-1), density=[0.2, 0.2], broken_streamlines=False)
    axs[0,3].streamplot(xv2, yv2, np.rot90(velocities['y_velocity_edge'][t,faces[3],:,:].numpy(), k=-1), np.rot90(velocities['x_velocity_edge'][t,faces[3],:,:].numpy(), k=-1), density=[0.2, 0.2], broken_streamlines=False)
    
    axs[1,0].pcolor(xv1, yv1, inter_weights['x_velocity_edge'][t,0,faces[0],:,:].numpy(), cmap='seismic', vmin=-1, vmax=1)
    axs[1,0].axvline(x = 2.5, ymin = 0.34, ymax = 0.65, color ='black', linestyle="--")
    axs[1,1].pcolor(xv1, yv1, inter_weights['x_velocity_edge'][t,0,faces[1],:,:].numpy(), cmap='seismic', vmin=-1, vmax=1)
    axs[1,1].axvline(x = 2.5, ymin = 0.34, ymax = 0.65, color ='black', linestyle="--")
    axs[1,2].pcolor(xv2, yv2, np.rot90(inter_weights['y_velocity_edge'][t,0,faces[2],:,:].numpy(), k=-1), cmap='seismic', vmin=-1, vmax=1)
    axs[1,2].axhline(y = 2.5, xmin = 0.34, xmax = 0.65, color ='black', linestyle="--")
    l=axs[1,3].pcolor(xv2, yv2, np.rot90(inter_weights['y_velocity_edge'][t,0,faces[3],:,:].numpy(), k=-1), cmap='seismic', vmin=-1, vmax=1)
    axs[1,3].axhline(y = 2.5, xmin = 0.34, xmax = 0.65, color ='black', linestyle="--")
    
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        
    # fig.colorbar(l, ax=axs.ravel().tolist(), orientation='horizontal',  pad=0.05, label=r'Modified interpolation weights $\tilde{w}$')
    fig.colorbar(l, cax=axbig, location='bottom', label = r'Modified interpolation weights $\tilde{w}$')
    fig.tight_layout()
    
    return fig

def plotter6(endTimes, pred_nn, case, t=20, folder='../drag_computation'):
    '''Comparison of drag coefficient'''
    
    state_proto_nn = tf.nest.map_structure(tf.make_tensor_proto, pred_nn)
    # state_proto_cl = tf.nest.map_structure(tf.make_tensor_proto, pred_cl)
    endTimes_proto = tf.make_tensor_proto(tf.squeeze(endTimes))
    state_np_nn = tf.nest.map_structure(tf.make_ndarray, state_proto_nn)
    # state_np_cl = tf.nest.map_structure(tf.make_ndarray, state_proto_cl)
    endTimes_np = tf.make_ndarray(endTimes_proto)
    os.makedirs(folder, exist_ok=True)
    
    for i in range(t):
        endTime = endTimes_np[i]
        #extract the interior batch data from all the states except for U
        data_batch_int = {k:np.expand_dims(v[0,i,:], axis=-1) for k,v in state_np_nn.items() if k not in {'x_velocity', 'y_velocity'} and '_b' not in k}
        #extract the boundary batch data from all the states except for U_b
        data_batch_bou = {}
        for k_b, v_b in state_np_nn.items():
            if k_b not in {'x_velocity_b', 'y_velocity_b'} and '_b' in k_b:
                data_batch_bou.update({k_b:{k:np.expand_dims(tf.squeeze(v[0,i,:]), axis=-1) for k,v in state_np_nn[k_b].items()}})
        data_batch = dict(data_batch_int | data_batch_bou)
        #extract x_velocity and y_velocity and gather them in a np.array(ncells,2)
        u_x_b = {k:tf.expand_dims(np.squeeze(v[0,i,:]), axis=-1) for k,v in state_np_nn['x_velocity_b'].items()}
        u_y_b = {k:tf.expand_dims(np.squeeze(v[0,i,:]), axis=-1) for k,v in state_np_nn['y_velocity_b'].items()}
        data_batch['U'], data_batch['U_b'] = case.merge_vectorField(
            np.expand_dims(state_np_nn['x_velocity'][0,i,:], axis=-1), np.expand_dims(state_np_nn['y_velocity'][0,i,:], axis=-1),
            u_x_b, u_y_b)            
        templates = case.modify_templates(data_batch, endTime)
        case.write_BC_files(templates, endTime, folder)
    
    # fig, axes = plt.subplots(3, 2, figsize=[55, 30]) 
    return 0

def plotter7(ground_truth, pred_nn, pred_classic, t=0):
    '''Study on Reynolds effect'''
    
    fig, axes = plt.subplots(3, 2, figsize=[55, 30]) 
    return fig
