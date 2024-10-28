# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:40:37 2022

@author: jesusglezs97
"""
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from SRC import models
from SRC import equations_NS
from SRC.foamRW import foamRW
from SRC import mesh_definition as mesh
from SRC.classic_solver import classicSolver
from SRC import plotters as pl

# Fix random seed from reproducibility
tf.random.set_seed(42)
np.random.seed(42)

#%% UPLOAD CASE AND DATA

# Absolute paths using linux
# Here you should modify the paths '/home/jgonzalez/Desktop/cases/...' and '/home/jgonzalez/OpenFOAM' with your current paths
running_directory_DC = '/home/jgonzalez/Desktop/cases/training/baseCase_DC_x8'
classic_solver_path = '/home/jgonzalez/OpenFOAM/OpenFOAM-v2112-Adjoint/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/pimpleFoamSMfinal'
classic_running_directory = '/home/jgonzalez/Desktop/cases/training/baseCase'
OFcaller_path = '/home/jgonzalez/Desktop/cases/OFcaller/libOFcallerfinal.so' 
OF_solver_path = '/home/jgonzalez/OpenFOAM/OpenFOAM-v2112-Adjoint/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/myPimpleADDCFinal'
OF_solver_AD_path = '/home/jgonzalez/OpenFOAM/OpenFOAM-v2112-Adjoint/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/myPimpleVJPDCFinal'
data_path_coarse = '/home/jgonzalez/Desktop/cases/mycase_coarse_x8' 
data_path_fine = '/home/jgonzalez/Desktop/cases/mycase_fine_full'

# Absolute paths using WSL and Windows
# Here you should modify the paths '/mnt/c/Users/jgs_j/Desktop/cases/...' and '/home/jesusglezs97/OpenFOAM/' with your current paths
running_directory_DC = '/mnt/c/Users/jgs_j/Desktop/cases/training/baseCase_DC_x8'
classic_solver_path = '/home/jesusglezs97/OpenFOAM/OpenFOAM-v2112-Adjoint-v2/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/pimpleFoamSMfinal'
classic_running_directory = '/mnt/c/Users/jgs_j/Desktop/cases/training/baseCase'
OFcaller_path = '/mnt/c/Users/jgs_j/Desktop/cases/OFcaller/libOFcallerfinal.so' 
OF_solver_path = '/home/jesusglezs97/OpenFOAM/OpenFOAM-v2112-Adjoint-v2/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/myPimpleADDCFinal'
OF_solver_AD_path = '/home/jesusglezs97/OpenFOAM/OpenFOAM-v2112-Adjoint-v2/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/myPimpleVJPDCFinal'
data_path_coarse = '/mnt/c/Users/jgs_j/Desktop/cases/mycase_coarse_x8' 
data_path_fine = '/mnt/c/Users/jgs_j/Desktop/cases/mycase_fine_full'

column_case = foamRW(data_path_coarse, running_directory_DC, mesh.coarse_mesh_dict_x8(), snapshots = 400, snap_0 = 100, training=True)
column_case.upload_fine_data(data_path_fine, mesh.fine_mesh_dict())

case = column_case

train_in, train_out, timesteps_initial, timesteps_to_predict = case.make_train_data(example_time_steps=1)
case.plotter(18.08,'x_velocity', mode = 'coarse')
case.plotter(18.08,'x_velocity', mode = 'fine')
case.plotter(18.08,'x_velocity', mode = 'downsampled')


#%% DEFINING CUSTOM LOSS

def myCustomLoss(y_true, y_pred): #(batch, timesteps, Ncells)

    red_sum_num = tf.reduce_sum(tf.abs(y_true-y_pred), axis=(-1,-2)) #(batch,)
    red_sum_den = tf.reduce_sum(tf.abs(y_true), axis=(-1,-2)) #(batch,)
    
    red_mean = tf.reduce_mean(red_sum_num/(red_sum_den+1e-8))
    result = red_mean * 100
    
    return result

#%% EXTRACT A CERTAIN PATCH FROM DATA

def extract_patch(patch, timesteps, case, num_samples=1):
    
    train_in, train_out, timesteps_initial, timesteps_to_predict = case.make_train_data(example_time_steps=timesteps)
    timestep_initial_red = timesteps_initial[patch:patch+num_samples]
    timestep_to_predict_red = timesteps_to_predict[patch:patch+num_samples,:]
    train_out_red = {k:v[patch:patch+num_samples,:,:] for k,v in train_out.items()}
    train_in_red = {}
    for k,v in train_in.items():
        if "_b" not in k:
            train_in_red[k] = v[patch:patch+num_samples,:]
        else:
            train_in_red[k] = {k_b:v_b[patch:patch+num_samples,:] for k_b,v_b in v.items()}
            
    return train_in_red, train_out_red, timestep_initial_red, timestep_to_predict_red


#%% RELOAD NETWORK WEIGHTS

def reload_model_weights(model, weights_path):
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(
          optimizer=opt, 
          loss = {"x_velocity": myCustomLoss, "y_velocity": myCustomLoss},
          loss_weights={"x_velocity": 10, "y_velocity": 1},
          run_eagerly=True)
    # Initialize the model
    initial_state, final_state, timestep_initial, timestep_to_predict = extract_patch(0, 1, case)
    _ = model.predict([initial_state, timestep_initial, timestep_to_predict])
    # Upload weights
    model.load_weights(weights_path)
    return model

#%% LOAD TRADITIONAL SOLVER

model_classic = classicSolver(
    equation = equations_NS.NS_kw(), 
    case = case, 
    solver = classic_solver_path, 
    libCaller = OFcaller_path)

# Example of use
train_in, train_out, timesteps_initial, timesteps_to_predict = case.make_train_data(example_time_steps=2)
y_coarse = model_classic.call(train_in, timesteps_initial, timesteps_to_predict)


#%% COMPUTE ERROR OF THE OF CALLING METHOD

t_final = 20
timesteps_initial = tf.expand_dims(tf.convert_to_tensor(case.times[0], dtype=tf.float64), axis=0)
timesteps_to_predict = tf.expand_dims(tf.convert_to_tensor(case.times[1:t_final], dtype=tf.float64), axis=0)
train_in = {}
train_out = {}
for k,v in case.data.items():
    if "_b" not in k:
        train_in[k] = v[0:1,:]
        train_out[k] = tf.expand_dims(v[1:t_final,:], axis=0)
    else:
        train_in[k] = {k_b:v_b[0:1,:] for k_b,v_b in v.items()}
        train_out[k] = {k_b:tf.expand_dims(v_b[1:t_final,:], axis=0) for k_b,v_b in v.items()}

y_classic_test = model_classic.call(train_in, timesteps_initial, timesteps_to_predict)

calling_error = {}
print('Error (%) of OpenFOAM calling method:\n')
for k,v in train_out.items():
    if "_b" not in k:
        error = myCustomLoss(v,y_classic_test[k]).numpy()
    else:
        error = sum([myCustomLoss(v_b, y_classic_test[k][k_b]).numpy() for k_b, v_b in v.items()])
    print(f'Error on field {k}: {error:.5f}')
    calling_error[k] = error
        

#%% Classic solver dataframe
loss_cl_x = [0.11362483, 0.13899286, 0.16178864, 0.18324146, 0.20135545, 0.22243555, 0.24293987, 0.26292434, 0.28246877, 0.30165476, 0.32055593, 0.33921754, 0.3576728, 0.37595347, 0.39408198]
loss_cl_y = [1.6611109, 1.8723145, 2.038344, 2.184454, 2.2491417, 2.3914013, 2.531916, 2.671244, 2.809794, 2.947718, 3.0851438, 3.222196, 3.3590236, 3.4956765, 3.6321826]
loss_cl = {}
loss_cl['x_velocity_loss'] = loss_cl_x
loss_cl['y_velocity_loss'] = loss_cl_y
df_cl = pd.DataFrame(loss_cl)

#%% HYPERPARAMETER TUNNING

import keras_tuner

def call_existing_code(layers, activation, stencil_x, stencil_y, lr):
    OF_solver_path = '/home/jgonzalez/OpenFOAM/OpenFOAM-v2112-Adjoint/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/myPimpleADFinal'
    OF_solver_AD_path = '/home/jgonzalez/OpenFOAM/OpenFOAM-v2112-Adjoint/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/myPimpleVJPFinal'

    model_nn = models.InverseProblemModel(
        equations_NS.NS_kw(),
        case, OF_solver_path, OF_solver_AD_path, OFcaller_path,
        stencil_size=(stencil_x, stencil_y), num_layers=layers,
        constrained_accuracy_order=1, 
        core_model_func=models.DeepConvectionNet,
        target = {'p', 'x_velocity', 'y_velocity'},
        learned_keys = {'x_velocity_edge', 'y_velocity_edge'},
        activation=activation,
        )
    model_nn.compile(
          optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
          loss={"x_velocity": myCustomLoss, "y_velocity": myCustomLoss},
          loss_weights={"x_velocity": 14, "y_velocity": 1},
          #metrics={"x_velocity": 'mae', "y_velocity": 'mae'},
          run_eagerly=True # certain functions during the fitting model.fit() the @tf.function decorator prohibits the execution of functions like tensor.numpy() for performance reasons.
          )
    return model_nn


def build_model(hp):
    layers = hp.Int("layers", min_value=4, max_value=10, step=3)
    activation = hp.Choice("activation", ["relu", "tanh"])
    stencil_x = hp.Choice("stencil_x", [2, 3, 4])
    stencil_y = hp.Choice("stencil_y", [2, 3, 4])
    lr = hp.Choice("lr", values=[1e-2])#, 1e-1, 1e-4])
    # call existing model-building code with the hyperparameter values.
    model = call_existing_code(
        layers=layers, activation=activation, stencil_x=stencil_x, stencil_y=stencil_y, lr=lr
    )
    return model

# tuner = keras_tuner.RandomSearch(
#     hypermodel=build_model,
#     objective="val_accuracy",
#     max_trials=3,
#     executions_per_trial=2,
#     overwrite=True,
#     directory="my_dir",
#     project_name="helloworld",
# )
tuner = keras_tuner.Hyperband(build_model,
                     objective="val_loss",
                     max_epochs=20,
                     factor=3,
                     hyperband_iterations=10,
                     directory="hp_tunning",
                     project_name="try2_3ts_20e_nn_norm",)

tuner.search_space_summary()

# initial_snap = 250
# ts = 3
# train_in, train_out, timesteps_initial, timesteps_to_predict = extract_patch(initial_snap, ts, case, num_samples=3)
train_in, train_out, timesteps_initial, timesteps_to_predict = case.make_train_data(example_time_steps=3)

tuner.search([train_in, timesteps_initial, timesteps_to_predict],
             train_out, validation_split=0.1, epochs=20, 
             verbose=1, shuffle=False)
tuner.results_summary()
best_hps=tuner.get_best_hyperparameters()[0]
final_model = tuner.hypermodel.build(best_hps)
tuner.results_summary()

#%% Learning rate vs loss plot
    
OF_solver_path = '/home/jgonzalez/OpenFOAM/OpenFOAM-v2112-Adjoint/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/myCustomPimpleSMv2AD'
OF_solver_AD_path = '/home/jgonzalez/OpenFOAM/OpenFOAM-v2112-Adjoint/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/myCustomPimpleVJP'
OFcaller_path = '/home/jgonzalez/Desktop/cases/OFcaller/libOFcaller.so' 

lr_array = [10e-5, 50e-5, 10e-4, 50e-4, 10e-3, 50e-3, 10e-2, 50e-2, 10e-1]

loss_x = np.zeros(len(lr_array))
loss_y = np.zeros(len(lr_array))
loss_t = np.zeros(len(lr_array))

epochs = 20

initial_snap_tr = 250
cases = 12
ts = 1
final_snap_tr = initial_snap_tr + cases

for lr in range(len(lr_array)):
    train_in, train_out, timesteps_initial, timesteps_to_predict = column_case.make_train_data(example_time_steps=1)
    train_in = {k:v[initial_snap_tr:final_snap_tr,:] for k,v in train_in.items()}
    train_out = {k:v[initial_snap_tr:final_snap_tr,:,:] for k,v in train_out.items()}
    timesteps_initial = timesteps_initial[initial_snap_tr:final_snap_tr]
    timesteps_to_predict = timesteps_to_predict[initial_snap_tr:final_snap_tr,:]
    # Generate model and compile it
    model_nn = models.InverseProblemModel(
        equations_NS.NSeqk_w(diffusion_coefficient=0.01, density=1000), 
        column_case, OF_solver_path, OF_solver_AD_path, OFcaller_path,
        stencil_size=(4,4), num_layers=4,
        constrained_accuracy_order=1, 
        core_model_func=models.DeepConvectionNet,
        target = {'p', 'x_velocity', 'y_velocity'},
        learned_keys = {'x_velocity_edge', 'y_velocity_edge'},
        activation='tanh',
        )
    ratio_1 = int(df_cl['y_velocity_loss'][0]/df_cl['x_velocity_loss'][0])
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model_nn.compile(
          optimizer=opt, 
          loss={"x_velocity": myCustomLoss, "y_velocity": myCustomLoss},
          loss_weights={"x_velocity": ratio_1, "y_velocity": 1},
          run_eagerly=True
          )
    
    # Running training process
    history_nn = model_nn.fit(
        [train_in, timesteps_initial, timesteps_to_predict], 
        train_out, epochs=epochs, batch_size=2, 
        validation_split=0.1,
        verbose=1, shuffle=True)
    
    train_in, train_out, timesteps_initial, timesteps_to_predict = column_case.make_train_data(example_time_steps=2)
    train_in = {k:v[initial_snap_tr:final_snap_tr,:] for k,v in train_in.items()}
    train_out = {k:v[initial_snap_tr:final_snap_tr,:,:] for k,v in train_out.items()}
    timesteps_initial = timesteps_initial[initial_snap_tr:final_snap_tr]
    timesteps_to_predict = timesteps_to_predict[initial_snap_tr:final_snap_tr,:]

    ratio_2 = int(df_cl['y_velocity_loss'][1]/df_cl['x_velocity_loss'][1])
    
    opt = tf.keras.optimizers.Adam(learning_rate=lr_array[lr])
    model_nn.compile(
          optimizer=opt, 
          loss={"x_velocity": myCustomLoss, "y_velocity": myCustomLoss},
          loss_weights={"x_velocity": ratio_2, "y_velocity": 1},
          run_eagerly=True
          )
    
    # Running training process
    history_nn = model_nn.fit(
        [train_in, timesteps_initial, timesteps_to_predict], 
        train_out, epochs=epochs, batch_size=2, 
        validation_split=0.1,
        verbose=1, shuffle=True)
    
    # Save losses
    df_history = pd.DataFrame(history_nn.history)
    loss_x[lr] = df_history['x_velocity_loss'][epochs-1]
    loss_y[lr] = df_history['y_velocity_loss'][epochs-1]
    loss_t[lr] = df_history['loss'][epochs-1]
    
plt.semilogx(np.array(lr_array), loss_x, marker='.')
plt.title('u_x loss vs learning rate')
plt.xlabel('learning rate (log scale)')
plt.ylabel('loss')
plt.tight_layout()
plt.savefig('lr_vs_loss_u_x.png', dpi=288, bbox_inches='tight')
plt.figure()
plt.semilogx(np.array(lr_array), loss_y, marker='.') 
plt.title('u_y loss vs learning rate')
plt.xlabel('learning rate (log scale)')
plt.ylabel('loss')
plt.tight_layout()
plt.savefig('lr_vs_loss_u_y.png', dpi=288, bbox_inches='tight')
plt.figure()
plt.semilogx(np.array(lr_array), loss_t, marker='.', label = 'total loss vs learning rate')
plt.title('total loss vs learning rate')
plt.xlabel('learning rate (log scale)')
plt.ylabel('loss')
plt.tight_layout()
plt.savefig('lr_vs_loss_u_t.png', dpi=288, bbox_inches='tight')

#%% EXPERIMENT 1: STUDY ON PERFORMANCE OF THE MODEL WITH MESH RESOLUTION

# Relative paths for WS
classic_solver_path = '/home/jgonzalez/OpenFOAM/OpenFOAM-v2112-Adjoint/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/pimpleFoamSMfinal'
classic_running_directory = '/home/jgonzalez/Desktop/cases/training/baseCase'
OFcaller_path = '/home/jgonzalez/Desktop/cases/OFcaller/libOFcallerfinal.so' 
OF_solver_path = '/home/jgonzalez/OpenFOAM/OpenFOAM-v2112-Adjoint/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/myPimpleADDCFinal'
OF_solver_AD_path = '/home/jgonzalez/OpenFOAM/OpenFOAM-v2112-Adjoint/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/myPimpleVJPDCFinal'
meshes = [mesh.coarse_mesh_dict_x4(), mesh.coarse_mesh_dict_x8(), mesh.coarse_mesh_dict_x16(), mesh.coarse_mesh_dict_x32()]
folders = ['../mycase_coarse_x4', '../mycase_coarse_x8', '../mycase_coarse_x16', '../mycase_coarse_x32']
resolution = [4, 8, 16, 32]

# Run the classic solver to compute the weighting factor of the loss function
# Every resolution case will have the same weighting factors (we select x8 case)
column_case = foamRW(folders[1], running_directory_DC, meshes[1], snapshots = 500, snap_0 = 100, training=True)
column_case.upload_fine_data('../mycase_fine_full', mesh.fine_mesh_dict())

model_classic = classicSolver(
        equation = equations_NS.NS_kw(), 
        case = case, 
        solver = classic_solver_path, 
        libCaller = OFcaller_path)

y_classic_pred = model_classic.call(train_in, timesteps_initial, timesteps_to_predict)
error_cl_x = myCustomLoss(train_out['x_velocity'], y_classic_pred['x_velocity'])
error_cl_y = myCustomLoss(train_out['y_velocity'], y_classic_pred['y_velocity'])
print(f'Error classic solver: ({error_cl_x}, {error_cl_y})')
ratio = error_cl_y[-1]/error_cl_x[-1]

for i in range(len(resolution)):
    r = resolution[i]
    running_directory_DC = '/home/jgonzalez/Desktop/cases/training/baseCase_DC'+f'_x{r}'
    mesh_i = meshes[i]
    column_case = foamRW(folders[i], running_directory_DC, mesh_i, snapshots = 500, snap_0 = 100, training=True)
    column_case.upload_fine_data('../mycase_fine_full', mesh.fine_mesh_dict())
    case = column_case
    case.plotter(18.08,'x_velocity', mode = 'fine')
    case.plotter(18.08,'x_velocity', mode = 'downsampled')

    folder_path = f'resolution_study/x{r}/'
    model_nn = models.InverseProblemModel(
        equations_NS.NS_kw(), 
        case, OF_solver_path, OF_solver_AD_path, OFcaller_path,
        stencil_size=(4,3), num_layers=4,
        constrained_accuracy_order=1, 
        core_model_func=models.DeepConvectionNet,
        target = {'p', 'x_velocity', 'y_velocity'},
        learned_keys = {'x_velocity_edge', 'y_velocity_edge'},
        activation='tanh',
        )

    # Construct training data for the desired accumulation of timesteps
    train_in, train_out, timesteps_initial, timesteps_to_predict = case.make_train_data(example_time_steps=1)

    checkpointer = tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath=folder_path+'3.weights.{epoch:02d}.hdf5',
                                                      verbose=1, save_best_only=True, save_weights_only=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=10e-6, min_delta=0.01, verbose=1)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model_nn.compile(
          optimizer=opt, 
          loss = {"x_velocity": myCustomLoss, "y_velocity": myCustomLoss},
          loss_weights={"x_velocity": ratio, "y_velocity": 1},
          run_eagerly=True)

    # Running training process
    history_nn = model_nn.fit(
        [train_in, timesteps_initial, timesteps_to_predict], 
        train_out, epochs=100, batch_size=30, 
        verbose=1,
        callbacks=[checkpointer, reduce_lr])
    
    # Postprocess training results
    df_history = pd.DataFrame(history_nn.history)
    df_history.plot(y=['loss','x_velocity_loss','y_velocity_loss'], marker='.', title='Training loss', xlabel='Epoch', ylabel='Loss')
    plt.savefig(folder_path+'1.Training_loss_1.png', dpi=288, bbox_inches='tight')
    # df_history.plot(y=['val_loss','val_x_velocity_loss','val_y_velocity_loss'], marker='.', title='Validation loss', xlabel='Epoch', ylabel='Loss')
    # plt.savefig(folder_path+'1.Validation_loss_1.png', dpi=288, bbox_inches='tight')
    df_history.to_csv(folder_path+'0.training_losses_1.csv', index=False)
    model_nn.save_weights(folder_path+'2.final_weights_train_1.h5') 
    
loss_cl = {}
loss_cl['x_velocity_loss'] = error_cl_x
loss_cl['y_velocity_loss'] = error_cl_y
df_cl = pd.DataFrame(loss_cl)
df_cl.to_csv(folder_path+'0.CLASSIC_training_and_validation_losses_1.csv', index=False)

#%% TRAINING OF THE MODEL

folder_path = 'CFINAL_noBounding/'

model_nn = models.InverseProblemModel(
    equations_NS.NS_kw(), 
    case, OF_solver_path, OF_solver_AD_path, OFcaller_path,
    stencil_size=(4,3), bounding_perc=0.3,
    constrained_accuracy_order=1, 
    core_model_func=models.DeepConvectionNet,
    target = {'p', 'x_velocity', 'y_velocity'},
    learned_keys = {'x_velocity_edge', 'y_velocity_edge'},
    activation='tanh',
    )

# model_nn.core_model.global_xy.trainable=False

ts_accumulation = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
epochs_per_ts = np.array([100,50,40,30,20,20,20,10,10,10,10,10,10])

for i in range(len(ts_accumulation)):
    ts = ts_accumulation[i]

    # Construct training data for the desired accumulation of timesteps
    train_in, train_out, timesteps_initial, timesteps_to_predict = case.make_train_data(example_time_steps=ts)
    
    ratio = int(df_cl['y_velocity_loss'][ts-1]/df_cl['x_velocity_loss'][ts-1])
    
    checkpointer = tf.keras.callbacks.ModelCheckpoint(monitor='loss', filepath=folder_path+f'{ts}_'+'weights.{epoch:02d}-{loss:.6f}.hdf5',
                                                      verbose=1, save_best_only=True, save_weights_only=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=10e-6, min_delta=0.01, verbose=1)

    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    model_nn.compile(
          optimizer=opt, 
          loss = {"x_velocity": myCustomLoss, "y_velocity": myCustomLoss},
          loss_weights={"x_velocity": ratio, "y_velocity": 1},
          run_eagerly=True)

    # Running training process
    history_nn = model_nn.fit(
        x=[train_in, timesteps_initial, timesteps_to_predict], 
        y=train_out, epochs=epochs_per_ts[i], batch_size=30, 
        verbose=1,
        callbacks=[checkpointer, reduce_lr])
    
    # Postprocess training results
    df_history = pd.DataFrame(history_nn.history)
    df_history.plot(y=['loss','x_velocity_loss','y_velocity_loss'], marker='.', title='Training loss', xlabel='Epoch', ylabel='Loss')
    plt.savefig(folder_path+f'3.Training_loss_{ts}.png', dpi=288, bbox_inches='tight')
    # df_history.plot(y=['val_loss','val_x_velocity_loss','val_y_velocity_loss'], marker='.', title='Validation loss', xlabel='Epoch', ylabel='Loss')
    # plt.savefig(folder_path+f'3.Validation_loss_{ts}.png', dpi=288, bbox_inches='tight')
    df_history.to_csv(folder_path+f'0.training_losses_{ts}.csv', index=False)
    model_nn.save_weights(folder_path+f'final_weights_train_{ts}.h5') 


#%% EXPERIMENT 2: PREDICTION INSIDE THE TRAINING DISTRIBUTION

# Upload training data
running_directory_DC = '/mnt/c/Users/jgs_j/Desktop/cases/training/baseCase_DC_x8'
classic_solver_path = '/home/jesusglezs97/OpenFOAM/OpenFOAM-v2112-Adjoint-v2/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/pimpleFoamSMfinal'
classic_running_directory = '/mnt/c/Users/jgs_j/Desktop/cases/training/baseCase'
OFcaller_path = '/mnt/c/Users/jgs_j/Desktop/cases/OFcaller/libOFcallerfinal.so' 
OF_solver_path = '/home/jesusglezs97/OpenFOAM/OpenFOAM-v2112-Adjoint-v2/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/myPimpleADDCFinal'
OF_solver_AD_path = '/home/jesusglezs97/OpenFOAM/OpenFOAM-v2112-Adjoint-v2/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/myPimpleVJPDCFinal'

case_train = foamRW('../mycase_coarse_x8', running_directory_DC, mesh.coarse_mesh_dict_x8(), snapshots = 100, snap_0 = 300, training=True)
case_train.upload_fine_data('../mycase_fine_full', mesh.fine_mesh_dict())

# Load DC and classic solvers
folder_path = 'CFINAL_noBounding/'

model_nn = models.InverseProblemModel(
    equations_NS.NS_kw(), 
    case_train, OF_solver_path, OF_solver_AD_path, OFcaller_path,
    stencil_size=(4,3), num_layers=4,
    constrained_accuracy_order=1, 
    core_model_func=models.DeepConvectionNet,
    target = {'p', 'x_velocity', 'y_velocity'},
    # num_layers_loc = 4,
    learned_keys = {'x_velocity_edge', 'y_velocity_edge'},
    activation='tanh',
    )

model_nn = reload_model_weights(model_nn, folder_path+'final_weights_train_13.h5', case=case_train)

model_classic = classicSolver(
    equation = equations_NS.NS_kw(), 
    case = case_train, 
    solver = classic_solver_path, 
    libCaller = OFcaller_path)

# Predict on specific batch 
acc_ts = 399 #399
sample = 0  #0
train_in, train_out, timesteps_initial, timesteps_to_predict = extract_patch(sample, acc_ts, case_train,
                                                                             target={'x_velocity', 'y_velocity', 'p', 'k', 'omega', 'nut'})
y_pred_nn_train = model_nn.predict_all([train_in, timesteps_initial, timesteps_to_predict])
y_pred_cl_train = model_classic.call(train_in, timesteps_initial, timesteps_to_predict)

# train_out, y_pred_nn_train, y_pred_cl_train = pl.load_data('result_train_0.3')

# Generation of plots
pplot1 = pl.plotter1_ux(train_out, y_pred_nn_train, y_pred_cl_train)
pplot2 = pl.plotter1_uy(train_out, y_pred_nn_train, y_pred_cl_train)
pplot3 = pl.plotter1_other(train_out, y_pred_nn_train, y_pred_cl_train, 'p', 'p')
pplot4 = pl.plotter1_other(train_out, y_pred_nn_train, y_pred_cl_train, 'nut', 'nut')
pplot5 = pl.plotter1_other(train_out, y_pred_nn_train, y_pred_cl_train, 'omega', 'omega')
pplot6 = pl.plotter1_other(train_out, y_pred_nn_train, y_pred_cl_train, 'k', 'k')
pplot7 = pl.plotter2(train_out, y_pred_nn_train, y_pred_cl_train, case_train)
pplot8 = pl.plotter4_ux(train_out, y_pred_nn_train, y_pred_cl_train, case_train, t=-1, y_coord_plane=30)
pplot9 = pl.plotter4_uy(train_out, y_pred_nn_train, y_pred_cl_train, case_train, t=-1, y_coord_plane=30)
pplot10 = pl.plotter4_other(train_out, y_pred_nn_train, y_pred_cl_train, case_train,'p', 'p',  t=-1, y_coord_plane=30)
pplot11 = pl.plotter4_other(train_out, y_pred_nn_train, y_pred_cl_train, case_train,'nut', 'nut',  t=-1, y_coord_plane=30)
pplot12 = pl.plotter4_other(train_out, y_pred_nn_train, y_pred_cl_train, case_train, 'omega', 'omega',  t=-1, y_coord_plane=30)
pplot13 = pl.plotter4_other(train_out, y_pred_nn_train, y_pred_cl_train, case_train, 'k', 'k',  t=-1, y_coord_plane=30)
pplot14 = pl.plotter4_ux(train_out, y_pred_nn_train, y_pred_cl_train, case_train, t=-1, y_coord_plane=10)
pplot15 = pl.plotter4_uy(train_out, y_pred_nn_train, y_pred_cl_train, case_train, t=-1, y_coord_plane=10)
pplot16 = pl.plotter4_other(train_out, y_pred_nn_train, y_pred_cl_train, case_train,'p', 'p',  t=-1, y_coord_plane=0)
pplot17 = pl.plotter4_other(train_out, y_pred_nn_train, y_pred_cl_train, case_train,'nut', 'nut',  t=-1, y_coord_plane=10)
pplot18 = pl.plotter4_other(train_out, y_pred_nn_train, y_pred_cl_train, case_train, 'omega', 'omega',  t=-1, y_coord_plane=10)
pplot19 = pl.plotter4_other(train_out, y_pred_nn_train, y_pred_cl_train, case_train, 'k', 'k',  t=-1, y_coord_plane=10)

plot3 = pl.plotter1_paper_1(train_out, y_pred_nn_train, y_pred_cl_train)
plot4 = pl.plotter2_u(train_out, y_pred_nn_train, y_pred_cl_train, case_train)
plot5 = pl.plotter2_p(train_out, y_pred_nn_train, y_pred_cl_train, case_train)
plot6 = pl.plotter4_paper1(train_out, y_pred_nn_train, y_pred_cl_train, case_train, t=-1, y_coord_plane=10)
pl.plotter_snapshot_p(train_out['p'],case_train)

#%% EXPERIMENT 3.1: TESTING GENERALIZATION EXTENDING THE DOMAIN IN TIME

# Upload new data
running_directory_ext = '/mnt/c/Users/jgs_j/Desktop/cases/training/baseCase_DC_x8'
classic_solver_path = '/home/jesusglezs97/OpenFOAM/OpenFOAM-v2112-Adjoint-v2/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/pimpleFoamSMfinal'
classic_running_directory = '/mnt/c/Users/jgs_j/Desktop/cases/training/baseCase'
OFcaller_path = '/mnt/c/Users/jgs_j/Desktop/cases/OFcaller/libOFcallerfinal.so' 
OF_solver_path = '/home/jesusglezs97/OpenFOAM/OpenFOAM-v2112-Adjoint-v2/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/myPimpleADDCFinal'
OF_solver_AD_path = '/home/jesusglezs97/OpenFOAM/OpenFOAM-v2112-Adjoint-v2/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/myPimpleVJPDCFinal'

case_ext = foamRW('../mycase_coarse_x8', running_directory_ext, mesh.coarse_mesh_dict_x8(), snapshots = 126, snap_0 = 0, training=True)
case_ext.upload_fine_data('../mycase_fine_extended', mesh.fine_mesh_dict())

# Load DC and classic solvers
folder_path = 'CFINAL_noBounding/'

model_nn = models.InverseProblemModel(
    equations_NS.NS_kw(), 
    case_ext, OF_solver_path, OF_solver_AD_path, OFcaller_path,
    stencil_size=(4,3), num_layers=4,
    constrained_accuracy_order=1, 
    core_model_func=models.DeepConvectionNet,
    target = {'p', 'x_velocity', 'y_velocity'},
    # num_layers_loc = 4,
    learned_keys = {'x_velocity_edge', 'y_velocity_edge'},
    activation='tanh',
    )

model_nn = reload_model_weights(model_nn, folder_path+'final_weights_train_13.h5', case=case_ext)

model_classic = classicSolver(
    equation = equations_NS.NS_kw(), 
    case = case_ext, 
    solver = classic_solver_path, 
    libCaller = OFcaller_path)

# Predict on new data
acc_ts = 125
sample = 0 
train_in, train_out, timesteps_initial, timesteps_to_predict = extract_patch(sample, acc_ts, case_ext, target={'x_velocity', 'y_velocity','k', 'nut', 'omega'})
y_pred_nn_ext = model_nn.predict([train_in, timesteps_initial, timesteps_to_predict])
y_pred_cl_ext = model_classic.call(train_in, timesteps_initial, timesteps_to_predict)
pl.save_data(train_out, y_pred_nn_ext, y_pred_cl_ext, 'result_new_time')
train_out, y_pred_nn_ext, y_pred_cl_ext = pl.load_data('result_ext_0.3')

# Generate plots
pplot = pl.plotter1_paper_1(train_out, y_pred_nn_ext, y_pred_cl_ext, startTime=20.04, ticks=[20,21,22,23,24,25])
pplot = pl.plotter2_u(train_out, y_pred_nn_ext, y_pred_cl_ext, case_train)
plot9 = pl.plotter1_paper_2(train_out, y_pred_nn_ext, y_pred_cl_ext)
pplot = pl.plotter2_p(train_out, y_pred_nn_ext, y_pred_cl_ext, case_train)
pplot = pl.plotter4_paper(train_out, y_pred_nn_ext, y_pred_cl_ext, case_train, t=-1, y_coord_plane=20)

plot9.savefig(f'figures/9.error_all_{sample}_ts_{acc_ts}.pgf', bbox_inches='tight')

pl.plotter6(timesteps_to_predict,y_pred_nn_train, case_train, t=399, folder='../drag_train_nn')
pl.plotter6(timesteps_to_predict,y_pred_cl_train, case_train, t=399, folder='../drag_train_cl')

#%% EXPERIMENT 3.2: TESTING GENERALIZATION EXTENDING THE DOMAIN IN SPACE - TANDEM COLUMNS

# Upload new data
running_directory_tandem = '/home/jgonzalez/Desktop/cases/training/tandem_coarse_x8'
classic_solver_path = '/home/jgonzalez/OpenFOAM/OpenFOAM-v2112-Adjoint/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/pimpleFoamSMfinal'
classic_running_directory = '/home/jgonzalez/Desktop/cases/training/tandem_baseCase'
OFcaller_path = '/home/jgonzalez/Desktop/cases/OFcaller/libOFcallerfinal.so' 
OF_solver_path = '/home/jgonzalez/OpenFOAM/OpenFOAM-v2112-Adjoint/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/myPimpleADDCFinal'
OF_solver_AD_path = '/home/jgonzalez/OpenFOAM/OpenFOAM-v2112-Adjoint/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/myPimpleVJPDCFinal'

case_tandem = foamRW('../tandem_coarse', running_directory_tandem, mesh.tandem_coarse_mesh_dict_x8(), snapshots = 500, snap_0 = 199, training=True)
case_tandem.upload_fine_data('../tandem_fine_ext', mesh.tandem_fine_mesh_dict())

# Load DC and classic solvers
folder_path = 'CFINAL_noBounding/'

model_nn = models.InverseProblemModel(
    equations_NS.NS_kw(), 
    case_tandem, OF_solver_path, OF_solver_AD_path, OFcaller_path,
    stencil_size=(4,3), num_layers=4,
    constrained_accuracy_order=1, 
    core_model_func=models.DeepConvectionNet,
    target = {'p', 'x_velocity', 'y_velocity'},
    # num_layers_loc = 4,
    learned_keys = {'x_velocity_edge', 'y_velocity_edge'},
    activation='tanh',
    )

model_nn = reload_model_weights(model_nn, folder_path+'final_weights_train_13.h5', case=case_tandem)

model_classic = classicSolver(
    equation = equations_NS.NS_kw(), 
    case = case_tandem, 
    solver = classic_solver_path, 
    libCaller = OFcaller_path)

# Predict on new data
acc_ts = 450
sample = 0
train_in, train_out, timesteps_initial, timesteps_to_predict = extract_patch(sample, acc_ts, case_tandem, target={'x_velocity', 'y_velocity', 'p', 'k', 'omega', 'nut'})
y_pred_nn_tandem = model_nn.predict_all([train_in, timesteps_initial, timesteps_to_predict])
y_pred_cl_tandem = model_classic.call(train_in, timesteps_initial, timesteps_to_predict)
pl.save_data(train_out, y_pred_nn_tandem, y_pred_cl_tandem,'result_tandem_ext')
train_out, y_pred_nn_tandem, y_pred_cl_tandem = pl.load_data('result_tandem_0.3')

# Generate plots
# pplot = pl.plotter1_paper_1(train_out, y_pred_nn_tandem, y_pred_cl_tandem, startTime=8.04, ticks=[8,12,16,20])
# pplot = pl.plotter1_paper_2(train_out, y_pred_nn_tandem, y_pred_cl_tandem, startTime=8.04, ticks=[8,12,16,20])
plot10 = pl.plotter1_paper_3(train_out, y_pred_nn_tandem, y_pred_cl_tandem, startTime=8.04)
pplot = pl.plotter2_u(train_out, y_pred_nn_tandem, y_pred_cl_tandem, case_tandem)
plot11 = pl.plotter4_paper2(train_out, y_pred_nn_tandem, y_pred_cl_tandem, case_tandem, t=-1, y_coord_plane=10)

plot10.savefig(f'figures/10.train_plot_1_all_s_{sample}_ts_{acc_ts}.pgf', bbox_inches="tight")
plot11.savefig(f'figures/11.train_plot_4_centerline_s_{sample}_ts_{acc_ts}.pgf', bbox_inches='tight')

pl.plotter6(timesteps_to_predict,y_pred_nn_tandem, case_tandem, t=300, folder='../drag_tandem_nn')
pl.plotter6(timesteps_to_predict,y_pred_cl_tandem, case_tandem, t=300, folder='../drag_tandem_cl')
train_data_tandem = case_tandem.drag_data(example_time_steps=300)
pl.plotter6(timesteps_to_predict, train_data_tandem, case_tandem, t=300, folder='../drag_tandem_tr')

#%% COMPARISON OF PREDICTION TIME

from time import time

# Upload training data
running_directory_DC = '/mnt/c/Users/jgs_j/Desktop/cases/training/baseCase_DC_x8'
classic_solver_path = '/home/jesusglezs97/OpenFOAM/OpenFOAM-v2112-Adjoint-v2/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/pimpleFoamSMfinal'
classic_running_directory = '/mnt/c/Users/jgs_j/Desktop/cases/training/baseCase'
OFcaller_path = '/mnt/c/Users/jgs_j/Desktop/cases/OFcaller/libOFcallerfinal.so' 
OF_solver_path = '/home/jesusglezs97/OpenFOAM/OpenFOAM-v2112-Adjoint-v2/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/myPimpleADDCFinal'
OF_solver_AD_path = '/home/jesusglezs97/OpenFOAM/OpenFOAM-v2112-Adjoint-v2/platforms/linux64GccDPInt64OptA1SDCO_FOAM/bin/myPimpleVJPDCFinal'

case_coarse = foamRW('../mycase_coarse_x8', running_directory_DC, mesh.coarse_mesh_dict_x8(), snapshots = 10, snap_0 = 300, training=True)
case_coarse.upload_fine_data('../mycase_fine_full', mesh.fine_mesh_dict())

# Load DC and classic solvers
folder_path = 'CFINAL_noBounding/'

model_nn = models.InverseProblemModel(
    equations_NS.NS_kw(), 
    case_coarse, OF_solver_path, OF_solver_AD_path, OFcaller_path,
    stencil_size=(4,3), num_layers=4,
    constrained_accuracy_order=1, 
    core_model_func=models.DeepConvectionNet,
    target = {'p', 'x_velocity', 'y_velocity'},
    # num_layers_loc = 4,
    learned_keys = {'x_velocity_edge', 'y_velocity_edge'},
    activation='tanh',
    )

model_nn = reload_model_weights(model_nn, folder_path+'final_weights_train_13.h5', case=case_coarse)

model_classic_coarse = classicSolver(
    equation = equations_NS.NS_kw(), 
    case = case_coarse, 
    solver = classic_solver_path, 
    libCaller = OFcaller_path)

# Predict on specific batch 
acc_ts = 2
sample = 0 
train_in, train_out, timesteps_initial, timesteps_to_predict = extract_patch(sample, acc_ts, case_coarse)
start_time = time()
y_pred_nn_train = model_nn.predict_all([train_in, timesteps_initial, timesteps_to_predict])
print('Time DC model = ', round(time()-start_time,4), ' s')
start_time = time()
y_pred_cl_coarse = model_classic_coarse.call(train_in, timesteps_initial, timesteps_to_predict)
print('Time Baseline model = ', round(time()-start_time,4), ' s')
