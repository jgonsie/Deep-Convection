#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 19:02:00 2023

@author: jesusglezs97
"""
import ctypes
import numpy as np
import tensorflow as tf
import math
from SRC import tensor_ops

class classicSolver(object):
    """Forward problem calling OpenFOAM"""
    
    def __init__(self, equation, case, solver, libCaller):
        self.evolving_keys = sorted(equation.evolving_keys)
        self.evolving_keys.append('phi')
        self.evolving_keys_b = sorted(equation.evolving_keys_b)
        self.evolving_keys_b.append('phi_b')
        self.case = case
        self.ncells = case.ncells_coarse
        self.nfaces = case.FoamMesh.num_inner_face
        self.boundaries = {-1*(v.id+10): {'nfaces': v.num, 'name': k.decode('utf-8')} for k,v in case.FoamMesh.boundary.items() if k not in {b'defaultFaces'}}
        self.boundaries_nfaces_v = [v['nfaces'] for v in self.boundaries.values()]
        self.boundaries_nfaces_sum = sum(self.boundaries_nfaces_v)
        self.size_estimation = (len(self.evolving_keys)-1)*self.ncells +self.nfaces + len(self.evolving_keys_b)*self.boundaries_nfaces_sum
        self.groups = self.get_groups()
        self.running_directory = case.running_directory
        self.solver = solver.encode('utf-8')
        
        self.c_lib = ctypes.CDLL(libCaller)
        self.c_lib.caller.argtypes = [ctypes.c_char_p, ctypes.c_char_p, 
                                      ctypes.c_char_p, ctypes.POINTER(ctypes.c_int),
                                      ctypes.c_int, ctypes.c_int]
        
        self.c_lib.caller.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
        self.c_lib.memory_cleaning.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int]
        

    def get_groups(self):
        '''Creates a dictionary to relate position of the OpenFOAM's return
        and the Python order'''
       
        boundaries = [x['nfaces'] for x in self.boundaries.values()]
        nfields_b = len(self.evolving_keys_b)
        a = np.array(boundaries)*nfields_b
        number = np.array([sum(a[:i+1]) for i in range(len(a))])
        position = number-1
        result = {}
        for it in range(sum(boundaries)*nfields_b):
            if it > position[-1]: it = divmod(it,position[-1])[1]-1
            c = min((x for x in position if x >= it), key=lambda x: x - it)
            group_b = np.where(position == c)[0][0]
            if group_b == 0: div = it
            else: div = it-number[group_b-1]
            group_f, face = divmod(div, boundaries[group_b])
            result[it] = (group_b, group_f, face)
            
        return result
    
    
    def construct_chars (self, state, startTimes, endTimes, n_batch, timestep_id):
        ''' Creates the chars of fields and case directories to send to the OFcaller'''
        
        def char_position (char):
            result = [0]
            for i in range(len(char)):
                if char[i] == '\x00': result.append(i+1)
                
            return result
        
        def construct_argument (files_OF_format):
            result = ''
            for key in files_OF_format.keys():
                result += key + '_dict {' 
                result += files_OF_format[key] + ';};'
            
            result += 'startTime '+str(startTime)+';'
            result += 'endTime '+str(endTime)+';'
            result += '\0'
            
            return result
        
        #convert tf.Tensors into np.arrays for manipulation
        state_proto = tf.nest.map_structure(tf.make_tensor_proto, state)
        startTimes_proto = tf.make_tensor_proto(startTimes)
        endTimes_proto = tf.make_tensor_proto(endTimes)
        state_np = tf.nest.map_structure(tf.make_ndarray, state_proto)
        self.startTimes_np = tf.make_ndarray(startTimes_proto)
        endTimes_np = tf.make_ndarray(endTimes_proto)
        
        dict_fields = ''
        
        for i in range(n_batch):

            startTime = self.startTimes_np[i]
            endTime = endTimes_np[i]
            #extract the interior batch data from the state except for U
            data_batch_int = {k:np.expand_dims(v[i,:], axis=-1) for k,v in state_np.items() if k not in {'x_velocity', 'y_velocity'} and '_b' not in k}
            #extract the boundary batch data from the state except for U_b
            data_batch_bou = {}
            for k_b, v_b in state_np.items():
                if k_b not in {'x_velocity_b', 'y_velocity_b'} and '_b' in k_b:
                    data_batch_bou.update({k_b:{k:np.expand_dims(tf.squeeze(v[i,:]), axis=-1) for k,v in state_np[k_b].items()}})
            data_batch = dict(data_batch_int | data_batch_bou)
            #include phi data from previous timestep if it is not the first timestep simulated
            if timestep_id != 0:
                data_batch['phi'] = np.expand_dims(self.phi[i,:], axis=-1)
                data_batch['phi_b'] = {k:np.expand_dims(tf.squeeze(v[i,:]), axis=-1) for k,v in self.phi_b.items()}
            #extract x_velocity and y_velocity and gather them in a np.array(ncells,2)
            u_x_b = {k:tf.expand_dims(np.squeeze(v[i,:]), axis=-1) for k,v in state_np['x_velocity_b'].items()}
            u_y_b = {k:tf.expand_dims(np.squeeze(v[i,:]), axis=-1) for k,v in state_np['y_velocity_b'].items()}
            data_batch['U'], data_batch['U_b'] = self.case.merge_vectorField(
                np.expand_dims(state_np['x_velocity'][i,:], axis=-1), np.expand_dims(state_np['y_velocity'][i,:], axis=-1),
                u_x_b, u_y_b)
            files_OF_format = self.case.modify_templates(data_batch, startTime, mode = 'string')
            dict_fields ="".join([dict_fields, construct_argument (files_OF_format)])
            
        case_directory_char = (self.running_directory+'\0').encode('utf-8')
        sizes_fields = char_position(dict_fields)
        fields_char = dict_fields.encode('utf-8')

        return case_directory_char, fields_char, sizes_fields

    def OpenFOAMCaller(self, state, x0, x1, x2, x3, x4, x5):
        ''''Call to the external function OpenFOAM'''
        
        ncases = x4
        nfields = len(self.evolving_keys)
        nfields_b = len(self.evolving_keys_b)

        returned_c = self.c_lib.caller(x0, x1, x2, (ctypes.c_int * len(x3))(*x3), x4, x5) #charge the pointer to the array of pointers

        # initializing result containers
        results = {k:np.ones((ncases, self.ncells))*-99. for k in self.evolving_keys if k != 'phi'}
        self.phi = np.ones((ncases, self.nfaces))*-99.
        results_b = {}
        for k in self.evolving_keys_b:
            results_b[k] = {v['name']:np.ones((ncases,v['nfaces']))*-99. for b,v in self.boundaries.items()}

        for i in range(ncases):
            case = ctypes.cast(returned_c, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))[i] #pointer to the array of case i
            for j in range(self.ncells*(nfields-1)):
                group = int(math.floor(j/self.ncells)) #Field represented
                cell = j - group*self.ncells
                results[self.evolving_keys[group]][i,cell] = case[j]
                
            self.phi[i,:] = case[j+1:j+1+self.nfaces]
            end = j+self.nfaces+1

            for k in range(end,end+self.boundaries_nfaces_sum*nfields_b):
                (group_b, group_f, face) = self.groups[k-end]
                results_b[self.evolving_keys_b[group_f]][self.boundaries[group_b]['name']][i,face] = case[k]

        _ = self.c_lib.memory_cleaning(returned_c, ncases) #delete shared memory location

        results_TF_masked = {k:tf.convert_to_tensor(results[k], dtype=tf.float32) 
                      for k,v in results.items()}
        results_b_TF_masked = {k:{b_k:tf.convert_to_tensor(b_v, dtype=tf.float32) 
                            for b_k,b_v in results_b[k].items() if b_k in v.keys()} 
                            for k,v in state.items() if k in self.evolving_keys_b}
        self.phi_b = results_b['phi_b'] 
        
        return {**results_TF_masked, **results_b_TF_masked}
    
    def run_cases (self, state, timestep_current, timestep_to_predict, timestep_id):
        
        n_batch, _ = list(state.values())[0].shape
            
        case_directory_char, fields_char, sizes_fields = self.construct_chars (
            state, timestep_current, timestep_to_predict, n_batch, timestep_id)
           
        results = self.OpenFOAMCaller(state, self.solver, case_directory_char, fields_char, sizes_fields,
                                    n_batch, self.size_estimation)
        
        timestep_id += 1
        return (results, timestep_to_predict, timestep_id)
    
    def call(self, states, timesteps_current, timesteps_to_predict):
        
        step = 0
        
        def advance(evolving_variables, timestep_to_predict):
            evolving_state, evolving_timesteps, timestep_id = evolving_variables
            return self.run_cases(evolving_state, evolving_timesteps, timestep_to_predict, timestep_id)
        
        advanced = tf.scan(
            advance, tf.transpose(timesteps_to_predict), initializer=(states, timesteps_current, step))
        advanced = tensor_ops.moveaxis(advanced[0], source=0, destination=1)
        
        return advanced
