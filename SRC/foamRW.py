#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:22:05 2023

@author: jesusglezs97
"""

import numpy as np
import tensorflow as tf
import openfoamparser as Ofpp
import os
import matplotlib.pyplot as plt
from typing import Dict, List
from joblib import Parallel, delayed

from SRC import grids
from SRC import tensor_ops
from SRC import equations_NS 


"""Contains all the necessary tools for loading and converting data from 
OpenFOAM format to Python format"""


def get_mesh_ncells(mesh_dict):
    '''Returns the number of cells in the mesh'''
    
    n_x_cells, n_y_cells, _, _, _ = get_mesh_statistics(mesh_dict)
    result = n_x_cells*n_y_cells
    
    for block in mesh_dict.keys():
        if mesh_dict[block]['isEmpty'] == True: 
            result -= mesh_dict[block]['n_cells'][0]*mesh_dict[block]['n_cells'][0]
            
    return result

def get_mesh_statistics(mesh_dict):
    '''Gets relevant data from the mesh dictionary'''
    
    n_x_cells, n_y_cells = (0,0) #Total number of cells (120,40)
    relPosX_old, relPosY_old = (0,0)
    prev = 0
    block_cells = {} #Dict with the number of cells of each block [256, 128, 256, ...]
    divs_x = [0] #Cell divisions on x axis
    divs_y = [0] #Cell divisions on y axis
    

    for block in mesh_dict.keys(): 
        relPosX_new, relPosY_new = mesh_dict[block]['relPos']
        if relPosX_new != relPosX_old: 
            n_x_cells += mesh_dict[block]['n_cells'][0]
            divs_x.append(divs_x[-1]+mesh_dict[block]['n_cells'][0])
            
        if relPosY_new > relPosY_old: 
            n_y_cells += mesh_dict[block]['n_cells'][1]
            divs_y.append(divs_y[-1]+mesh_dict[block]['n_cells'][1])
            relPosY_old = relPosY_new
            
        if mesh_dict[block]['isEmpty'] == False:    
            block_cells[block] = (prev, prev+np.prod(mesh_dict[block]['n_cells']))
            prev += np.prod(mesh_dict[block]['n_cells'])
        relPosX_old = relPosX_new
        

        
    return (n_x_cells, n_y_cells, block_cells, divs_x, divs_y)

def upload_data(case_directory, times, fieldNames, case, dtype=tf.float32):
    '''Reads the data from OpenFOAM and loads it in Python in vector form'''
    
    result = {}
    ncells = case.num_cell+1
    ncells_b = {}
    for k in case.boundary.keys():
        ncells_b[k] = len(list(case.boundary_cells(k)))

    def upload_time(time, field):
        def prepare_data(reading, ncells):
            if type(reading).__module__ != np.__name__: # reading is not a np.array (velocity) of ncells
                result_t = (reading * np.ones((ncells)))
            elif reading.shape[0] != ncells: # reading is a np.array of float/int
                result_t = (reading * np.ones((ncells, len(reading))))
            else: # reading is a np.array of ncells
                result_t = (reading)
            return result_t
        
        route = os.path.join(case_directory, str(time), field)
        upload = Ofpp.parse_field_all(route)
        result_centers = prepare_data(upload[0], ncells)
        result_boundaries = {}
        for k,v in upload[1].items():
            if len(list(v.keys())) >= 1 and b'value' in list(v.keys()):
                result_boundaries[k] = prepare_data(v[b'value'], ncells_b[k])

        return  (result_centers, result_boundaries)
   
    for field in fieldNames:
        temp = Parallel(n_jobs=-1)(delayed(upload_time)(timer, field) for timer in times)    
        center_values = [t[0] for t in temp]
        if sorted(temp[0][1].keys()) != sorted(temp[-1][1].keys()):
            raise ValueError (f'foamRW: Not same boundaries for all timesteps in field: {field}.\nRemember to substitute $internalField by the corresponding value in the OpenFOAM field files!')
        boundary_values = {k:[dic[k] for _, dic in temp] for k in temp[0][1].keys()}

        if field == 'U':
            result['x_velocity'] = tf.convert_to_tensor([i[:,0] for i in center_values], dtype=dtype)
            result['y_velocity'] = tf.convert_to_tensor([i[:,1] for i in center_values], dtype=dtype)
            result['x_velocity_b']  = {}
            result['y_velocity_b']  = {}
            for boundary in ncells_b.keys():
                if boundary in boundary_values.keys():
                    result['x_velocity_b'][boundary.decode('utf-8')] = tf.convert_to_tensor([i[:,0] for i in boundary_values[boundary]], dtype=dtype)
                    result['y_velocity_b'][boundary.decode('utf-8')] = tf.convert_to_tensor([i[:,1] for i in boundary_values[boundary]], dtype=dtype)
                # else:
                #     result['x_velocity_b'][boundary] = None
                #     result['y_velocity_b'][boundary] = None
        else:
            result[field] = tf.convert_to_tensor(center_values, dtype=dtype)
            result[field+'_b']  = {}
            for boundary in ncells_b.keys():
                if boundary in boundary_values.keys():
                    # print(boundary)
                    result[field+'_b'][boundary.decode('utf-8')] = tf.convert_to_tensor(boundary_values[boundary], dtype=dtype)
                # else:
                #     result[field+'_b'][boundary] = None
        
    return result

def vector_to_matrix_conversor(mesh_dict, vector_data, dtype=tf.float32):
    '''Transforms data from vector form {key: tf.Tensor(batch,ncells)} into 
    matrix form {key: tf.Tensor(batch, nx, ny)}'''
    
    n_x_cells, n_y_cells, block_cells, divs_x, divs_y = get_mesh_statistics(mesh_dict)
    
    result = {}
    
    for field, data in vector_data.items():
        temp1 = []
        batch,_ = data.shape
        for b in range(batch): 
                temp2 = np.ones([n_y_cells, n_x_cells])*-99
                for block in mesh_dict.keys():
                    if mesh_dict[block]['isEmpty'] == False:
                        pos_x_ini = divs_x[mesh_dict[block]['relPos'][0]-1]
                        pos_y_ini = divs_y[mesh_dict[block]['relPos'][1]-1]
                        pos_x_end = divs_x[mesh_dict[block]['relPos'][0]]
                        pos_y_end = divs_y[mesh_dict[block]['relPos'][1]]
                        temp2[pos_y_ini:pos_y_end,pos_x_ini:pos_x_end] = np.reshape(
                            data[b,block_cells[block][0]:block_cells[block][1]],
                            (mesh_dict[block]['n_cells'][1],mesh_dict[block]['n_cells'][0]))
                    
                # temp1.append(np.transpose(temp2[::-1,:])) #Flip through the y axis and transpose result (40,120)->(120,40)
                temp1.append(temp2[::-1,:])
                result[field] = tf.convert_to_tensor(temp1, dtype=dtype)
        
    return  result


# TODO: Remove these two functions and adapt the code in downsampling
def vector_to_matrix_conversor_tf(mesh_dict, vector_data, dtype=tf.float32):
    '''Transforms data from vector form {key: np.array(batch,ncells)} into 
    matrix form {key: tf.Tensor(batch, nx, ny)} using tf operations'''
    
    batch, ncells = vector_data[list(vector_data.keys())[0]].shape
    cells_position = vector_to_matrix_conversor(mesh_dict, {'cells': np.expand_dims(tf.range(ncells), axis=0)})
    indices = tf.tile(tf.cast(cells_position['cells'], dtype=tf.int32), [batch,1,1])
    bc_positions = tf.where(tf.less(indices, 0)) 
    indices_corr = tf.tensor_scatter_nd_update(indices, bc_positions, tf.zeros(len(bc_positions), dtype=tf.int32))
    
    result = {}
    for field, data in vector_data.items():    
        temp = tf.gather(data, indices_corr, axis = 1, batch_dims = 1)
        result[field] = tf.tensor_scatter_nd_update(temp, bc_positions, -99*tf.ones(len(bc_positions), dtype=dtype))
    return  result

def matrix_to_vector_conversor_tf (mesh_dict, ncells, matrix_data, dtype=tf.float32):
    '''Transforms data from matrix form {key: tf.Tensor(batch, nx, ny)} into 
    vector form {key: np.array(batch,ncells)}using tf operations'''
    
    batch, _, _ = matrix_data[list(matrix_data.keys())[0]].shape
    cells = tf.range(ncells)[tf.newaxis,:]
    cells_matrix = vector_to_matrix_conversor_tf(mesh_dict, {'x':cells}, dtype=tf.int32)['x']
    indices = tf.reshape(cells_matrix, [1,-1])
    # Option 1
    # indices = tf.squeeze(tf.stack([tf.where(indices[0,:] == i) for i in range(ncells)]))[tf.newaxis,:]
    # option 2
    mask = tf.equal(tf.transpose(indices), tf.range(ncells))
    indices = tf.argmax(mask, axis=0, output_type=tf.int32)[tf.newaxis,:]
    
    indices = tf.tile(indices, [batch, 1])
    
    result = {}
    for field, data in matrix_data.items():
        flat = tf.reshape(data, [batch,-1])
        result[field] = tf.gather(flat, indices, axis = 1, batch_dims = 1)
    return result
    
def matrix_to_vector_conversor (mesh_dict, ncells, matrix_data, dtype=tf.float32):
    '''Transforms data from matrix {key: (tf.Tensor)} form into vector form 
    {time:{key: (np.array)}}'''
    
    _, _, block_cells, divs_x, divs_y = get_mesh_statistics(mesh_dict)
    result = {}
    
    #Turn {key: tf.Tensor} into {key: np.array}
    matrix_data_proto = tf.nest.map_structure(tf.make_tensor_proto, matrix_data)
    matrix_data_np = tf.nest.map_structure(tf.make_ndarray, matrix_data_proto)
    
    for field, data in matrix_data_np.items():
        temp1 = []
        batch, _, _ = data.shape
        flipped_data = data[:,::-1,:]
        for b in range(batch): 
            temp2 = np.zeros([ncells])
            
            for block in mesh_dict:
                    if mesh_dict[block]['isEmpty'] == False:
                        pos_x_ini = divs_x[mesh_dict[block]['relPos'][0]-1]
                        pos_y_ini = divs_y[mesh_dict[block]['relPos'][1]-1]
                        pos_x_end = divs_x[mesh_dict[block]['relPos'][0]]
                        pos_y_end = divs_y[mesh_dict[block]['relPos'][1]]

                        temp2[block_cells[block][0]:block_cells[block][1]] = flipped_data[b,pos_y_ini:pos_y_end,pos_x_ini:pos_x_end].flatten()
            
            temp1.append(temp2)
        result[field] = tf.convert_to_tensor(temp1, dtype=dtype)
    
    return result


class foamRW(object):
    
    def __init__(self, case_directory, running_directory, mesh_dict_coarse, 
                 snap_0 = 0, snapshots = 0, equation = equations_NS.NS_kw(),  
                 read_coarse_data = False, training = False, dtype=tf.float32):
        
        self.case_directory = case_directory
        self.running_directory = running_directory
        self.mesh_dict_coarse = mesh_dict_coarse
        self.equation = equation
        self.snap_0 = snap_0
        self.snapshots = snapshots
        self.training = training
        
        self.FoamMesh = Ofpp.FoamMesh(self.case_directory)
        self.ncells_coarse = get_mesh_ncells(self.mesh_dict_coarse)
        
        # Get the temporal discretization and the timestep size
        self.times_origin, self.deltaT = self.get_timesteps(case_directory, 
                                                            training)
        if snapshots != 0: 
            self.times = self.times_origin[snap_0:snap_0+snapshots]
        
        # Get all the field names involved in OpenFoam
        self.fieldNames_all = self.get_fieldNames(self.equation) 
        
        # Upload coarse data in case user wants and the case is for training
        if (read_coarse_data == True and training == True):
            self.data = upload_data(self.case_directory, self.times, 
                                    self.fieldNames_all, self.FoamMesh, dtype)
            
        # Get mesh features (just valid for cartesian equispaced meshes)        
        self.n_x_cells, self.n_y_cells, self.block_cells, self.divs_x, self.divs_y = get_mesh_statistics(self.mesh_dict_coarse) 
        self.grid_coarse = grids.Grid(
            size_x=self.n_x_cells , size_y=self.n_y_cells, 
            step=self.mesh_dict_coarse['B1']['cell_size'][0])
        
        # Get the indexing vectors for vector-matrix conversion
        self.indices_vmc, self.bc_positions, self.indices_mvc = self.build_indices_conversion(self.mesh_dict_coarse,
                                                                self.ncells_coarse)
        
        # Build the OF fields files templates
        if training == None:
            self.file_templates_0, 
            self.file_templates_1 = self.get_templates(from_template = False)
        else:
            self.file_templates_0, self.file_templates_1 = self.get_templates()
        
        
    def get_timesteps (self, route, training):
        '''Returns a list with the timesteps looking at the OpenFOAM case 
        folder, and the deltaT'''
        
        if training == True:
            content = os.listdir(route)
            times = []
            for folder in content:
                 try:
                     times.append(float(folder))
                 except ValueError:
                     continue
            final_times = np.sort(np.array(times))
            
            check_times = np.linspace(final_times[0],final_times[-1],len(final_times))
            assert final_times.all() == check_times.all()
            
            final_times = list(final_times) #Convert np.array into list to change float to int 
            for i in range(len(final_times)):
                if final_times[i] == int(final_times[i]): 
                    final_times[i] = int(final_times[i])
    
            deltaT = final_times[1] - final_times[0] # supposing homogeneous timesteps
        else:   
        #TODO: Create an method to create timesteps from reading from controldict
            raise ValueError ("Feature pending of implementation. Please, select training=True")
            
        return final_times, deltaT
    
    def get_fieldNames (self, equation):
        '''Get each field name looking at equation definition'''
        
        content = {equation.key_definitions[key][1] for key in equation.evolving_keys}
        
        return content 
    
    def get_fieldNames_bypath (self, path):
        '''Get each field name looking at the OpenFOAM folder'''
        
        content = os.listdir(path)
        
        return content 
    
    def clean_template (self, template):
        ''' Cleans the templates of the boundary condition files'''
        
        intervals = []
        for line in range (len(template)):
            if b'nonuniform List' in template[line] and b'}' not in template[line+1]:
                n = 0
                line_iter = line+1
                while n != 1:
                    if template[line_iter].startswith(b')\n'): n = 1
                    else:
                        line_iter += 1
                intervals.append((line+1, line_iter+1))
               
        start = 0
        end = 0  
        dif = 0
        for i in intervals:
            start = i[0]-dif
            end = i[1]-dif
            dif += i[1]-i[0]
            del template[start:end]
            
        return template    
    
    def get_templates (self, from_template = True):
        ''' Obtains the templates of the boundary condition 
        files from the templates folder or from the boundary conditions folder'''
        
        def get_template_entries (fieldName, template):
            '''Generates a list with the lines where to insert new information'''
            result = []
            for line in range(len(template)):
                if b'internalField   nonuniform List' in template[line]:
                    result.append((line+1, b'internalField'))
                elif b'value           nonuniform List' in template[line] and b'}' not in template[line+1]:
                    n = 0
                    line_iter = line-1
                    while n != 1:
                        if template[line_iter].startswith(b'    {\n'):
                            boundary_name = [i for i in self.FoamMesh.boundary.keys() if i in template[line_iter-1]][0]

                            if fieldName == 'Ufaces':
                                #Extract from the field the bounday face values
                                boundary_cells = list(self.FoamMesh.boundary_cells(boundary_name))
                                boundary = []
                                for i in boundary_cells:
                                    candidate = list(set(self.FoamMesh.cell_faces[i]) & set(self.FoamMesh.cell_faces[i-1]))
                                    if len(candidate) == 1: boundary.append(candidate[0])
                                    else: raise ValueError ('foamRW: Not unique coincidence looking for intersection of faces')
                            else:
                                #Extract from the field the bounday cell values
                                # boundary = list(self.FoamMesh.boundary_cells(boundary_name))
                                boundary = boundary_name.decode('utf-8')
                            result.append((line+1, boundary))
                            n = 1
                        else:
                            line_iter -= 1
                            
            return result
        
        if from_template == True:
            source_0 = 'template/zero'
            source_1 = 'template/non_zero'
        else:
            source_0 = str(self.times[0])
            source_1 = str(self.times[-1])
            
        fieldNames_0 = self.get_fieldNames_bypath(os.path.join(self.case_directory, source_0))
        fieldNames_1 = self.get_fieldNames_bypath(os.path.join(self.case_directory, source_1))
        
        templates_0 = {}
        templates_1 = {}
        self.info_templates = {}
        
        for field in fieldNames_0:
            route_0 = os.path.join(self.case_directory, source_0, field)
            try:
                with open(route_0, "rb") as f:
                    content_0 = f.readlines()

            except FileNotFoundError:
                    print('file not found: %s'%route_0)
                    
            templates_0[field] = content_0
            
        for field in fieldNames_1:
            route_1 = os.path.join(self.case_directory, source_1, field)            
            try:
                with open(route_1, "rb") as f:
                    content_1 = f.readlines()

            except FileNotFoundError:
                    print('file not found: %s'%route_1)
                    
            templates_1[field] = self.clean_template(content_1)
            self.info_templates[field] = get_template_entries(field, templates_1[field])
            
        return templates_0, templates_1
    
#------------------------------------------------------------------------------

    def write_BC_files (self, templates, time, destination_dir):
        '''Writes the new boundary condition files into the appropriate folder'''
        
        directory = os.path.join(destination_dir, str(time))
        if not os.path.exists(directory):
            os.mkdir(directory)
        for field_name, field_template in templates.items():
            route = os.path.join(directory, field_name)
            with open(route, "wb") as f:
                    for line in field_template:
                        f.write(line)
        
        return None
    
    def formatting_Field(self, fieldName: str, field: np.array):
        '''Generates a string with the format of OpenFOAM'''
        
        ncells, components = field.shape
        intro = [str.encode("".join([str(ncells),"\n"])), str.encode("(\n")]
        if components == 1 and fieldName == 'p':
            core = [str.encode("".join([str(line[0]), "\n"])) for line in field]
        elif components == 1:
            core = [str.encode("".join([str(line[0]), "\n"])) for line in field]
        elif components == 3:
            core = [str.encode("".join(["(", str(line[0]), " ", str(line[1]), " ", str(line[2]), ")\n"])) for line in field]
        
        outro = [str.encode(")\n")]
        
        return [*intro, *core, *outro]
        
    def insert_field (self, fieldName: str, new_field: Dict[str, np.array], template: List[bytes]) -> List[bytes]:
        '''Receives a new field and insert it into the appropriate template of the
        boundary condition file'''
        
        template = template.copy()
        
        for insert in reversed(self.info_templates[fieldName]): # Starting from the end to not change the line id
            if insert[1] == b'internalField':
                template[insert[0]:insert[0]] = self.formatting_Field(fieldName, new_field[fieldName])
            elif fieldName == 'Ufaces' or fieldName =='dU':
                template[insert[0]:insert[0]] = self.formatting_Field(fieldName, new_field[fieldName][np.ix_(insert[1])])
            else:
                template[insert[0]:insert[0]] = self.formatting_Field(fieldName, new_field[fieldName+'_b'][insert[1]])
            
        return template
    
    def modify_templates (self, new_fields, time, mode = 'bytes'):
        '''Modify all the templates of the boundary condition files to include 
        the new fields'''
        new_state = {}
        fields_interior = {k for k in new_fields.keys() if '_b' not in k}
        fields_boundary = {k for k in new_fields.keys() if '_b' in k}

        for field in fields_interior:
            if field == 'dU': temp_state = self.insert_field('dU', new_fields,
                    self.file_templates_1['U'])
            elif field != 'Ufaces' and field+'_b' not in fields_boundary:
                raise ValueError (f'foamRW: The field {field} has not its boundary values')
                
            if time == 0 and field != 'Ufaces': 
                temp_state = self.file_templates_0[field]
            else:
                if field == 'Ufaces': value = {field: new_fields[field]}
                else: value = {field: new_fields[field], field+'_b': new_fields[field+'_b']}
                temp_state = self.insert_field(field, value, self.file_templates_1[field])
                temp_state[12] = b'    location    "'+str.encode(str(time))+b'";\n'
            if mode == 'string':
                temp_state_str =''
                for i in temp_state:
                    temp_state_str += i.decode('utf-8')
                new_state[field] = temp_state_str
            else:
                new_state[field] = temp_state 
        
        return new_state    
    
    def get_derivative_template(self, array):
        
        template = self.file_templates_1['U'].copy()
        
        for insert in reversed(self.info_templates['U']):
            if insert[1] == b'internalField':
                template[insert[0]:insert[0]] = self.formatting_Field('dU', array)
            else: 
                template[insert[0]-2:insert[0]+1] = [str.encode('        type           fixedValue;\n'),
                                                     str.encode('        value           uniform (0 0 0);\n')]
                
        result =''
        for i in template:
            result += i.decode('utf-8')
        return result
                
    def merge_vectorField (self, x_interior, y_interior, x_boundary, y_boundary):
        '''Gathers vector fields described by its x(ncells,1) and y(ncells,1)
        components and creates a bidimensional array(ncells,2) as output'''
        
        if x_interior.shape != y_interior.shape: raise ValueError ('foamRW: Error merging vector field: different dimension of x and y')
        ncells, _ = x_interior.shape
        
        result_interior = np.zeros((ncells,3))
        for cell in range(ncells):
            result_interior[cell, :] = np.array([x_interior[cell,0], y_interior[cell,0], 0])
            
        result_boundary = {}
        for boundary in x_boundary.keys():
            if x_boundary[boundary].shape != y_boundary[boundary].shape: raise ValueError ('foamRW: Error merging vector field: different dimension of x and y')
            ncells, _ = x_boundary[boundary].shape
            boundary_value = np.zeros((ncells,3))
            for cell in range(ncells):
                boundary_value[cell,:] = np.array([x_boundary[boundary][cell,0], y_boundary[boundary][cell,0], 0])
            result_boundary[boundary] = boundary_value
            
        return (result_interior, result_boundary)
        
    
    def upload_fine_data(self, case_directory_fine, mesh_dict_fine):
                
        self.case_directory_fine = case_directory_fine
        self.mesh_dict_fine = mesh_dict_fine
        
        self.FoamMesh_fine = Ofpp.FoamMesh(self.case_directory_fine)
        
        self.times_origin, self.deltaT = self.get_timesteps(case_directory_fine,
                                                            self.training)
        self.times = self.times_origin[self.snap_0:self.snap_0+self.snapshots]
        
        fine_fieldNames = self.fieldNames_all
        self.data_fine = upload_data(self.case_directory_fine, self.times, fine_fieldNames, self.FoamMesh_fine)
        # self.data_mat_fine = vector_to_matrix_conversor(self.mesh_dict_fine, self.fieldNames_all, data_vec) 
        
        n_x_cells, n_y_cells, _, _, _ = get_mesh_statistics(self.mesh_dict_fine) # only for equispaced cells
        self.grid_fine = grids.Grid(
            size_x=n_x_cells , size_y=n_y_cells, 
            step=self.mesh_dict_fine['B1']['cell_size'][0])
        
        self.downsampling(self.grid_fine)
        
        return
        
    def downsampling (self, grid_fine):
        '''Downsamples numerical data from a fine mesh into a coaser one'''

        evolving_keys_defs = {k:v for k,v in self.equation.key_definitions.items() if k in self.equation.evolving_keys} 
        evolving_keys_b_defs = {k:v for k,v in self.equation.key_definitions.items() if k in self.equation.evolving_keys_b} 
       
        # Downsampling center data
        data_center_fine = {k:v for k,v in self.data_fine.items() if k in self.equation.evolving_keys}
        data_center_fine_mat = vector_to_matrix_conversor_tf(self.mesh_dict_fine, data_center_fine)
        data_downsampled = tensor_ops.regrid(data_center_fine_mat, evolving_keys_defs, grid_fine, self.grid_coarse)
        data_center_downsampled = matrix_to_vector_conversor(self.mesh_dict_coarse, self.ncells_coarse, data_downsampled)
        
        # Downsampling boundary data
        data_boundary_fine = {k:v for k,v in self.data_fine.items() if k in self.equation.evolving_keys_b}
        data_boundary_downsampled = tensor_ops.regrid_b(data_boundary_fine, evolving_keys_b_defs, grid_fine, self.grid_coarse)
        
        self.data_downsampled = dict(data_center_downsampled | data_boundary_downsampled)
        
        return
    
    def make_train_data (self,  example_time_steps=4,
                         target = {'x_velocity', 'y_velocity', 'p'}, dtype=tf.float32):
        '''Generates train_input and train_out for TF model and all the folders
        and input files needed by OF'''
        
        # remove the last several time steps of  data, as training input
        train_input = {k: v[:-example_time_steps] for k, v in self.data_downsampled.items() if k in self.equation.evolving_keys}
        fields_b = {k: v for k, v in self.data_downsampled.items() if k in self.equation.evolving_keys_b}
        train_input_b = {}
        for fk, fv in fields_b.items():
            boundary = {}
            for bk, bv in fv.items():
                if bv == None: boundary[bk] = bv
                else: boundary[bk] = bv[:-example_time_steps]
            train_input_b[fk] = boundary
        train_input = dict(train_input | train_input_b)
        
        # get number of times in training data
        batch, _= train_input[list(train_input.keys())[0]].shape

        # initial timestep of each entry of the training data
        timesteps_initial = self.times[:-example_time_steps]
        timesteps_initial = tf.convert_to_tensor(timesteps_initial, dtype=dtype)
        
        # create timesteps_to_predict of each entry of the training data
        timesteps_to_predict = [self.times[b+1:b+example_time_steps+1] for b in range (batch)]
        timesteps_to_predict = tf.convert_to_tensor(timesteps_to_predict, dtype=dtype)
        
        train_output = {}
        for field in target:
            output_list = []
            for shift in range(1, example_time_steps+1):
              # output time series, starting from each single time step
              output_slice = self.data_downsampled[field][shift:len(self.times)- example_time_steps + shift] 
              output_list.append(output_slice)

              train_output[field] = tf.stack(output_list, axis=1)  # concat along shift_time dimension, after sample dimension

        print('\n train_input shape:')  
        for k in self.fieldNames_all:
            if k == 'U':
                print('x_velocity', train_input['x_velocity'].shape)
                print('y_velocity', train_input['y_velocity'].shape)
            else:
                print(k, train_input[k].shape) 
            
        print('\n train_output shape:')
        for k,v in train_output.items():
            print(k, v.shape) 

        return train_input, train_output, timesteps_initial, timesteps_to_predict 

    def drag_data (self, example_time_steps=4, target = {'x_velocity', 'p', 'y_velocity','k', 'nut', 'omega'},
                   target_b = {'x_velocity_b', 'p_b', 'y_velocity_b','k_b', 'nut_b', 'omega_b'}):

        train_output = {}
        for field in target:
            output_list = []
            for shift in range(1, example_time_steps+1):
              # output time series, starting from each single time step
              output_slice = self.data_downsampled[field][shift:len(self.times)- example_time_steps + shift] 
              output_list.append(output_slice)
        
              train_output[field] = tf.stack(output_list, axis=1)  # concat along shift_time dimension, after sample dimension

        for field_b in target_b:
            boundary = {}
            for b, v in self.data_downsampled[field_b].items():
                output_list = []
                for shift in range(1, example_time_steps+1):
                  # output time series, starting from each single time step
                  output_slice = v[shift:len(self.times)- example_time_steps + shift] 
                  output_list.append(output_slice)
                  
                  boundary[b] = tf.stack(output_list, axis=1) 
            train_output[field_b] = boundary
            
        return train_output
    
    def faces_dict_generator(self):
        
        x_faces = []
        y_faces = []
        
        for i in range(self.FoamMesh.num_inner_face):
            points = self.FoamMesh.faces[i]
            x = np.array([self.FoamMesh.points[p][0] for p in points])
            y = np.array([self.FoamMesh.points[p][1] for p in points])
            
            if x.max() == x.min(): x_faces.append(i)
            elif y.max() == y.min(): y_faces.append(i)
            else: raise ValueError ('Non orthogonal mesh detected: it is not supported')
            
        cells = {'x_faces':np.array(x_faces), 
                 'y_faces':np.array(y_faces)} 
        
        return cells
    
    def build_indices_conversion(self, mesh_dict, ncells):
        '''Builds the indices for vector-matrix and matrix-vector conversions'''
        
        cells = tf.range(ncells)[tf.newaxis,:]
        cells_matrix = tf.squeeze(vector_to_matrix_conversor_tf(mesh_dict, {'x':cells}, dtype=tf.int32)['x'])
        bc_positions = tf.where(tf.less(cells_matrix, 0)) 
        indices_vmc = tf.tensor_scatter_nd_update(cells_matrix, bc_positions, tf.zeros(len(bc_positions), dtype=tf.int32))

        indices = tf.reshape(cells_matrix, [1,-1])
        mask = tf.equal(tf.transpose(indices), tf.range(ncells))
        indices_mvc = tf.argmax(mask, axis=0, output_type=tf.int32)

        return indices_vmc, bc_positions, indices_mvc
    
    def vmc(self, data_vector):
        '''Transforms data from vector form tf.Tensor(batch,ncells) into matrix 
        form tf.Tensor(batch, nx, ny)'''
        
        def b_vmc (data):
            temp = tf.gather(data, self.indices_vmc)
            result = tf.tensor_scatter_nd_update(temp, self.bc_positions, -99*tf.ones(len(self.bc_positions)))
            return result
    
        return tf.map_fn(b_vmc, data_vector)
        
    
    def mvc(self, data_matrix):
        '''Transforms data from matrix form tf.Tensor(batch, nx, ny) into vector 
        form tf.Tensor(batch,ncells)'''

        def b_mvc (data):
            flat = tf.reshape(data, [-1])
            result = tf.gather(flat, self.indices_mvc)
            return result
        
        return tf.map_fn(b_mvc, data_matrix)
    
    
    def plotter(self, time, magnitude, mode):
        '''Plots the data of the square column case in matrix form'''
        
        if mode == 'coarse': 
            data = self.data
            grid_x, grid_y = self.grid_coarse.get_mesh()
            mesh = self.mesh_dict_coarse
        elif mode == 'fine':
            data = self.data_fine
            grid_x, grid_y = self.grid_fine.get_mesh()
            mesh = self.mesh_dict_fine
        elif mode == 'downsampled':
            data = self.data_downsampled
            grid_x, grid_y = self.grid_coarse.get_mesh()
            mesh = self.mesh_dict_coarse
        else:
            raise ValueError('foamRW: mode not recognized')
            
        t = self.times.index(time)
        snapshot_dict = {magnitude: tf.expand_dims(data[magnitude][t,:], axis=0)}
        field_dict = vector_to_matrix_conversor(mesh, snapshot_dict)
        field = tf.squeeze(field_dict[magnitude]).numpy().transpose()[:,::-1]
        fig = plt.figure(figsize=(6, 4))
        plt.pcolor(grid_x, grid_y, field, cmap='jet', vmin=-field.max(), vmax=field.max())
        plt.title(f'Velocity squared column t={self.times[t]}')
        plt.xlabel('x [m]')
        plt.xticks(np.linspace(0,15,4))
        plt.ylabel('y [m]')
        plt.yticks(np.linspace(0,5,3))
        plt.colorbar(orientation='horizontal', pad=0.3, label='U m/s')
        x = [2, 3, 3, 2, 2]
        y = [2, 2, 3, 3, 2]
        # plt.fill(x, y, color='grey')
        fig.tight_layout()
        plt.show()
        return fig
    
    def plotter_vector (self, vector_data, limits = (None, None), magnitude = None):
        '''Plots the data of the square column case in vector form'''
        
        grid_x, grid_y = self.grid_coarse.get_mesh()
        mesh = self.mesh_dict_coarse
        field_dict = vector_to_matrix_conversor(mesh, {'field':tf.expand_dims(vector_data,axis=0)})
        
        field = tf.squeeze(field_dict['field']).numpy().transpose()[:,::-1]
        
        if limits[0] == None: lim1 = tf.reduce_min(vector_data).numpy()
        else: lim1 = limits[0]
        if limits[1] == None: lim2 = tf.reduce_max(vector_data).numpy()
        else: lim2 = limits[1]        
        fig = plt.figure(figsize=(6, 4))
        plt.pcolor(grid_x, grid_y, field, cmap='jet', vmin=lim1, vmax=lim2)
        if magnitude == None:
            plt.title(f'Field at squared column')
        else:
            # plt.title(f'{magnitude} at squared column')
            plt.title(f'{magnitude}')
        plt.xlabel('x [m]')
        plt.xticks(np.linspace(0,15,4))
        plt.ylabel('y [m]')
        plt.yticks(np.linspace(0,5,3))
        plt.colorbar(orientation='horizontal', pad=0.3, label='U m/s')
        x = [2, 3, 3, 2, 2]
        y = [2, 2, 3, 3, 2]
        # plt.fill(x, y, color='grey')
        fig.tight_layout()
        plt.show()
        return fig
    
    def plotter_video (self, vector_data, limits = (None, None), magnitude = None, size=25, legend = True, colormap = 'jet'):
        '''Plots the data of the square column case in vector form for video generation'''
        
        grid_x, grid_y = self.grid_coarse.get_mesh()
        mesh = self.mesh_dict_coarse
        field_dict = vector_to_matrix_conversor(mesh, {'field':tf.expand_dims(vector_data,axis=0)})

        field = tf.squeeze(field_dict['field']).numpy().transpose()[:,::-1]
        if limits[0] == None: lim1 = tf.reduce_min(vector_data).numpy()
        else: lim1 = limits[0]
        if limits[1] == None: lim2 = tf.reduce_max(vector_data).numpy()
        else: lim2 = limits[1]        
        
        ax = plt.pcolor(grid_x, grid_y, field, cmap=colormap, vmin=lim1, vmax=lim2)
        if magnitude == None:
            text = 'u [m/s]'
        else:
            text = magnitude
            
        plt.xlabel('x [m]', fontsize=size)
        plt.xticks(np.linspace(0,15,4), fontsize=size)
        plt.ylabel('y [m]', fontsize=size)
        plt.yticks(np.linspace(0,5,3), fontsize=size)
        if legend == True:
            a = plt.colorbar(orientation='vertical')
            a.set_label(label=text,size=size)
            a.ax.tick_params(labelsize=size)
        x = [2, 3, 3, 2, 2]
        y = [2, 2, 3, 3, 2]
        plt.fill(x, y, color='grey')

        return ax
    
    def prepare_data(self, vector_data):
        '''Transforms vector data into matrix form for video generation'''
        
        mesh = self.mesh_dict_coarse
        field_dict = vector_to_matrix_conversor(mesh, {'field':tf.expand_dims(vector_data,axis=0)})
        field = tf.squeeze(field_dict['field']).numpy().transpose()
        return field
    
    def patches_index (self, stencil_size):
        '''Returns a list with the indexes of the vector form for each patch'''
        size_x, size_y = stencil_size
        stencil_left = size_x//2 
        stencil_right = size_x//2 + size_x%2 
        stencil_down = size_y//2 
        stencil_up = size_y - stencil_down - 1
        
        faces_dict = self.faces_dict_generator()
        x_faces = faces_dict['x_faces']
        y_faces = faces_dict['y_faces']

        cells_position = vector_to_matrix_conversor(self.mesh_dict_coarse, {'cells': np.expand_dims(np.arange(self.ncells_coarse), axis=0)})
        cell_pos_proto = tf.make_tensor_proto(tf.squeeze(cells_position['cells']))
        cell_pos_np = tf.make_ndarray(cell_pos_proto)
        # Padding for x component
        self.cell_position_x = np.pad(cell_pos_np,((stencil_up, stencil_down),(stencil_left-1, stencil_right-1)), 'constant', constant_values=((-99,-99),(-99,-99)))
        if stencil_right > 1:
            self.cell_position_x_b = np.pad(cell_pos_np,((0, 0),(0, 1)), 'edge')
            self.cell_position_x_b = np.pad(self.cell_position_x_b,((stencil_up, stencil_down),(stencil_left-1, 0)), 'constant', constant_values=((-99,-99),(-99,-99))) 
            self.cell_position_x_b = np.pad(self.cell_position_x_b,((0, 0),(0, stencil_right-2)), 'edge')
        else : 
            self.cell_position_x_b = np.pad(cell_pos_np,((stencil_up, stencil_down),(stencil_left-1, stencil_right-1)), 'constant', constant_values=((-99,-99),(-99,-99)))
        # Padding for y component
        self.cell_position_y = np.rot90(np.pad(cell_pos_np,((stencil_left-1, stencil_right-1),(stencil_down, stencil_up)), 'constant', constant_values=((-99,-99),(-99,-99))),1) 
        if stencil_up >= 1:
            self.cell_position_y_b = np.pad(cell_pos_np,((0, 0),(0, 1)), 'edge')
            self.cell_position_y_b = np.pad(self.cell_position_y_b,((stencil_left-1, stencil_right-1),(stencil_down, 0)), 'constant', constant_values=((-99,-99),(-99,-99))) 
            self.cell_position_y_b = np.pad(self.cell_position_y_b,((0, 0),(0, stencil_up-1)), 'edge')
        else : 
            self.cell_position_y_b = np.pad(cell_pos_np,((stencil_left-1, stencil_right-1),(stencil_down, stencil_up)), 'constant', constant_values=((-99,-99),(-99,-99)))
        #counterclokwise flip
        self.cell_position_y_b = np.rot90(self.cell_position_y_b,1)
        
        result = []
        for face in range(self.FoamMesh.num_inner_face):
            if face in x_faces: 
                cell_position = self.cell_position_x
                cell_position_b = self.cell_position_x_b
                idx = [0, 1, 1, 1]
            elif face in y_faces: 
                cell_position = self.cell_position_y
                cell_position_b = self.cell_position_y_b
                idx = [0, 1, 0, 0]
            else: raise ValueError('foamRW: Not recognised face')
            
            owner = self.FoamMesh.owner[face]
            
            index = np.argwhere(cell_position == np.array(owner))
            if len(index) != 1: raise ValueError('foamRW: Not unique coincidence looking for the stencil')
            id_row, id_col = index[0]

            stencil = cell_position_b[id_row-stencil_up+idx[0]:id_row+stencil_down+idx[1],id_col-stencil_left+idx[2]:id_col+stencil_right+idx[3]]
            assert stencil.shape == (stencil_size[1], stencil_size[0]) #DeCo MOD6
            result.append(tf.convert_to_tensor(stencil, dtype = tf.int32))
    
        return result    
  
    def boundary_cells_index_x (self):
        cells_position = vector_to_matrix_conversor(self.mesh_dict_coarse, {'cells': np.expand_dims(tf.range(self.ncells_coarse), axis=0)})
        result_cells = cells_position['cells'][0,:,:]
        result_cells = tf.cast(result_cells, tf.int32)
        result_cells_np = result_cells.numpy()
        result_cells_np[result_cells_np < 0] = 0
        ny, nx = result_cells.shape
        result_faces = np.zeros((ny,nx-1))
        for i in range(ny):
            for j in range(nx-1):
                if result_cells_np[i,j] == 0 or result_cells_np[i,j+1] == 0: result_faces[i,j] = 0
                else:
                    candidate = list(set(self.FoamMesh.cell_faces[result_cells_np[i,j]]) & set(self.FoamMesh.cell_faces[result_cells_np[i,j+1]]))
                    if len(candidate) == 1: result_faces[i,j] = candidate[0]
                    else: raise ValueError('foamRW: Not unique coincidence looking for the intersection of faces')
        return tf.convert_to_tensor(result_cells_np, dtype=tf.int32), tf.convert_to_tensor(result_faces, dtype=tf.int32)

    def boundary_cells_index_y (self):
        cells_position = vector_to_matrix_conversor(self.mesh_dict_coarse, {'cells': np.expand_dims(tf.range(self.ncells_coarse), axis=0)})
        result_cells = cells_position['cells'][0,:,:]
        result_cells = tf.cast(result_cells, tf.int32)
        result_cells_np = result_cells.numpy()
        result_cells_np[result_cells_np < 0] = 0
        ny, nx = result_cells.shape
        result_faces = np.zeros((ny-1,nx))
        for i in range(nx):
            for j in range(ny-1):
                if result_cells_np[j,i] == 0 or result_cells_np[j+1,i] == 0: result_faces[j,i] = 0
                else:
                    candidate = list(set(self.FoamMesh.cell_faces[result_cells_np[j,i]]) & set(self.FoamMesh.cell_faces[result_cells_np[j+1,i]]))
                    if len(candidate) == 1: result_faces[j,i] = candidate[0]
                    else: raise ValueError('foamRW: Not unique coincidence looking for the intersection of faces')
        return tf.convert_to_tensor(result_cells_np, dtype=tf.int32), tf.convert_to_tensor(result_faces, dtype=tf.int32)

#TODO: Check if some of these last functions are unused and can be deleted
    def build_continuity_arrays(self):
        '''Creates the arrays with the faces of each cell to compute continuity'''
        
        faces_output = {}
        leftx = []
        rightx = []
        topy = []
        bottomy = []
        
        faces_np = np.array([sorted(row) for row in self.FoamMesh.faces])
        for i in range(self.ncells_coarse):
            faces = self.FoamMesh.cell_faces[i]
            points = [self.FoamMesh.faces[f] for f in faces]
            points = np.unique(np.array([x for xs in points for x in xs]))
            x = np.array([self.FoamMesh.points[p][0] for p in points])
            y = np.array([self.FoamMesh.points[p][1] for p in points])
            cell_points = [[x.min(),y.min()],[x.max(), y.min()],
                           [x.max(),y.max()], [x.min(), y.max()]]
            pos = []
            for p_x, p_y in cell_points:
                pos.append([np.argmax(np.all(self.FoamMesh.points == [p_x, p_y, z], axis=1)) for z in [-0.5, 0.5]])
                
            leftx.append(np.argmax(np.all(faces_np==sorted(pos[0]+pos[3]), axis=1)))
            rightx.append(np.argmax(np.all(faces_np==sorted(pos[1]+pos[2]), axis=1)))
            topy.append(np.argmax(np.all(faces_np==sorted(pos[3]+pos[2]), axis=1)))
            bottomy.append(np.argmax(np.all(faces_np==sorted(pos[0]+pos[1]), axis=1)))
        
        faces_output['left'] = tf.convert_to_tensor(leftx)
        faces_output['right'] = tf.convert_to_tensor(rightx)
        faces_output['top'] = tf.convert_to_tensor(topy)
        faces_output['bottom'] = tf.convert_to_tensor(bottomy)
        
        outlet_correction = {}
        start = self.FoamMesh.boundary[b'outlet'].start
        faces_outlet = np.arange(start, start+self.FoamMesh.boundary[b'outlet'].num)
        position_faces_outlet = np.where(np.isin(faces_output['right'], faces_outlet))[0]
        params = [leftx[i] for i in position_faces_outlet]
        outlet_correction['indices'] = tf.convert_to_tensor(position_faces_outlet, dtype=tf.int32)
        outlet_correction['params'] = tf.convert_to_tensor(params)
        
        num_faces = {}
        num_faces[b'outlet'] = self.FoamMesh.boundary[b'outlet'].start
        num_faces[b'inlet'] = self.FoamMesh.boundary[b'inlet'].start
        num_faces[b'walls'] = self.FoamMesh.boundary[b'upperwall'].start
        
        return faces_output, outlet_correction, num_faces

    def info_boundary_patches(self, faces_dict, stencil_size):
        '''Returns information about the faces which patch touches the boundary'''
        
        # Identify patches touching boundary
        patches = tf.stack(self.patches_index(stencil_size), axis=0)
        faces_boundary = tf.where(patches==-99)[:,0]
        faces_boundary, _ = tf.unique(faces_boundary)
        set_faces_boundary = set(faces_boundary.numpy())
        # Separate patches in function of the orientation
        faces_x = list(set_faces_boundary.intersection(faces_dict['x_faces']))
        faces_y = list(set_faces_boundary.intersection(faces_dict['y_faces']))
        faces_x.sort()
        faces_y.sort()
        result_x = {}
        result_y = {}
        # Gather info of patches
        result_x['faces'] = faces_x
        result_y['faces'] = faces_y
        result_x['positions'] = [np.searchsorted(faces_dict['x_faces'], value) for value in faces_x]
        result_y['positions'] = [np.searchsorted(faces_dict['y_faces'], value) for value in faces_y]
        result_x['owners'] = [self.FoamMesh.owner[i] for i in faces_x]
        result_x['neighbours'] = [self.FoamMesh.neighbour[i] for i in faces_x]
        result_y['owners'] = [self.FoamMesh.owner[i] for i in faces_y]
        result_y['neighbours'] = [self.FoamMesh.neighbour[i] for i in faces_y]
        result = {}
        result['x_velocity_edge'] = result_x
        result['y_velocity_edge'] = result_y
        
        return result
    
    
# import mesh_definition as mesh
# column_case0 = foamRW('../mycase_coarse_full', '', mesh.coarse_mesh_dict(), snapshots=5, snap_0 = 250, training=True)
# column_case0.upload_fine_data('../mycase_fine_full', mesh.fine_mesh_dict())
# train_in, train_out, timesteps_initial, timesteps_to_predict = column_case0.make_train_data(example_time_steps=2)
# # column_case1.plotter(0.08,'x_velocity', mode = 'coarse')
# # # # column_case.plotter(0.08,'x_velocity', mode = 'fine')
# # # # column_case.plotter(0.08,'x_velocity', mode = 'downsampled')
# column_case1 = foamRW('../mycase_coarse_full', '', mesh.coarse_mesh_dict(), snapshots=1, snap_0 = 1, training=True)
# dU = column_case1.get_derivative_template(np.random.rand(4736,3))
# new_fields = {}
# new_fields.update({'omega_b':{k:tf.expand_dims(tf.squeeze(v), axis=-1).numpy() for k,v in column_case1.data['omega_b'].items()}})
# new_fields.update({'omega': tf.expand_dims(tf.squeeze(column_case1.data['omega']), axis=-1).numpy()})
# u_x = tf.expand_dims(tf.squeeze(column_case1.data['x_velocity']), axis=-1).numpy()
# u_y = tf.expand_dims(tf.squeeze(column_case1.data['y_velocity']), axis=-1).numpy()
# u_x_b = {k:tf.expand_dims(tf.squeeze(v), axis=-1).numpy() for k,v in column_case1.data['x_velocity_b'].items()}
# u_y_b = {k:tf.expand_dims(tf.squeeze(v), axis=-1).numpy() for k,v in column_case1.data['y_velocity_b'].items()}
# U, U_b = column_case1.merge_vectorField(u_x, u_y, u_x_b, u_y_b)
# new_fields.update({'U': U, 'U_b': U_b})
# new_fields['Ufaces'] = np.random.rand(9296,3)
# new_fields['phi'] = np.random.rand(9296,1)
# new_fields['phi_b'] = {'outlet': np.random.rand(40,1)}

# new_state_str = column_case1.modify_templates(new_fields, '0', mode = 'string')
