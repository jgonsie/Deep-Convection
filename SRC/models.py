# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:40:37 2022

@author: jesusglezs97
"""

"""Models evaluate spatial state derivatives.

Models encapsulate the machinery that provides all spatial state derivatives
to the governing equation. They can employ different techniques to produce
state derivatives, such as finite difference methods or neural networks.
"""
from typing import (Any, Dict, List, Optional, Mapping, Set, TypeVar, Union)
import numpy as np
import tensorflow as tf 
import ctypes
import math

from SRC import equations
from SRC import polynomials
from SRC import states
from SRC import tensor_ops

T = TypeVar('T')

def numparize(x: tf.Tensor):
    return x.numpy()

def sorted_values(x: Dict[Any, T]) -> List[T]:
  """Returns the sorted values of a dictionary."""
  return [x[k] for k in sorted(x)]


def sorted_learned_keys(a: Set) -> List:
  """Returns the sorted learned keys."""
  x, y, others = [],[],[]
  for key in a:
      if key[-1] == 'x':
          x.append(key)
      elif key[-1] == 'y':
          y.append(key)
      else: others.append(key)
    
  return sorted(x)+sorted(y)+sorted(others)


def stack_dict(state: Dict[Any, tf.Tensor]) -> tf.Tensor:
  """Stack a dict of tensors along its last axis."""

  return tf.stack(sorted_values(state), axis=-1) #Convert a dict of tensors into a tensor along the last dimension 


def stack_dict_FCDnet(state: Dict[Any, tf.Tensor]) -> tf.Tensor:
  """Stack a dict of tensors along its last axis."""
  
  return tf.concat(sorted_values(state), axis=-1) #Convert a dict of tensors into a tensor along the last dimension 


def build_stencils(
    key: states.StateDefinition,
    parent: states.StateDefinition,
    stencil_size: (int,int),
    grid_step: float
) -> List[np.ndarray]:
  """Create stencils for use with learned coefficients."""
  stencils = []
  axis = key.offset[1]
  if axis == 1: stencil_size = tuple(reversed(stencil_size)) #correction for non-squared stencils

  for parent_offset, key_offset, size in zip(parent.offset, key.offset, stencil_size):
    # same position parent-key (parent.offset == key.offset):
        # - even stencil: not zero-centered stencil
        # - odd stencil: zero-centered stencil
    # different position parent-key (parent.offset != key.offset):
        # - even stencil: zero-centered stencil
        # - odd stencil: not zero-centered stencil

    # examples:
    # stencil_size=5 -> [-2, -1, 0, 1, 2] zero-centered
    # stencil_size=4 -> [-2, -1, 0, 1] not zero-centered
    int_range = np.arange(size) - size // 2 - (size % 2 * axis * key_offset) # ( ) term is the correction to obtain a coherent transformation

    stencil = grid_step * (0.5 * abs(key_offset - parent_offset) + int_range)

    stencils.append(stencil)

  return stencils #DeCo MOD4
    
ConstraintLayer = Union[
    polynomials.PolynomialAccuracy, polynomials.PolynomialBias] #type PolynomialsAccuracy or PolynomialBias
  
class extract_patches_from_vector (tf.keras.layers.Layer): 
    def __init__(self, stencil_size, case):
      # # Convert list to Tensor
      indexes_list = tf.convert_to_tensor(case.patches_index(stencil_size), dtype=tf.int32) #(9296,Nx,Ny)
      # Find boundary cells filtering indexes less than zero
      condition = tf.less(indexes_list, 0) #(9296,Nx,Ny)
      positions = tf.where(condition) #(6232,3)
      self.positions = tf.cast(positions, dtype = tf.int32)
      self.positives, _ = self.positions.shape
      # Change index of boundary cells to zero 
      self.indexes = tf.tensor_scatter_nd_update(indexes_list, self.positions, tf.zeros(len(self.positions), dtype=tf.int32))
      
      super().__init__(name='vectorToPatch', trainable = False)
  
    def correct_boundary_cells(self, patches_uncorrected):
        patches_cor = tf.tensor_scatter_nd_update(patches_uncorrected, self.positions, tf.zeros(self.positives))
        
        return patches_cor
    
    def call(self, inputs): #inputs.shape = (b, ncells)
      #apply gather function to get the cell values
      patches_wbc = tf.gather(inputs, self.indexes, axis=1)  #(b, nfaces, Nx, Ny)
      #correct positions of boundary cells to zero
      patches = tf.vectorized_map(self.correct_boundary_cells, patches_wbc) #(b, nfaces, Nx, Ny)

      return patches
    
class VaryingCoefficientsLayer(tf.keras.layers.Layer):
  """Layer representing varying coefficients for a single derivative. PAL"""

  def __init__(
      self,
      constraint_layer: ConstraintLayer,
      stencils: List[np.ndarray],
      shifts: List[int],
      boundary_patches : Dict[Any, np.ndarray],
      bounding_perc: Any,
      input_key: Optional[str] = None,
      output_key: Optional[str] = None,
  ):
    self.constraint_layer = constraint_layer
    self.stencils = stencils
    self.shifts = shifts
    self.bounding_perc = bounding_perc
    self.input_key = input_key
    self.output_key = output_key
    self.kernel_size = constraint_layer.input_size
    self.positions = tf.convert_to_tensor(boundary_patches['positions'], dtype=tf.int32)
    self.owners = boundary_patches['owners']
    self.neighbours = boundary_patches['neighbours']

    super().__init__(trainable=False)
     
  def bounding(self, ufaces, ucenters):
      # Extract boundary faces from ufaces
      ufaces_b = tf.gather(ufaces, self.positions, axis=-1)
      # Compute upwind for boundary_faces
      owner = tf.gather(ucenters, self.owners, axis=-1)[:,:,tf.newaxis]
      neighbour = tf.gather(ucenters, self.neighbours, axis=-1)[:,:,tf.newaxis]
      side_cells = tf.concat([owner,neighbour], axis = -1)
      side_max = tf.reduce_max(side_cells, axis=-1) #(batch,Nfaces)
      side_min = tf.reduce_min(side_cells, axis=-1) #(batch,Nfaces)
      flux_cd = tf.reduce_mean(side_cells, axis=-1)
      upwind = tf.where(flux_cd >= 0., tf.squeeze(owner), tf.squeeze(neighbour))
      # Compute_bounding range
      # perc = 0.3
      upper_lim = side_max + self.bounding_perc * tf.abs(side_max)
      lower_lim = side_min - self.bounding_perc * tf.abs(side_min)
      # Compute the filter and the ufaces_bounded for boundary_faces
      ufaces_valid = tf.logical_and(tf.greater_equal(ufaces_b, lower_lim), tf.less_equal(ufaces_b, upper_lim))
      ufaces_bounded = tf.where(ufaces_valid, ufaces_b, upwind)
      # Insert back ufaces_bounded into complete ufaces
      result = tf.tensor_scatter_nd_update(tf.transpose(ufaces), self.positions[:,tf.newaxis], tf.transpose(ufaces_bounded)) #[nFaces, b]
      
      return tf.transpose(result)
      
      
  def call(self, inputs):
    (kernel, source, ucenters) = inputs #(b, nFaces, nStencil-1) (b, nFaces, Nx, Ny) (b,nCells)
    coefficients = self.constraint_layer(kernel) #(b, nFaces, nStencil)    
    b, nfaces, _, _ = source.shape
    patches = tf.reshape(source, [b,nfaces,-1])
    ufaces = tf.einsum('bxs,bxs->bx', coefficients, patches) #(batch,nFaces,nStencil)->(batch,nFaces)
    ufaces_bounded = self.bounding(ufaces, ucenters)
    
    return ufaces_bounded
        
def build_output_layers(
    equation, 
    grid, 
    learned_keys, 
    boundary_patches, 
    bounding_perc = 0.0, 
    stencil_size=(4,4),
    initial_accuracy_order=1,
    constrained_accuracy_order=1,
    layer_cls=VaryingCoefficientsLayer,
) -> Dict[str, ConstraintLayer]:
  """Build a map of output layers for spatial derivative models."""

  layers = {}
  modeled = set()

  for key in learned_keys:
    parent = equation.find_base_key(key)
    key_def = equation.key_definitions[key]
    modeled.add(key_def)
    parent_def = equation.key_definitions[parent]
    boundary_patches_def = boundary_patches[key]
    stencils = build_stencils(key_def, parent_def, stencil_size, grid.step)
    shifts = [k - p for p, k in zip(parent_def.offset, key_def.offset)]
    
    constraint_layer = polynomials.constraint_layer(
        stencils, equation.METHOD, key_def.derivative_orders[:2],
        constrained_accuracy_order, initial_accuracy_order, grid.step,
    )
    
    layers[key] = layer_cls(
        constraint_layer, stencils, shifts, boundary_patches_def, bounding_perc,
        input_key=parent, output_key=key_def)

  return layers

def _normalize_velocity(u_x, u_y, component):
    '''Normalizes velocity vector u_i_norm = u_i/max(sqrt(u_x²+u_y²))'''
    u_v = tf.math.sqrt(tf.math.square(u_x)+tf.math.square(u_y))
    u_max = tf.reduce_max(u_v, axis=-1, keepdims=True)
    if component == 'x_velocity':
        result = u_x/(u_max + 1e-8)
    elif component == 'y_velocity':
        result = u_y/(u_max + 1e-8)
        
    return result

def _denormalize_velocity(u_x_norm, u_y_norm, u_x_ori, u_y_ori, component):
    u_v = tf.math.sqrt(tf.math.square(u_x_ori)+tf.math.square(u_y_ori))
    u_max = tf.reduce_max(u_v, axis=-1, keepdims=True)
    if component == 'x_velocity':
        result = u_x_norm * u_max
    elif component == 'y_velocity':
        result = u_y_norm * u_max
        
    return result
    
def _normalize_velocity2(u_i):
    '''Normalizes velocity vector u_i_norm = u_i/max(u_i)'''
    u_max = tf.reduce_max(u_i, axis=-1, keepdims=True)
    result = u_i/(u_max + 1e-8)
        
    return result

def _normalize_pressure(p):
    '''Extracts the mean and normalizes the pressure between 0 and max'''
    
    p_atm = 101325
    p_red = p-p_atm
    
    return p_red

def _normalize_data(array, range_val=(-1,1), epsilon=1e-8):
    ''' Normalizes data of shape (batch, ncells) between a given range'''
  
    a = range_val[0]
    b = range_val[1]
    if a >= b: raise ValueError (f'Wrong range for normalization: [{a},{b}]')
    
    array_min = tf.reduce_min(array, axis=-1, keepdims=True)
    array_max = tf.reduce_max(array, axis=-1, keepdims=True)
    result = (b-a) * (array - array_min) / (array_max - array_min + epsilon) + a #epsilon to avoid dividing by zero (y_velocity at the beginning)
    # mean, variance = tf.nn.moments(padded, axes=[0,1])
    # result[key] = (padded - mean) / tf.sqrt(variance + epsilon) # epsilon to avoid dividing by zero
    return result


def _denormalize_data(array, array_origin, range_val=(-1,1)):
    ''' Denormalizes data of shape between a given range'''
  
    a = range_val[0]
    b = range_val[1]
    if a >= b: raise ValueError (f'Wrong range for denormalization: [{a},{b}]')
    
    array_min = tf.reduce_min(array, axis=-1, keepdims=True)
    array_max = tf.reduce_max(array, axis=-1, keepdims=True)
    result = (array - a) * (array_max - array_min) / (b-a) + array_min
    
    return result
    
def createNets(stencil_size, target, num_outputs, num_layers_glob=2, 
               num_layers_loc=4, neurons_glob=None, neurons_loc=None, #<-------------------------XXX
               activation='tanh', **kwargs):
    
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42) #TODO: I should try if gorot initialization works better, is it random?
    (Nx, Ny) = stencil_size
    # Ni = len(target)
    Ni = 5 #<------------------------------------------------------------------XXX
    input_neurons = Nx*Ny*Ni
    
    if neurons_loc == None and neurons_glob == None:
        dif = input_neurons-num_outputs
        s, res = divmod(dif, 2)
        sl, resl = divmod(s+res, num_layers_loc)
        neurons_loc = [input_neurons-sl-resl]
        neurons_loc += [neurons_loc[0]-i*sl for i in range(1,num_layers_loc)]
        sg, resg = divmod(s, num_layers_glob)
        neurons_glob = [neurons_loc[-1]-sg-resg]
        neurons_glob += [neurons_glob[0]-i*sg for i in range(1,num_layers_glob)]

    model_l1 = tf.keras.Sequential()
    model_l2 = tf.keras.Sequential()
    input_size_l = (None, None, input_neurons)
    model_l1.add(tf.keras.layers.Input(shape=input_size_l))
    model_l2.add(tf.keras.layers.Input(shape=input_size_l)) 
    for i in range(num_layers_loc):
        layer_1 = tf.keras.layers.Dense(neurons_loc[i], activation=activation,
                kernel_initializer=initializer, dtype=tf.float32, **kwargs)
        model_l1.add(layer_1)
        layer_2 = tf.keras.layers.Dense(neurons_loc[i], activation=activation,
                kernel_initializer=initializer, dtype=tf.float32, **kwargs)
        model_l2.add(layer_2)
        
    model_g = tf.keras.Sequential()
    input_size_g = (None, None, neurons_loc[-1])
    model_g.add(tf.keras.layers.Input(shape=input_size_g)) 
    for i in range(num_layers_glob):
        layer = tf.keras.layers.Dense(neurons_glob[i], activation=activation,
                kernel_initializer=initializer, dtype=tf.float32, **kwargs)
        model_g.add(layer)
        
    print(f'Layers local: {num_layers_loc}; Neurons local: {neurons_loc}')
    print(f'Layers global: {num_layers_glob}; Neurons global: {neurons_glob}')
    return model_l1, model_l2, model_g


class DeepConvectionNet(tf.keras.Model):
    """Fully connected dense layers taking patches as inputs."""
    def __init__(self, num_outputs, stencil_size, target, extract_patches_layer,
                 faces_dict, output_layers, num_layers_glob, num_layers_loc,
                 case, **kwargs):
        super().__init__()
        
        # Locals and global nets
        self.local_x, self.local_y, self.global_xy = createNets(stencil_size, target, num_outputs, **kwargs)
        # Extract patches layer
        self.extract_patches_from_vector = extract_patches_layer
        
        self.faces_axis = tf.nest.map_structure(lambda x: tf.constant(x, dtype=tf.int32), faces_dict)
        
        self.output_layers = output_layers
        
        self.case = case
        
    def compute_r_x(self, data_vector, u_inlet=1):
        data_matrix_or = self.case.vmc(data_vector)
        data_matrix = tf.where(data_matrix_or == -99, tf.zeros_like(data_matrix_or), data_matrix_or)
        data_c = data_matrix
        shape = data_matrix.shape.as_list() #(b,y,x)
        first_column = u_inlet * tf.ones(shape[:2]+[1], dtype=data_matrix.dtype)
        last_column = data_matrix[:,:,-1]
        data_d = tf.concat([data_matrix, tf.expand_dims(last_column,axis=2)], axis=2)[:,:,1:]
        data_u = tf.concat([first_column, data_matrix], axis=2)[:,:,:-1]
        r = (data_c-data_u)/(data_d-data_c+1e-8)
        r = tf.where(r >= 2, 2*tf.ones_like(r), r)
        r = tf.where(r < 0, tf.zeros_like(r), r)
        r = self.case.mvc(r)
        
        return r

    def compute_r_y(self, data_vector):
        data_matrix_or = self.case.vmc(data_vector)
        data_matrix = tf.where(data_matrix_or == -99, tf.zeros_like(data_matrix_or), data_matrix_or)
        data_c = data_matrix
        shape = data_matrix.shape.as_list() #(b,y,x)
        zero_row = tf.zeros(shape[0:1]+[1]+shape[2:3], dtype=data_matrix.dtype)
        data_d = tf.concat([zero_row, data_matrix], axis=1)[:,:-1,:]
        data_u = tf.concat([data_matrix, zero_row], axis=1)[:,1:,:]
        r = (data_c-data_u)/(data_d-data_c+1e-8)
        r = tf.where(r >= 2, 2*tf.ones_like(r), r)
        r = tf.where(r < 0, tf.zeros_like(r), r)
        r = self.case.mvc(r)
        
        return r
    
    def call(self, inputs):  
        
        patches = {}
        patches_origin= {}
        
        for key,value in inputs.items():
            
            if key == 'x_velocity' or key == 'y_velocity': 
                field_norm = _normalize_velocity2(value)
            elif key == 'p': 
                field_norm = _normalize_pressure(value) #(b,Ncells)
            else:
                raise KeyError(f'Not recognised field: {key}')
                
            patches[key]= self.extract_patches_from_vector(field_norm) #(b,Nfaces,Nx,Ny)

            if key == 'x_velocity' or key == 'y_velocity':
                patches_origin[key] = self.extract_patches_from_vector(value) #(b,Nfaces,Nx,Ny)
          
        # Compute r from velocity <--------------------------------------------XXX
        
        r_x = self.compute_r_x(inputs['x_velocity'])    
        r_y = self.compute_r_y(inputs['y_velocity']) 
        
        patches['r_x'] = self.extract_patches_from_vector(r_x)
        patches['r_y'] = self.extract_patches_from_vector(r_y)

        # Stack patches
        input_tensor = stack_dict(patches) #(b, nFaces, Nx, Ny, nFields)
        shape = input_tensor.shape.as_list()

        input_tensor = tf.reshape(input_tensor, shape[:2]+[-1])
        
        # Divide patches in function of the orientation of the face
        input_tensor_x = tf.gather(input_tensor, self.faces_axis['x_faces'], axis=1)
        input_tensor_y = tf.gather(input_tensor, self.faces_axis['y_faces'], axis=1)

        # Send each group to correspondent local net
        output_loc_x = self.local_x(input_tensor_x)
        output_loc_y = self.local_y(input_tensor_y)
        
        # Send to global net
        global_x = self.global_xy(output_loc_x) #(b, nFaces_x, 2*[Nx*Ny-1])
        global_y = self.global_xy(output_loc_y) #(b, nFaces_y, 2*[Nx*Ny-1])
        
        # Compute remaining coefficients
        size_splits = [self.output_layers[key].kernel_size for key in self.output_layers]
        heads_x = tf.split(global_x, size_splits, axis=-1)
        source_xx = tf.gather(patches_origin['x_velocity'], self.faces_axis['x_faces'], axis=1)
        result_xx = self.output_layers['x_velocity_edge']([heads_x[0], source_xx, inputs['x_velocity']]) #(b, nFaces_x)
        source_xy = tf.gather(patches_origin['y_velocity'], self.faces_axis['x_faces'], axis=1)
        result_xy = self.output_layers['x_velocity_edge']([heads_x[1], source_xy, inputs['y_velocity']]) #(b, nFaces_x)
        
        heads_y = tf.split(global_y, size_splits, axis=-1)
        source_yx = tf.gather(patches_origin['x_velocity'], self.faces_axis['y_faces'], axis=1)
        result_yx = self.output_layers['y_velocity_edge']([heads_y[0], source_yx, inputs['x_velocity']]) #(b, nFaces_y)
        source_yy = tf.gather(patches_origin['y_velocity'], self.faces_axis['y_faces'], axis=1)
        result_yy = self.output_layers['y_velocity_edge']([heads_y[1], source_yy, inputs['y_velocity']]) #(b, nFaces_y)
        
        # Join output face velocities
        indices_x = self.faces_axis['x_faces'][:,tf.newaxis]
        indices_y = self.faces_axis['y_faces'][:,tf.newaxis]
        result = {}
        
        result_xx_T = tf.transpose(result_xx, perm=(1,0))
        result_yx_T = tf.transpose(result_yx, perm=(1,0))
        scatter_x = tf.scatter_nd(indices_x, result_xx_T, shape[1::-1]) #[nFaces, b]
        scatter_xy = tf.tensor_scatter_nd_update(scatter_x, indices_y, result_yx_T)
        result['x_velocity_edge'] = tf.transpose(scatter_xy, perm=(1,0))
        
        result_xy_T = tf.transpose(result_xy, perm=(1,0))
        result_yy_T = tf.transpose(result_yy, perm=(1,0))
        scatter_x = tf.scatter_nd(indices_x, result_xy_T, shape[1::-1]) #[nFaces, b]
        scatter_xy = tf.tensor_scatter_nd_update(scatter_x, indices_y, result_yy_T)
        result['y_velocity_edge'] = tf.transpose(scatter_xy, perm=(1,0))
        
        return result
    
    
class ForwardProblemModel(tf.keras.layers.Layer):
    """Forward problem calling OpenFOAM"""
    
    def __init__(self, equation, case, solver, solver_AD, libCaller):
        self.evolving_keys = sorted(equation.evolving_keys)
        self.evolving_keys.append('phi')
        self.evolving_keys_b = sorted(equation.evolving_keys_b)
        self.evolving_keys_b.append('phi_b')
        self.case = case
        self.ncells = case.ncells_coarse
        self.nfaces = case.FoamMesh.num_inner_face
        self.boundaries = {str(-1*(v.id+10)): {'nfaces': v.num, 'name': k.decode('utf-8')} for k,v in case.FoamMesh.boundary.items() if k not in {b'defaultFaces'}}
        self.boundaries_nfaces_v = [v['nfaces'] for v in self.boundaries.values()]
        self.boundaries_nfaces_sum = sum(self.boundaries_nfaces_v)
        self.size_estimation = (len(self.evolving_keys)-1)*self.ncells +self.nfaces + len(self.evolving_keys_b)*self.boundaries_nfaces_sum
        self.groups = self.get_groups()
        self.running_directory = case.running_directory
        self.solver = solver.encode('utf-8')
        self.solver_AD = solver_AD.encode('utf-8')
        
        self.c_lib = ctypes.CDLL(libCaller)
        self.c_lib.caller.argtypes = [ctypes.c_char_p, ctypes.c_char_p, 
                                      ctypes.c_char_p, ctypes.POINTER(ctypes.c_int),
                                      ctypes.c_int, ctypes.c_int]
        
        self.c_lib.caller.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
        self.c_lib.memory_cleaning.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int]
        
        self.upstream= []
        
        super().__init__(trainable=False)
    
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
    
    def construct_chars (self, state, coefficients, startTimes, endTimes, n_batch, timestep_id):
        '''Creates the chars of fields and case directories to send to the OFcaller'''
        
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
        coefficients_proto = tf.nest.map_structure(tf.make_tensor_proto, coefficients)
        startTimes_proto = tf.make_tensor_proto(startTimes)
        endTimes_proto = tf.make_tensor_proto(endTimes)
        state_np = tf.nest.map_structure(tf.make_ndarray, state_proto)
        coefficients_np = tf.nest.map_structure(tf.make_ndarray, coefficients_proto)
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
            data_batch['Ufaces'] = np.concatenate((tf.expand_dims(coefficients_np['x_velocity_edge'][i,:], axis=-1), tf.expand_dims(coefficients_np['y_velocity_edge'][i,:], axis=-1), np.zeros([self.nfaces,1])), axis=1)
            files_OF_format = self.case.modify_templates(data_batch, startTime, mode = 'string')
            dict_fields ="".join([dict_fields, construct_argument (files_OF_format)])
   
        case_directory_char = (self.case.running_directory+'\0').encode('utf-8')
        sizes_fields = char_position(dict_fields)
        fields_char = dict_fields.encode('utf-8')

        return case_directory_char, fields_char, sizes_fields

    def OpenFOAMCaller(self, x0, x1, x2, x3, x4, x5):
        ''''Call to the external function OpenFOAM'''
        
        ncases = x4
        nfields = len(self.evolving_keys)
        nfields_b = len(self.evolving_keys_b)
        
        returned_c = self.c_lib.caller(x0, x1, x2, (ctypes.c_int * len(x3))(*x3), x4, x5)

        # initializing result containers
        results = {k:np.ones((ncases, self.ncells))*-99. for k in self.evolving_keys if k != 'phi'}
        self.phi = np.ones((ncases, self.nfaces))*-99.
        results_b = {}
        for k in self.evolving_keys_b:
            results_b[k] = {v['name']:np.ones((ncases,v['nfaces']))*-99. for v in self.boundaries.values()}
        
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
                results_b[self.evolving_keys_b[group_f]][self.boundaries[str(group_b)]['name']][i,face] = case[k]
   
        _ = self.c_lib.memory_cleaning(returned_c, ncases) #delete shared memory location
        
        results_TF_masked = {k:tf.convert_to_tensor(results[k], dtype=tf.float32) 
                      for k,v in results.items()}
        results_b_TF_masked = {k:{b_k:tf.convert_to_tensor(b_v, dtype=tf.float32) 
                            for b_k,b_v in results_b[k].items() if b_k in v.keys()} 
                            for k,v in self.state.items() if k in self.evolving_keys_b}
        self.phi_b = results_b['phi_b'] 
        
        return {**results_TF_masked, **results_b_TF_masked}
    
      
    @tf.custom_gradient        
    def run_cases (self, coefficients):

        self.n_batch, _ = list(coefficients.values())[0].shape
            
        self.case_directory_char, self.fields_char, self.sizes_fields = self.construct_chars (
            self.state, coefficients, self.timestep_current, self.timestep_to_predict, self.n_batch, self.timestep_id)
        
   
        new_fields = self.OpenFOAMCaller(self.solver, self.case_directory_char, self.fields_char, self.sizes_fields,
                                    self.n_batch, self.size_estimation)

        my_fields = ['x_velocity', 'y_velocity'] #TODO: Esto lo tengo que añadir como input en el modelo min_keys

        result_target = {k:v for k,v in new_fields.items() if k in my_fields}
        self.result_non_target.update({k:v for k,v in new_fields.items() if k not in my_fields})
        
        def custom_grad(x_vel_upstream, y_vel_upstream): #(10,4736), (10,4736)
            self.upstream.append({'x_velocity_edge': x_vel_upstream, 'y_velocity_edge': y_vel_upstream})
            
            def OpenFOAMCaller_AD(x0, x1, x2, x3, x4, x5):
                ''''Call to the external function OpenFOAM'''
                
                ncases = x4
                nfields = 2
                
                returned_c = self.c_lib.caller(x0, x1, x2, (ctypes.c_int * len(x3))(*x3), x4, x5) #charge the pointer to the array of pointers

                results_OF = np.ones((ncases, self.nfaces, nfields))*-99

                for i in range(ncases):
                    case = ctypes.cast(returned_c, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))[i] #pointer to the array of case i
                    for j in range(self.nfaces*nfields):
                        group = int(math.floor(j/self.nfaces)) #Field represented
                        cell = j - group*self.nfaces
                        results_OF[i,cell,group] = case[j]
           

                _ = self.c_lib.memory_cleaning(returned_c, ncases)
                
                results = {}
                x_velocity_new = []
                y_velocity_new = []
                
                for i in range (ncases):
                    x_velocity_new.append(results_OF[i,:,0])
                    y_velocity_new.append(results_OF[i,:,1])
                    
                results['x_velocity_edge'] = tf.convert_to_tensor(x_velocity_new, dtype=tf.float32)
                results['y_velocity_edge'] = tf.convert_to_tensor(y_velocity_new, dtype=tf.float32)
                
                return results

            def modify_chars (dUx, dUy):
                dict_fields = ''
                for i in range(self.n_batch):
                    temp_char = self.fields_char[self.sizes_fields[i]:self.sizes_fields[i+1]]
                    data_batch = {}
                    data_batch = np.concatenate((tf.expand_dims(dUx[i,:], axis=-1), tf.expand_dims(dUy[i,:], axis=-1), np.zeros([self.ncells,1])), axis=1)
                    files = self.case.get_derivative_template(data_batch)
                    dict_fields = "".join([dict_fields, temp_char.decode('utf-8')[0:-1],'dU_dict {', files, ';};\0'])
                
                sizes_fields = [0]
                for i in range(len(dict_fields)):
                    if dict_fields[i] == '\x00': sizes_fields.append(i+1)
                dict_fields = dict_fields.encode('utf-8')
  
                return dict_fields, sizes_fields
           
            modified_fields_char, modified_sizes_fields = modify_chars(x_vel_upstream, y_vel_upstream)           
            new_fields = OpenFOAMCaller_AD(self.solver_AD, self.case_directory_char, modified_fields_char, modified_sizes_fields,
                                                self.n_batch, self.nfaces*2)
            
            return new_fields

        return result_target, custom_grad

    def call(self, state, coefficients, timestep_current, timestep_to_predict, timestep_id):
        
        self.state = state
        self.result_non_target = {}
        self.timestep_current = timestep_current
        self.timestep_to_predict = timestep_to_predict
        self.timestep_id = timestep_id
        
        result = self.run_cases(coefficients)
    
        result.update(self.result_non_target)
        return result
    

class TimeStepModel(tf.keras.Model):
  """Model that predicts the state at the next time-step."""

  def __init__(
      self,
      equation: equations.Equation,
      num_time_steps: int = 1,
      target: Optional[str] = None,
      name: str = 'time_step_model',
  ):
    """Initialize a time-step model."""

    super().__init__(name=name)
    if num_time_steps < 1:
      raise ValueError('must use at least one time step')

    self.equation = equation    
    self.num_time_steps = num_time_steps

    if target is None and len(equation.evolving_keys) == 1:
      (target,) = equation.evolving_keys
    self.target = target

  
  def call(self, inputs: List) -> Dict[str, tf.Tensor]:
    """Predict the target state after multiple time-steps.

    Args:
      inputs[0]: dict of tensors with dimensions [batch, ncells].
      inputs[1]: tensor with dimensions [batch, 1].
      inputs[2]: tensor with dimensions [batch, num_timesteps_to_predict].

    Returns:
      labels: tensor with dimensions [batch, time, ncells], giving the target
        value of the predicted state at steps [1, ..., self.num_time_steps]
        for model training.
    """
    
    constant_state = {k:v for k,v in inputs[0].items()
                      if k in self.equation.constant_keys }
    initial_evolving_inputs = {k:v for k,v in inputs[0].items() 
                      if k in self.equation.evolving_keys | self.equation.evolving_keys_b}
    timesteps_initial = inputs[1]
    timesteps_to_predict = inputs[2]

    def advance(evolving_variables, timestep_to_predict):
        evolving_state, evolving_timesteps, timestep_id = evolving_variables
        return self.take_time_step({**evolving_state, **constant_state}, evolving_timesteps, timestep_to_predict, timestep_id) #From LearnedInterpolationModel
  
    step = 0

    advanced = tf.scan(
        advance, tf.transpose(timesteps_to_predict), initializer=(initial_evolving_inputs, timesteps_initial, step))
    
    advanced = tensor_ops.moveaxis(advanced[0], source=0, destination=1)

    result  = {k:v for k,v in advanced.items() if '_b' not in k}# if k in {'x_velocity', 'y_velocity','p'}}

    return result

  def predict_all(self, inputs: List) -> Dict[str, tf.Tensor]:
    """Predict the target state after multiple time-steps.

    Args:
      inputs[0]: dict of tensors with dimensions [batch, ncells].
      inputs[1]: tensor with dimensions [batch, 1].
      inputs[2]: tensor with dimensions [batch, num_timesteps_to_predict].

    Returns:
      labels: tensor with dimensions [batch, time, ncells], giving the target
        value of the predicted state at steps [1, ..., self.num_time_steps]
        for model training.
    """
    
    constant_state = {k:v for k,v in inputs[0].items()
                      if k in self.equation.constant_keys }
    initial_evolving_inputs = {k:v for k,v in inputs[0].items() 
                      if k in self.equation.evolving_keys | self.equation.evolving_keys_b}
    timesteps_initial = inputs[1]
    timesteps_to_predict = inputs[2]

    def advance(evolving_variables, timestep_to_predict):
        evolving_state, evolving_timesteps, timestep_id = evolving_variables
        return self.take_time_step({**evolving_state, **constant_state}, evolving_timesteps, timestep_to_predict, timestep_id) #From LearnedInterpolationModel
  
    step = 0

    advanced = tf.scan(
        advance, tf.transpose(timesteps_to_predict), initializer=(initial_evolving_inputs, timesteps_initial, step))
    
    advanced = tensor_ops.moveaxis(advanced[0], source=0, destination=1)

    return advanced
    
class LearnedInterpolationModel(TimeStepModel):
  """Model that predicts the next time-step using Learned Interpolation method.
  """

  def __init__(self, equation, case, solver, solver_AD, libCaller, num_time_steps=1, target=None,
               learned_keys = None, name='learned_interpolation_model'):

    super().__init__(equation, num_time_steps, target, name)

    self.forward_problem = ForwardProblemModel(equation, case, solver, solver_AD, libCaller)
    self.target = target
    
  def take_time_step(
      self, state: Mapping[str, tf.Tensor], timestep_current: tf.Tensor,
      timestep_to_predict: tf.Tensor, timestep_id: int) -> Dict[str, tf.Tensor]:
    
    input_NN = {k: v for k, v in state.items() if k in self.target} # select only viewed fields
    
    coefficients = self.inverse_problem(input_NN) 
    
    result = self.forward_problem(state, coefficients, timestep_current, timestep_to_predict, timestep_id)
   
    timestep_id += 1
    
    return (result, timestep_to_predict, timestep_id)


class InverseProblemModel(LearnedInterpolationModel):
  """Model that computes the learned interpolation coefficients."""

  def __init__(self, equation, case, solver, solver_AD, libCaller, 
               stencil_size=(4,4), bounding_perc = 0.0, 
               initial_accuracy_order=1, 
               constrained_accuracy_order=1, 
               num_layers_glob=2, num_layers_loc=4,
               learned_keys=None,
               fixed_keys=None, core_model_func=DeepConvectionNet,
               num_time_steps=1, target=None,
               name='inverse_problem_model', **kwargs):
#TODO: add learned keys and delete num_time_steps and initial_accuracy_order and constrained_accuracy_order?     
    """Initialize class.

    Args:
      core_model_func: callable (function or class object). It should return
        a Keras model (or layer) instance, which contains trainable weights.
        The returned core_model instance should take a dict of tensors as input
        (see the call() method in the base TimeStepModel class).
        Additional kwargs are passed to this callable to specify hyperparameters
        of core_model (such as number of layers and convolutional filters).
    """
    
    super().__init__(equation, case, solver, solver_AD, libCaller, 
                     num_time_steps, target, learned_keys, name)

    self.learned_keys = sorted_learned_keys(learned_keys)
    grid = case.grid_coarse
    faces_dict = case.faces_dict_generator()
    boundary_patches = case.info_boundary_patches(faces_dict, stencil_size)
    self.extract_patches_layer = extract_patches_from_vector(stencil_size, case)
    self.output_layers = build_output_layers(
        equation, grid, self.learned_keys, boundary_patches, bounding_perc,
        stencil_size, initial_accuracy_order, constrained_accuracy_order, 
        layer_cls=VaryingCoefficientsLayer)

    self.num_outputs = sum(
        layer.kernel_size for layer in self.output_layers.values())
    
    self.core_model = core_model_func(self.num_outputs, stencil_size, target, 
                                        self.extract_patches_layer, faces_dict, 
                                        self.output_layers, num_layers_glob, 
                                        num_layers_loc, case)

  def inverse_problem(self, state):
    """See base class."""

    result = self.core_model(state)

    return result
