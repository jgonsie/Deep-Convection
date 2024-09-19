# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:40:37 2022

@author: jesusglezs97
"""

"""Navier-Stokes equations definition."""

from SRC import equations
from SRC import polynomials
from SRC import states

StateDef = states.StateDefinition

X = states.Dimension.X
Y = states.Dimension.Y
Z = states.Dimension.Z

NO_DERIVATIVES = (0, 0, 0)
D_X = (1, 0, 0)
D_Y = (0, 1, 0)
D_XX = (2, 0, 0)
D_YY = (0, 2, 0)

NO_OFFSET = (0, 0)
X_PLUS_HALF = (1, 0)
Y_PLUS_HALF = (0, 1)

class NS_kw(equations.Equation):
  """Base class for Navier-Stokes k-w equations."""

  CONTINUOUS_EQUATION_NAME = 'navier_stokes'
  DISCRETIZATION_NAME = 'finite_volume'
  METHOD = polynomials.Method.FINITE_VOLUME
  MONOTONIC = False

  def __init__(self):
      
    self.key_definitions = {
        'k': StateDef('turb_kine_energy_centers', 'k', (), NO_DERIVATIVES, NO_OFFSET),
        'k_b': StateDef('turb_kine_energy_boundaries', 'k', (), NO_DERIVATIVES, NO_OFFSET),
        'nut': StateDef('turb_viscosity_centers', 'nut', (), NO_DERIVATIVES, NO_OFFSET),
        'nut_b': StateDef('turb_viscosity_boundaries', 'nut', (), NO_DERIVATIVES, NO_OFFSET),
        'omega': StateDef('dissipation_centers', 'omega', (), NO_DERIVATIVES, NO_OFFSET),
        'omega_b': StateDef('dissipation_boundaries', 'omega', (), NO_DERIVATIVES, NO_OFFSET),
        'p': StateDef('pressure_centers', 'p', (), NO_DERIVATIVES, NO_OFFSET),
        'p_b': StateDef('pressure_boundaries', 'p', (), NO_DERIVATIVES, NO_OFFSET),
        'x_velocity': StateDef('velocity_centers', 'U', (X,), NO_DERIVATIVES, NO_OFFSET), 
        'x_velocity_b': StateDef('velocity_boundaries', 'U', (X,), NO_DERIVATIVES, NO_OFFSET), 
        'y_velocity': StateDef('velocity_centers', 'U', (Y,), NO_DERIVATIVES, NO_OFFSET),
        'y_velocity_b': StateDef('velocity_boundaries', 'U', (Y,), NO_DERIVATIVES, NO_OFFSET),
        'x_velocity_edge':
            StateDef('velocity_centers', 'Ufaces', (X,), NO_DERIVATIVES, X_PLUS_HALF),
        'y_velocity_edge': 
            StateDef('velocity_centers', 'Ufaces', (Y,), NO_DERIVATIVES, Y_PLUS_HALF)
    }

    self.evolving_keys = {'x_velocity', 'y_velocity', 'k', 'nut', 'omega', 'p'}
    self.evolving_keys_b = {'x_velocity_b', 'y_velocity_b', 'k_b', 'nut_b', 'omega_b', 'p_b'}
    self.constant_keys = set()

    super().__init__()
    
class NS_LES(equations.Equation):
  """Base class for Navier-Stokes LES equations."""

  CONTINUOUS_EQUATION_NAME = 'navier_stokes'
  DISCRETIZATION_NAME = 'finite_volume'
  METHOD = polynomials.Method.FINITE_VOLUME
  MONOTONIC = False

  def __init__(self):
      
    self.key_definitions = {
        'nut': StateDef('turb_viscosity_centers', 'nut', (), NO_DERIVATIVES, NO_OFFSET),
        'nut_b': StateDef('turb_viscosity_boundaries', 'nut', (), NO_DERIVATIVES, NO_OFFSET),
        'p': StateDef('pressure_centers', 'p', (), NO_DERIVATIVES, NO_OFFSET),
        'p_b': StateDef('pressure_boundaries', 'p', (), NO_DERIVATIVES, NO_OFFSET),
        'x_velocity': StateDef('velocity_centers', 'U', (X,), NO_DERIVATIVES, NO_OFFSET), 
        'x_velocity_b': StateDef('velocity_boundaries', 'U', (X,), NO_DERIVATIVES, NO_OFFSET), 
        'y_velocity': StateDef('velocity_centers', 'U', (Y,), NO_DERIVATIVES, NO_OFFSET),
        'y_velocity_b': StateDef('velocity_boundaries', 'U', (Y,), NO_DERIVATIVES, NO_OFFSET),
        'x_velocity_edge':
            StateDef('velocity_centers', 'Ufaces', (X,), NO_DERIVATIVES, X_PLUS_HALF),
        'y_velocity_edge': 
            StateDef('velocity_centers', 'Ufaces', (Y,), NO_DERIVATIVES, Y_PLUS_HALF)
    }

    self.evolving_keys = {'x_velocity', 'y_velocity', 'p', 'nut'}
    self.evolving_keys_b = {'x_velocity_b', 'y_velocity_b', 'p_b', 'nut_b'}
    self.constant_keys = set()

    super().__init__()
