# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:40:37 2022

@author: jesusglezs97
"""

"""Equation classes describe differential equations.

Equation class encapsulate the relation between the spatial state derivatives
and time derivatives for different PDE. State derivatives can be used
combined differently to yield various update schemes. (e.g. finite differences
vs finite volumes).
"""

import collections
from typing import (
    Any, Dict, Iterator, Mapping, Set, Tuple, Type, TypeVar, Union)
import tensorflow as tf

from SRC import grids
from SRC import tensor_ops

Shape = Union[int, Tuple[int]]
T = TypeVar('T')


class Equation:
  """"Base class for PDEs.

  Defines method time_derivative that constructs time derivative of the
  current state using state derivatives provided by the model. The aim is
  to be able to use multiple models and the same integrator for uniform
  performance comparison and experimentation.

  Attributes:
    DISCRETIZATION_NAME: Name of the discretization method.
    METHOD: Discretization method type (finite difference or finite volume).
    MONOTONIC: Are dynamics guaranteed to be monotonic?
    key_definitions: a dict mapping strings to StateDefinitions, providing a map
      from keyword arguments required by time_derivative and take_time_step to
      StateDefintiion instances defining what these keys represent.
    evolving_keys: the set of variable names found in key_definitions that fully
      describe the time-dependent state of the equation.
    constant_keys: the set of variable names found in key_definitions that fully
      describe the time-independent state of the equation.
  """
  CONTINUOUS_EQUATION_NAME = ...  # type: str
  DISCRETIZATION_NAME = ...   # type: str
  METHOD = ...  # type: polynomials.Method
  MONOTONIC = ...  # type: bool

  key_definitions = ...  # type: Dict[str, states.StateDefinition]
  evolving_keys = ...  # type: Set[str]
  constant_keys = ...  # type: Set[str]

  def __init__(self):
    self._validate_keys()

  def get_parameters(self) -> Dict[str, Any]:
    """Return a dictionary of all parameters used to initialize this object."""
    return {}

  def _validate_keys(self):
    """Validate the key_definitions, evolving_keys and constant_keys attributes.
    """
    repeated_keys = self.evolving_keys & self.constant_keys  & self.evolving_keys_b #intersection
    if repeated_keys:
      raise ValueError('overlapping entries between evolving_keys and '
                       'constant_keys: {}'.format(repeated_keys))

    missing_keys = self.derived_keys - self.all_keys
    if missing_keys:
      raise ValueError('not all entries in evolving_keys and constant_keys '
                       'found in key_definitions: {}'.format(missing_keys))

    for key in self.base_keys:
      key_def = self.key_definitions[key]
      if key_def.derivative_orders != (0, 0, 0):
        raise ValueError('keys present in evolving keys and constant keys '
                         'cannot have derivatives, but {} is defined as {}'
                         .format(key, key_def))

    base_name_and_indices = []
    for key in self.base_keys:
      key_def = self.key_definitions[key]
      base_name_and_indices.append((key_def.name, key_def.tensor_indices))
    base_name_and_indices_set = set(base_name_and_indices)

    if len(base_name_and_indices_set) < len(base_name_and_indices):
      raise ValueError('(name, tensor_indices) pairs on each key found in '
                       'evolving_keys and constant keys must be unique, but '
                       'some are repeated: {}'
                       .format(base_name_and_indices))

    for key in self.derived_keys:
      key_def = self.key_definitions[key]
      name_and_indices = (key_def.name, key_def.tensor_indices)
      # print(key_def.name, key_def.tensor_indices)
      if name_and_indices not in base_name_and_indices_set:
        raise ValueError('all keys defined in key_definitions must have the '
                         'same (name, tensor_indices) pair as an entry found '
                         'in evolving_keys or state_keys, but this entry does '
                         'not: {}'.format(key))

  @property
  def all_keys(self) -> Set[str]:
    """The set of all defined keys."""
    return set(self.key_definitions)

  @property
  def base_keys(self) -> Set[str]:
    """Keys corresponding to non-derived entries in the state.

    Returns:
      The union of evolving and constant keys and non-numerical keys.
    """
    return self.evolving_keys | self.evolving_keys_b | self.constant_keys # union

  @property
  def derived_keys(self) -> Set[str]:
    """Keys corresponding to derived entries in the state.

    These can be estimated from other states using either learned or fixed
    finite difference schemes.

    Returns:
      The set of defined keys not found in evolving_keys or constant_keys.
    """
    return set(self.key_definitions) - self.base_keys

  def find_base_key(self, key: str) -> str:
    """Find the matching "base" key from which to estimate this key."""
    definition = self.key_definitions[key]
    for candidate in self.base_keys:
      candidate_def = self.key_definitions[candidate]
      if (candidate_def.name == definition.name and
          candidate_def.tensor_indices == definition.tensor_indices):
        return candidate
    raise AssertionError  # should be impossible per _validate_keys()

  def regrid(
      self,
      state: Dict[str, tf.Tensor],
      source: grids.Grid,
      destination: grids.Grid,
  ) -> Dict[str, tf.Tensor]:
    """Regrid this state to a coarser resolution.

    Equations should override this method if the default regridding logic
    (designed for finite volume methods) is not appropriate.

    Args:
      state: state(s) representing the initial configuration of the system
      source: fine resolution Grid.
      destination: coarse resolution Grid.

    Returns:
      Tensor(s) representing the input state at lower resolution.
    """
    return tensor_ops.regrid(state, self.key_definitions, source, destination)  # pytype: disable=bad-return-type

  def to_config(self) -> Dict[str, Any]:
    """Creates a configuration dict representing this equation."""
    return dict(
        continuous_equation=self.CONTINUOUS_EQUATION_NAME,
        discretization=self.DISCRETIZATION_NAME,
        parameters=self.get_parameters(),
    )

  @classmethod
  def from_config(cls: Type[T], config: Mapping[str, Any]) -> T:
    """Construct an equation from a configuration dict."""
    continuous_equation = config['continuous_equation']
    if continuous_equation != cls.CONTINUOUS_EQUATION_NAME:
      raise ValueError(
          'wrong continuous equation {} != {}'
          .format(continuous_equation, cls.CONTINUOUS_EQUATION_NAME))

    discretization = config['discretization']
    if discretization != cls.DISCRETIZATION_NAME:
      raise ValueError(
          'wrong discretization {} != {}'
          .format(discretization, cls.DISCRETIZATION_NAME))

    return cls(**config['parameters'])


def _breadth_first_subclasses(base: Type[T]) -> Iterator[Type[T]]:
  """Yields all subclasses of a given class in breadth-first order."""
  # https://stackoverflow.com/questions/3862310
  subclasses = collections.deque([base])
  while subclasses:
    subclass = subclasses.popleft()
    yield subclass
    subclasses.extend(subclass.__subclasses__())


def matching_equation_type(
    continuous_equation: str,
    discretization: str,
) -> Type[Equation]:
  """Find the matching equation type."""
  matches = []
  candidates = list(_breadth_first_subclasses(Equation))
  for subclass in candidates:
    if (subclass.CONTINUOUS_EQUATION_NAME == continuous_equation
        and subclass.DISCRETIZATION_NAME == discretization):
      matches.append(subclass)

  if not matches:
    equations_list = [c.__name__ for c in candidates]
    raise ValueError(
        'continuous equation {!r} and discretization {!r} not found '
        'in equations list {}. Maybe you forgot to import the '
        'module that defines the equation first?'
        .format(continuous_equation, discretization, equations_list))
  elif len(matches) > 1:
    raise ValueError('too many matches found: {}'.format(matches))

  return matches[0]


def equation_from_config(
    config: Mapping[str, Any],
    discretization: str = None,
) -> Equation:
  """Constructs an equation from the Equation protocol buffer.

  Args:
    config: equation specific configuration dictionary.
    discretization: override the discretization scheme for the equation. Needed
      for testing different implementation in training and evaluation.

  Returns:
    Equation object.

  Raises:
    ValueError: Provided protocol buffer was not recognized, check proto names.
  """
  continuous_equation = config['continuous_equation']
  if discretization is None:
    discretization = config['discretization']
  equation_type = matching_equation_type(continuous_equation, discretization)
  return equation_type.from_config(config)
