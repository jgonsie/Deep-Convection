# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:40:37 2022

@author: jesusglezs97
"""

"""StateDefinition and associated utilities."""

import enum
import typing
from typing import Any, Dict, Mapping, Tuple, Type, TypeVar

T = TypeVar('T')


class Prefix(enum.Enum):
  BASELINE = 'baseline_'  # prefix for derivatives evaluated on coarse grid
  MODEL = 'model_'  # prefix for states evaluated by the model
  EXACT = 'exact_'  # prefix for states obtained by coarse-graining


class Dimension(enum.IntEnum):
  """Enum denoting physical dimensions."""

  X = 1
  Y = 2
  Z = 3


class StateDefinition(typing.NamedTuple):
  """Description of the physical quantity that a state tensor corresponds to.

  Example usage:
    x_velocity = StateDefinition('velocity', (Dimension.X,), (0, 0, 0), (0, 0))

  Attributes:
    name: name of the corresponding physical variable.
    name_OF: name of the corresponding physical variable in OpenFOAM.
    tensor_indices: tuple of Dimension enum values
      indicating the physical tensor components that this state corresponds to,
      e.g., () for a scalar tensor, (Dimension.X,) for the x-component of a
      vector and (Dimension.Y,) for the y-component of a vector.
    derivative_orders: derivative orders with respect to x, y, t respectively.
    offset: number of half-integer shifts on the grid. An offset of (0, 0) is
      centered on the unit-cell.
  """

  name: str
  name_OF: str
  tensor_indices: Tuple[Dimension, ...]
  derivative_orders: Tuple[int, int, int]
  offset: Tuple[int, int]
  variant: str = ''

  @classmethod
  def from_config(cls: Type[T], config: Mapping[str, Any]) -> T:
    """Construct a state from a configuration dict."""
    name = config['name']
    tensor_indices = tuple(
        Dimension[index] for index in config['tensor_indices']
    )
    derivative_orders = tuple(config['derivative_orders'])
    offset = tuple(config['offset'])
    return cls(name, tensor_indices, derivative_orders, offset)

  def to_config(self) -> Dict[str, Any]:
    """Creates a configuration dict representing the state definition."""
    tensor_indices = [index.name for index in self.tensor_indices]
    return dict(
        name=self.name,
        tensor_indices=tensor_indices,
        derivative_orders=list(self.derivative_orders),
        offset=list(self.offset),
    )

  def swap_xy(self: T) -> T:
    """Swap x and y dimensions on this state."""

    def _tensor_index_swap(index: Dimension) -> Dimension:
      return {
          Dimension.X: Dimension.Y,
          Dimension.Y: Dimension.X,
      }.get(index, index)

    tensor_indices = tuple(map(_tensor_index_swap, self.tensor_indices))
    derivative_orders = (self.derivative_orders[1],
                         self.derivative_orders[0],
                         self.derivative_orders[2])
    offset = self.offset[::-1]
    return self._replace(tensor_indices=tensor_indices,
                         derivative_orders=derivative_orders,
                         offset=offset)

  def time_derivative(self: T) -> T:
    """Returns a StateDefinition with derivative_order_t incremented by 1."""
    derivatives = list(self.derivative_orders)
    derivatives[-1] += 1
    return self._replace(derivative_orders=tuple(derivatives))

  def with_prefix(self: T, prefix: Prefix) -> T:
    """Returns a StateDefinition with a prefix added to the name field."""
    return self._replace(name=prefix.value + self.name)

  def model(self: T) -> T:
    """Returns a StateDefinition for "model" states."""
    return self.with_prefix(Prefix.MODEL)

  def baseline(self: T) -> T:
    """Returns a StateDefinition for "baseline" states."""
    return self.with_prefix(Prefix.BASELINE)

  def exact(self: T) -> T:
    """Returns a StateDefinition for "exact" states."""
    return self.with_prefix(Prefix.EXACT)
