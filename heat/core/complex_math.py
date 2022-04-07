"""
This module handles operations focussing on complex numbers.
"""

import torch
from typing import Optional, Union

from . import _operations
from . import constants
from . import factories
from . import trigonometrics
from . import types
from .dndarray import DNDarray

__all__ = ["angle", "conj", "conjugate", "imag", "real", "iadd"]


def angle(x: DNDarray, deg: bool = False, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Calculate the element-wise angle of the complex argument.

    Parameters
    ----------
    x : DNDarray
        Input array for which to compute the angle.
    deg : bool, optional
        Return the angle in degrees (True) or radiands (False).
    out : DNDarray, optional
        Output array with the angles.

    Examples
    --------
    >>> ht.angle(ht.array([1.0, 1.0j, 1+1j, -2+2j, 3 - 3j]))
    DNDarray([ 0.0000,  1.5708,  0.7854,  2.3562, -0.7854], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.angle(ht.array([1.0, 1.0j, 1+1j, -2+2j, 3 - 3j]), deg=True)
    DNDarray([  0.,  90.,  45., 135., -45.], dtype=ht.float32, device=cpu:0, split=None)
    """
    a = _operations.__local_op(torch.angle, x, out)

    if deg:
        a *= 180 / constants.pi

    return a


def conjugate(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Compute the complex conjugate, element-wise.

    Parameters
    ----------
    x : DNDarray
        Input array for which to compute the complex conjugate.
    out : DNDarray, optional
        Output array with the complex conjugates.

    Examples
    --------
    >>> ht.conjugate(ht.array([1.0, 1.0j, 1+1j, -2+2j, 3 - 3j]))
    DNDarray([ (1-0j),     -1j,  (1-1j), (-2-2j),  (3+3j)], dtype=ht.complex64, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.conj, x, out)


# alias
conj = conjugate

# DNDarray method
DNDarray.conj = lambda self, out=None: conjugate(self, out)
DNDarray.conj.__doc__ = conjugate.__doc__


def imag(x: DNDarray) -> DNDarray:
    """
    Return the imaginary part of the complex argument. The returned DNDarray and the input DNDarray share the same underlying storage.

    Parameters
    ----------
    x : DNDarray
        Input array for which the imaginary part is returned.

    Examples
    --------
    >>> ht.imag(ht.array([1.0, 1.0j, 1+1j, -2+2j, 3 - 3j]))
    DNDarray([ 0.,  1.,  1.,  2., -3.], dtype=ht.float32, device=cpu:0, split=None)
    """
    if types.heat_type_is_complexfloating(x.dtype):
        return _operations.__local_op(torch.imag, x, None)
    else:
        return factories.zeros_like(x)


def real(x: DNDarray) -> DNDarray:
    """
    Return the real part of the complex argument. The returned DNDarray and the input DNDarray share the same underlying storage.

    Parameters
    ----------
    x : DNDarray
        Input array for which the real part is returned.

    Examples
    --------
    >>> ht.real(ht.array([1.0, 1.0j, 1+1j, -2+2j, 3 - 3j]))
    DNDarray([ 1.,  0.,  1., -2.,  3.], dtype=ht.float32, device=cpu:0, split=None)
    """
    if types.heat_type_is_complexfloating(x.dtype):
        return _operations.__local_op(torch.real, x, None)
    else:
        return x


def iadd(t1: Union[DNDarray, float], t2: Union[DNDarray, float]) -> DNDarray:
    """
    Element-wise addition of values from two operands, commutative.
    Takes the first and second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) whose elements are to be added
    as argument and returns a ``DNDarray`` containing the results of element-wise addition of ``t1`` and ``t2``.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand involved in the addition
    t2: DNDarray or scalar
        The second operand involved in the addition

    Examples
    --------
    >>> import heat as ht
    >>> ht.add(1.0, 4.0)
    DNDarray([5.], dtype=ht.float32, device=cpu:0, split=None)
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.add(T1, T2)
    DNDarray([[3., 4.],
              [5., 6.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = 2.0
    >>> ht.add(T1, s)
    DNDarray([[3., 4.],
              [5., 6.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__local_op(t1.larray.add_, t2)


DNDarray.__iadd__ = lambda self, other: iadd(self, other)
DNDarray.__iadd__.__doc__ = iadd.__doc__
