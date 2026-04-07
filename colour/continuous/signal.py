"""
Signal
======

Define support for continuous signal representation and manipulation.

This module provides the :class:`colour.continuous.Signal` class for
representing and operating on continuous signals with the specified domain and
range values, supporting interpolation and extrapolation operations.

-   :class:`colour.continuous.Signal`
"""

from __future__ import annotations

import typing
from collections.abc import Iterator, KeysView, Mapping, Sequence, ValuesView
from operator import pow  # noqa: A004
from operator import add, iadd, imul, ipow, isub, itruediv, mul, sub, truediv

import numpy as np

from colour.algebra import Extrapolator, KernelInterpolator
from colour.constants import DTYPE_FLOAT_DEFAULT
from colour.continuous import AbstractContinuousFunction

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        ArrayLike,
        Literal,
        NDArrayFloat,
        ProtocolExtrapolator,
        ProtocolInterpolator,
        Real,
        Self,
        Type,
    )

from colour.hints import Callable, DTypeFloat, cast
from colour.utilities import (
    as_float_array,
    attest,
    fill_nan,
    full,
    is_pandas_installed,
    multiline_repr,
    ndarray_copy,
    ndarray_copy_enable,
    optional,
    required,
    runtime_warning,
    tsplit,
    tstack,
    validate_method,
)
from colour.utilities.common import int_digest
from colour.utilities.documentation import is_documentation_building

if typing.TYPE_CHECKING or is_pandas_installed():
    from pandas import Series  # pragma: no cover
else:  # pragma: no cover
    from unittest import mock

    Series = mock.MagicMock()

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Signal",
]


class Signal(AbstractContinuousFunction):
    """
    Define the base class for a continuous signal.

    The class implements the :meth:`Signal.function` method so that evaluating
    the function for any independent domain variable :math:`x \\in\\mathbb{R}`
    returns a corresponding range variable :math:`y \\in\\mathbb{R}`. It adopts
    an interpolating function encapsulated inside an extrapolating function.
    The resulting function independent domain, stored as discrete values in
    the :attr:`colour.continuous.Signal.domain` property corresponds with the
    function dependent and already known range stored in the
    :attr:`colour.continuous.Signal.range` property.

    .. important::

        Specific documentation about getting, setting, indexing and slicing
        the continuous signal values is available in the
        :ref:`spectral-representation-and-continuous-signal` section.

    Parameters
    ----------
    data
        Data to be stored in the continuous signal.
    domain
        Values to initialise the :attr:`colour.continuous.Signal.domain`
        attribute with. If both ``data`` and ``domain`` arguments are
        defined, the latter will be used to initialise the
        :attr:`colour.continuous.Signal.domain` property.

    Other Parameters
    ----------------
    dtype
        Floating point data type.
    extrapolator
        Extrapolator class type to use as extrapolating function.
    extrapolator_kwargs
        Arguments to use when instantiating the extrapolating function.
    interpolator
        Interpolator class type to use as interpolating function.
    interpolator_kwargs
        Arguments to use when instantiating the interpolating function.
    name
        Continuous signal name.

    Attributes
    ----------
    -   :attr:`~colour.continuous.Signal.dtype`
    -   :attr:`~colour.continuous.Signal.domain`
    -   :attr:`~colour.continuous.Signal.range`
    -   :attr:`~colour.continuous.Signal.interpolator`
    -   :attr:`~colour.continuous.Signal.interpolator_kwargs`
    -   :attr:`~colour.continuous.Signal.extrapolator`
    -   :attr:`~colour.continuous.Signal.extrapolator_kwargs`
    -   :attr:`~colour.continuous.Signal.function`

    Methods
    -------
    -   :meth:`~colour.continuous.Signal.__init__`
    -   :meth:`~colour.continuous.Signal.__str__`
    -   :meth:`~colour.continuous.Signal.__repr__`
    -   :meth:`~colour.continuous.Signal.__hash__`
    -   :meth:`~colour.continuous.Signal.__getitem__`
    -   :meth:`~colour.continuous.Signal.__setitem__`
    -   :meth:`~colour.continuous.Signal.__contains__`
    -   :meth:`~colour.continuous.Signal.__eq__`
    -   :meth:`~colour.continuous.Signal.__ne__`
    -   :meth:`~colour.continuous.Signal.arithmetical_operation`
    -   :meth:`~colour.continuous.Signal.signal_unpack_data`
    -   :meth:`~colour.continuous.Signal.fill_nan`
    -   :meth:`~colour.continuous.Signal.to_series`

    Examples
    --------
    Instantiation with implicit *domain*:

    >>> range_ = np.linspace(10, 100, 10)
    >>> print(Signal(range_))
    [[  0.  10.]
     [  1.  20.]
     [  2.  30.]
     [  3.  40.]
     [  4.  50.]
     [  5.  60.]
     [  6.  70.]
     [  7.  80.]
     [  8.  90.]
     [  9. 100.]]

    Instantiation with explicit *domain*:

    >>> domain = np.arange(100, 1100, 100)
    >>> print(Signal(range_, domain))
    [[ 100.   10.]
     [ 200.   20.]
     [ 300.   30.]
     [ 400.   40.]
     [ 500.   50.]
     [ 600.   60.]
     [ 700.   70.]
     [ 800.   80.]
     [ 900.   90.]
     [1000.  100.]]

    Instantiation with a *dict*:

    >>> print(Signal(dict(zip(domain, range_))))
    [[ 100.   10.]
     [ 200.   20.]
     [ 300.   30.]
     [ 400.   40.]
     [ 500.   50.]
     [ 600.   60.]
     [ 700.   70.]
     [ 800.   80.]
     [ 900.   90.]
     [1000.  100.]]

    Instantiation with a *Pandas* :class:`pandas.Series`:

    >>> if is_pandas_installed():
    ...     from pandas import Series
    ...
    ...     print(Signal(Series(dict(zip(domain, range_)))))  # doctest: +SKIP
    [[ 100.   10.]
     [ 200.   20.]
     [ 300.   30.]
     [ 400.   40.]
     [ 500.   50.]
     [ 600.   60.]
     [ 700.   70.]
     [ 800.   80.]
     [ 900.   90.]
     [1000.  100.]]

    Retrieving domain *y* variable for arbitrary range *x* variable:

    >>> x = 150
    >>> range_ = np.sin(np.linspace(0, 1, 10))
    >>> Signal(range_, domain)[x]  # doctest: +ELLIPSIS
    np.float64(0.0359701...)
    >>> x = np.linspace(100, 1000, 3)
    >>> Signal(range_, domain)[x]  # doctest: +ELLIPSIS
    array([...e-..., 4.7669395...e-01, 8.4147098...e-01])

    Using an alternative interpolating function:

    >>> x = 150
    >>> from colour.algebra import CubicSplineInterpolator
    >>> Signal(range_, domain, interpolator=CubicSplineInterpolator)[
    ...     x
    ... ]  # doctest: +ELLIPSIS
    np.float64(0.0555274...)
    >>> x = np.linspace(100, 1000, 3)
    >>> Signal(range_, domain, interpolator=CubicSplineInterpolator)[
    ...     x
    ... ]  # doctest: +ELLIPSIS
    array([0.        , 0.4794253..., 0.8414709...])
    """

    def __init__(
        self,
        data: ArrayLike | dict | Self | Series | ValuesView | None = None,
        domain: ArrayLike | KeysView | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(kwargs.get("name"))

        self._dtype: Type[DTypeFloat] = DTYPE_FLOAT_DEFAULT
        self._domain: NDArrayFloat = np.array([])
        self._range: NDArrayFloat = np.array([])
        self._interpolator: Type[ProtocolInterpolator] = KernelInterpolator
        self._interpolator_kwargs: dict = {}
        self._extrapolator: Type[ProtocolExtrapolator] = Extrapolator
        self._extrapolator_kwargs: dict = {
            "method": "Constant",
            "left": np.nan,
            "right": np.nan,
        }

        self.range, self.domain = self.signal_unpack_data(data, domain)[::-1]

        self.dtype = kwargs.get("dtype", self._dtype)

        self.interpolator = kwargs.get("interpolator", self._interpolator)
        self.interpolator_kwargs = kwargs.get(
            "interpolator_kwargs", self._interpolator_kwargs
        )
        self.extrapolator = kwargs.get("extrapolator", self._extrapolator)
        self.extrapolator_kwargs = kwargs.get(
            "extrapolator_kwargs", self._extrapolator_kwargs
        )

        self._function: Callable | None = None

    @property
    def dtype(self) -> Type[DTypeFloat]:
        """
        Getter and setter for the continuous signal dtype.

        Parameters
        ----------
        value
            Value to set the continuous signal dtype with.

        Returns
        -------
        Type[DTypeFloat]
            Continuous signal dtype.
        """

        return self._dtype

    @dtype.setter
    def dtype(self, value: Type[DTypeFloat]) -> None:
        """Setter for the **self.dtype** property."""

        attest(
            value in DTypeFloat.__args__,
            f'"dtype" must be one of the following types: {DTypeFloat.__args__}',
        )

        self._dtype = value

        # The following self-assignments are written as intended and
        # triggers the rebuild of the underlying function.
        if self.domain.dtype != value or self.range.dtype != value:
            self.domain = self.domain
            self.range = self.range

    @property
    def domain(self) -> NDArrayFloat:
        """
        Getter and setter for the continuous signal's independent
        domain variable :math:`x`.

        Parameters
        ----------
        value
            Value to set the continuous signal independent domain
            variable :math:`x` with.

        Returns
        -------
        :class:`numpy.ndarray`
            Continuous signal independent domain variable
            :math:`x`.
        """
        pass

    @domain.setter
    def domain(self, value: ArrayLike) -> None:
        """Setter for the **self.domain** property."""
        pass

    @property
    def range(self) -> NDArrayFloat:
        """
        Getter and setter for the continuous signal's range
        variable :math:`y`.

        Parameters
        ----------
        value
            Value to set the continuous signal's range variable
            :math:`y` with.

        Returns
        -------
        :class:`numpy.ndarray`
            Continuous signal's range variable :math:`y`.
        """
        pass

    @range.setter
    def range(self, value: ArrayLike) -> None:
        """Setter for the **self.range** property."""
        pass

    @property
    def interpolator(self) -> Type[ProtocolInterpolator]:
        """
        Getter and setter for the continuous signal interpolator
        type.

        Parameters
        ----------
        value
            Value to set the continuous signal interpolator type
            with.

        Returns
        -------
        Type[ProtocolInterpolator]
            Continuous signal interpolator type.
        """

        return self._interpolator

    @interpolator.setter
    def interpolator(self, value: Type[ProtocolInterpolator]) -> None:
        """Setter for the **self.interpolator** property."""

        # TODO: Check for interpolator compatibility.
        self._interpolator = value
        self._function = None  # Invalidate the underlying continuous function.

    @property
    def interpolator_kwargs(self) -> dict:
        """
        Getter and setter for the interpolator instantiation time arguments.

        Parameters
        ----------
        value
            Value to set the continuous signal interpolator
            instantiation time arguments to.

        Returns
        -------
        :class:`dict`
            Continuous signal interpolator instantiation time
            arguments.
        """
        pass

    @interpolator_kwargs.setter
    def interpolator_kwargs(self, value: dict) -> None:
        """Setter for the **self.interpolator_kwargs** property."""
        pass

    @property
    def extrapolator(self) -> Type[ProtocolExtrapolator]:
        """
        Getter and setter for the continuous signal extrapolator type.

        Parameters
        ----------
        value
            Value to set the continuous signal extrapolator type with.

        Returns
        -------
        Type[ProtocolExtrapolator]
            Continuous signal extrapolator type.
        """

        return self._extrapolator

    @extrapolator.setter
    def extrapolator(self, value: Type[ProtocolExtrapolator]) -> None:
        """Setter for the **self.extrapolator** property."""

        # TODO: Check for extrapolator compatibility.
        self._extrapolator = value
        self._function = None  # Invalidate the underlying continuous function.

    @property
    def extrapolator_kwargs(self) -> dict:
        """
        Getter and setter for the continuous signal extrapolator
        instantiation time arguments.

        Parameters
        ----------
        value
            Value to set the continuous signal extrapolator
            instantiation time arguments to.

        Returns
        -------
        :class:`dict`
            Continuous signal extrapolator instantiation time
            arguments.
        """
        pass

    @extrapolator_kwargs.setter
    def extrapolator_kwargs(self, value: dict) -> None:
        """Setter for the **self.extrapolator_kwargs** property."""
        pass

    @property
    @ndarray_copy_enable(False)
    def function(self) -> Callable:
        """
        Getter for the continuous signal callable.

        Returns
        -------
        Callable
            Continuous signal callable.
        """

        if self._function is None:
            # Create the underlying continuous function.

            if self._domain.size != 0 and self._range.size != 0:
                self._function = self._extrapolator(
                    self._interpolator(
                        self._domain, self._range, **self._interpolator_kwargs
                    ),
                    **self._extrapolator_kwargs,
                )
            else:

                def _undefined_function(
                    *args: Any,  # noqa: ARG001
                    **kwargs: Any,  # noqa: ARG001
                ) -> None:
                    """
                    Raise a :class:`ValueError` exception.

                    Other Parameters
                    ----------------
                    args
                        Arguments.
                    kwargs
                        Keywords arguments.

                    Raises
                    ------
                    ValueError
                    """
                    pass

                self._function = cast("Callable", _undefined_function)

        return cast("Callable", self._function)

    @ndarray_copy_enable(False)
    def __str__(self) -> str:
        """
        Return a formatted string representation of the continuous signal.

        Returns
        -------
        :class:`str`
            Formatted string representation.

        Examples
        --------
        >>> range_ = np.linspace(10, 100, 10)
        >>> print(Signal(range_))
        [[  0.  10.]
         [  1.  20.]
         [  2.  30.]
         [  3.  40.]
         [  4.  50.]
         [  5.  60.]
         [  6.  70.]
         [  7.  80.]
         [  8.  90.]
         [  9. 100.]]
        """

        return str(tstack([self._domain, self._range]))

    @ndarray_copy_enable(False)
    def __repr__(self) -> str:
        """
        Return an evaluable string representation of the continuous signal.

        Returns
        -------
        :class:`str`
            Evaluable string representation.

        Examples
        --------
        >>> range_ = np.linspace(10, 100, 10)
        >>> Signal(range_)
        Signal([[  0.,  10.],
                [  1.,  20.],
                [  2.,  30.],
                [  3.,  40.],
                [  4.,  50.],
                [  5.,  60.],
                [  6.,  70.],
                [  7.,  80.],
                [  8.,  90.],
                [  9., 100.]],
               KernelInterpolator,
               {},
               Extrapolator,
               {'method': 'Constant', 'left': nan, 'right': nan})
        """

        if is_documentation_building():  # pragma: no cover
            return f"{self.__class__.__name__}(name='{self.name}', ...)"

        return multiline_repr(
            self,
            [
                {
                    "formatter": lambda x: repr(  # noqa: ARG005
                        tstack([self._domain, self._range])
                    ),
                },
                {
                    "name": "interpolator",
                    "formatter": lambda x: self._interpolator.__name__,  # noqa: ARG005
                },
                {"name": "interpolator_kwargs"},
                {
                    "name": "extrapolator",
                    "formatter": lambda x: self._extrapolator.__name__,  # noqa: ARG005
                },
                {"name": "extrapolator_kwargs"},
            ],
        )

    @ndarray_copy_enable(False)
    def __hash__(self) -> int:
        """
        Compute the hash of the continuous signal.

        Returns
        -------
        :class:`int`
            Object hash.
        """

        return hash(
            (
                int_digest(self._domain.tobytes()),
                int_digest(self._range.tobytes()),
                self.interpolator.__name__,
                repr(self.interpolator_kwargs),
                self.extrapolator.__name__,
                repr(self.extrapolator_kwargs),
            )
        )

    def __getitem__(self, x: ArrayLike | slice) -> NDArrayFloat:
        """
        Return the corresponding range variable :math:`y` for the specified
        independent domain variable :math:`x`.

        Parameters
        ----------
        x
            Independent domain variable :math:`x`.

        Returns
        -------
        :class:`numpy.ndarray`
            Variable :math:`y` range value.

        Examples
        --------
        >>> range_ = np.linspace(10, 100, 10)
        >>> signal = Signal(range_)
        >>> print(signal)
        [[  0.  10.]
         [  1.  20.]
         [  2.  30.]
         [  3.  40.]
         [  4.  50.]
         [  5.  60.]
         [  6.  70.]
         [  7.  80.]
         [  8.  90.]
         [  9. 100.]]
        >>> signal[0]
        np.float64(10.0)
        >>> signal[np.array([0, 1, 2])]
        array([10., 20., 30.])
        >>> signal[0:3]
        array([10., 20., 30.])
        >>> signal[np.linspace(0, 5, 5)]  # doctest: +ELLIPSIS
        array([10.        , 22.8348902..., 34.8004492..., \
47.5535392..., 60.        ])
        """

        if isinstance(x, slice):
            return self._range[x]

        return self.function(x)

    def __setitem__(self, x: ArrayLike | slice, y: ArrayLike) -> None:
        """
        Set the corresponding range variable :math:`y` for the specified
        independent domain variable :math:`x`.

        Parameters
        ----------
        x
            Independent domain variable :math:`x`.
        y
            Corresponding range variable :math:`y`.

        Examples
        --------
        >>> range_ = np.linspace(10, 100, 10)
        >>> signal = Signal(range_)
        >>> print(signal)
        [[  0.  10.]
         [  1.  20.]
         [  2.  30.]
         [  3.  40.]
         [  4.  50.]
         [  5.  60.]
         [  6.  70.]
         [  7.  80.]
         [  8.  90.]
         [  9. 100.]]
        >>> signal[0] = 20
        >>> signal[0]
        np.float64(20.0)
        >>> signal[np.array([0, 1, 2])] = 30
        >>> signal[np.array([0, 1, 2])]
        array([30., 30., 30.])
        >>> signal[0:3] = 40
        >>> signal[0:3]
        array([40., 40., 40.])
        >>> signal[np.linspace(0, 5, 5)] = 50
        >>> print(signal)
        [[  0.    50.  ]
         [  1.    40.  ]
         [  1.25  50.  ]
         [  2.    40.  ]
         [  2.5   50.  ]
         [  3.    40.  ]
         [  3.75  50.  ]
         [  4.    50.  ]
         [  5.    50.  ]
         [  6.    70.  ]
         [  7.    80.  ]
         [  8.    90.  ]
         [  9.   100.  ]]
        >>> signal[np.array([0, 1, 2])] = np.array([10, 20, 30])
        >>> print(signal)
        [[  0.    10.  ]
         [  1.    20.  ]
         [  1.25  50.  ]
         [  2.    30.  ]
         [  2.5   50.  ]
         [  3.    40.  ]
         [  3.75  50.  ]
         [  4.    50.  ]
         [  5.    50.  ]
         [  6.    70.  ]
         [  7.    80.  ]
         [  8.    90.  ]
         [  9.   100.  ]]
        """

        if isinstance(x, slice):
            self._range[x] = y
        else:
            x = np.atleast_1d(x).astype(self.dtype)
            y = np.resize(y, x.shape)

            # Matching domain, updating existing `self._range` values.
            mask = np.isin(x, self._domain)
            x_m = x[mask]
            indexes = np.searchsorted(self._domain, x_m)
            self._range[indexes] = y[mask]

            # Non matching domain, inserting into existing `self.domain`
            # and `self.range`.
            x_nm = x[~mask]
            indexes = np.searchsorted(self._domain, x_nm)
            if indexes.size != 0:
                self._domain = np.insert(self._domain, indexes, x_nm)
                self._range = np.insert(self._range, indexes, y[~mask])

        self._function = None  # Invalidate the underlying continuous function.

    def __contains__(self, x: ArrayLike | slice) -> bool:
        """
        Determine whether the continuous signal contains the specified
        independent domain variable :math:`x`.

        Parameters
        ----------
        x
            Independent domain variable :math:`x`.

        Returns
        -------
        :class:`bool`
            Whether :math:`x` domain value is contained.

        Examples
        --------
        >>> range_ = np.linspace(10, 100, 10)
        >>> signal = Signal(range_)
        >>> 0 in signal
        True
        >>> 0.5 in signal
        True
        >>> 1000 in signal
        False
        """

        return bool(
            np.all(
                np.where(
                    np.logical_and(
                        x >= np.min(self._domain),  # pyright: ignore
                        x <= np.max(self._domain),  # pyright: ignore
                    ),
                    True,
                    False,
                )
            )
        )

    @ndarray_copy_enable(False)
    def __eq__(self, other: object) -> bool:
        """
        Determine whether the continuous signal equals the specified object.

        Parameters
        ----------
        other
            Object to determine for equality with the continuous signal.

        Returns
        -------
        :class:`bool`
            Whether the specified object is equal to the continuous signal.

        Examples
        --------
        >>> range_ = np.linspace(10, 100, 10)
        >>> signal_1 = Signal(range_)
        >>> signal_2 = Signal(range_)
        >>> signal_1 == signal_2
        True
        >>> signal_2[0] = 20
        >>> signal_1 == signal_2
        False
        >>> signal_2[0] = 10
        >>> signal_1 == signal_2
        True
        >>> from colour.algebra import CubicSplineInterpolator
        >>> signal_2.interpolator = CubicSplineInterpolator
        >>> signal_1 == signal_2
        False
        """

        # NOTE: Comparing "interpolator_kwargs" and "extrapolator_kwargs" using
        # their string representation because of presence of NaNs.
        if isinstance(other, Signal):
            return all(
                [
                    np.array_equal(self._domain, other.domain),
                    np.array_equal(self._range, other.range),
                    self._interpolator is other.interpolator,
                    repr(self._interpolator_kwargs) == repr(other.interpolator_kwargs),
                    self._extrapolator is other.extrapolator,
                    repr(self._extrapolator_kwargs) == repr(other.extrapolator_kwargs),
                ]
            )

        return False

    def __ne__(self, other: object) -> bool:
        """
        Determine whether the continuous signal is not equal to the specified
        other object.

        Parameters
        ----------
        other
            Object to determine whether it is not equal to the continuous signal.

        Returns
        -------
        :class:`bool`
            Whether the specified object is not equal to the continuous
            signal.

        Examples
        --------
        >>> range_ = np.linspace(10, 100, 10)
        >>> signal_1 = Signal(range_)
        >>> signal_2 = Signal(range_)
        >>> signal_1 != signal_2
        False
        >>> signal_2[0] = 20
        >>> signal_1 != signal_2
        True
        >>> signal_2[0] = 10
        >>> signal_1 != signal_2
        False
        >>> from colour.algebra import CubicSplineInterpolator
        >>> signal_2.interpolator = CubicSplineInterpolator
        >>> signal_1 != signal_2
        True
        """

        return not (self == other)

    @ndarray_copy_enable(False)
    def _fill_domain_nan(
        self,
        method: Literal["Constant", "Interpolation"] | str = "Interpolation",
        default: Real = 0,
    ) -> None:
        """
        Fill NaNs in the signal's independent domain variable :math:`x` using the
        specified method.

        This private method modifies the domain values in-place, replacing NaN
        values according to the chosen filling strategy.

        Parameters
        ----------
        method
            Filling method to apply. *Interpolation* linearly interpolates
            through the NaN values, while *Constant* replaces NaN values with
            the specified ``default`` value.
        default
            Value to use when ``method`` is *Constant*.
        """
        pass

    @ndarray_copy_enable(False)
    def _fill_range_nan(
        self,
        method: Literal["Constant", "Interpolation"] | str = "Interpolation",
        default: Real = 0,
    ) -> None:
        """
        Fill NaNs in the continuous signal's range variable :math:`y` using
        the specified method.

        Parameters
        ----------
        method
            *Interpolation* method linearly interpolates through the NaNs,
            *Constant* method replaces NaNs with ``default``.
        default
            Value to use with the *Constant* method.

        Returns
        -------
        :class:`colour.continuous.Signal`
            NaNs filled continuous signal in corresponding range :math:`y`
            variable.
        """
        pass

    @ndarray_copy_enable(False)
    def arithmetical_operation(
        self,
        a: ArrayLike | AbstractContinuousFunction,
        operation: Literal["+", "-", "*", "/", "**"],
        in_place: bool = False,
    ) -> AbstractContinuousFunction:
        """
        Perform the specified arithmetical operation with operand :math:`a`.

        The operation can be performed either on a copy of the signal or
        in-place.

        Parameters
        ----------
        a
            Operand :math:`a`. Can be a numeric value, array-like object, or
            another continuous function instance.
        operation
            Arithmetical operation to perform. Supported operations are
            addition (``"+"``), subtraction (``"-"``), multiplication
            (``"*"``), division (``"/"``), and exponentiation (``"**"``).
        in_place
            Whether the operation is performed in-place on the current
            signal instance. Default is ``False``.

        Returns
        -------
        :class:`colour.continuous.Signal`
            Continuous signal after the arithmetical operation. If
            ``in_place`` is ``True``, returns the modified instance;
            otherwise returns a new instance.

        Examples
        --------
        Adding a single *numeric* variable:

        >>> range_ = np.linspace(10, 100, 10)
        >>> signal_1 = Signal(range_)
        >>> print(signal_1)
        [[  0.  10.]
         [  1.  20.]
         [  2.  30.]
         [  3.  40.]
         [  4.  50.]
         [  5.  60.]
         [  6.  70.]
         [  7.  80.]
         [  8.  90.]
         [  9. 100.]]
        >>> print(signal_1.arithmetical_operation(10, "+", True))
        [[  0.  20.]
         [  1.  30.]
         [  2.  40.]
         [  3.  50.]
         [  4.  60.]
         [  5.  70.]
         [  6.  80.]
         [  7.  90.]
         [  8. 100.]
         [  9. 110.]]

        Adding an `ArrayLike` variable:

        >>> a = np.linspace(10, 100, 10)
        >>> print(signal_1.arithmetical_operation(a, "+", True))
        [[  0.  30.]
         [  1.  50.]
         [  2.  70.]
         [  3.  90.]
         [  4. 110.]
         [  5. 130.]
         [  6. 150.]
         [  7. 170.]
         [  8. 190.]
         [  9. 210.]]

        Adding a :class:`colour.continuous.Signal` class:

        >>> signal_2 = Signal(range_)
        >>> print(signal_1.arithmetical_operation(signal_2, "+", True))
        [[  0.  40.]
         [  1.  70.]
         [  2. 100.]
         [  3. 130.]
         [  4. 160.]
         [  5. 190.]
         [  6. 220.]
         [  7. 250.]
         [  8. 280.]
         [  9. 310.]]
        """
        pass

    @staticmethod
    @ndarray_copy_enable(True)
    def signal_unpack_data(
        data: ArrayLike | dict | Series | Signal | ValuesView | None,
        domain: ArrayLike | KeysView | None = None,
        dtype: Type[DTypeFloat] | None = None,
    ) -> tuple:
        """
        Unpack specified data for continuous signal instantiation.

        Parameters
        ----------
        data
            Data to unpack for continuous signal instantiation.
        domain
            Values to initialise the :attr:`colour.continuous.Signal.domain`
            attribute with. If both ``data`` and ``domain`` arguments are
            defined, the latter will be used to initialise the
            :attr:`colour.continuous.Signal.domain` property.
        dtype
            Floating point data type.

        Returns
        -------
        :class:`tuple`
            Independent domain variable :math:`x` and corresponding range
            variable :math:`y` unpacked for continuous signal instantiation.

        Examples
        --------
        Unpacking using implicit *domain*:

        >>> range_ = np.linspace(10, 100, 10)
        >>> domain, range_ = Signal.signal_unpack_data(range_)
        >>> print(domain)
        [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
        >>> print(range_)
        [ 10.  20.  30.  40.  50.  60.  70.  80.  90. 100.]

        Unpacking using explicit *domain*:

        >>> domain = np.arange(100, 1100, 100)
        >>> domain, range = Signal.signal_unpack_data(range_, domain)
        >>> print(domain)
        [ 100.  200.  300.  400.  500.  600.  700.  800.  900. 1000.]
        >>> print(range_)
        [ 10.  20.  30.  40.  50.  60.  70.  80.  90. 100.]

        Unpacking using a *dict*:

        >>> domain, range_ = Signal.signal_unpack_data(dict(zip(domain, range_)))
        >>> print(domain)
        [ 100.  200.  300.  400.  500.  600.  700.  800.  900. 1000.]
        >>> print(range_)
        [ 10.  20.  30.  40.  50.  60.  70.  80.  90. 100.]

        Unpacking using a *Pandas* :class:`pandas.Series`:

        >>> if is_pandas_installed():
        ...     from pandas import Series
        ...
        ...     domain, range = Signal.signal_unpack_data(
        ...         Series(dict(zip(domain, range_)))
        ...     )
        ... # doctest: +ELLIPSIS
        >>> print(domain)  # doctest: +SKIP
        [ 100.  200.  300.  400.  500.  600.  700.  800.  900. 1000.]
        >>> print(range_)  # doctest: +SKIP
        [ 10.  20.  30.  40.  50.  60.  70.  80.  90. 100.]

        Unpacking using a :class:`colour.continuous.Signal` class:

        >>> domain, range_ = Signal.signal_unpack_data(Signal(range_, domain))
        >>> print(domain)
        [ 100.  200.  300.  400.  500.  600.  700.  800.  900. 1000.]
        >>> print(range_)
        [ 10.  20.  30.  40.  50.  60.  70.  80.  90. 100.]
        """
        pass

    def fill_nan(
        self,
        method: Literal["Constant", "Interpolation"] | str = "Interpolation",
        default: Real = 0,
    ) -> Signal:
        """
        Fill NaNs in independent domain variable :math:`x` and corresponding
        range variable :math:`y` using the specified method.

        Parameters
        ----------
        method
            *Interpolation* method linearly interpolates through the NaNs,
            *Constant* method replaces NaNs with ``default``.
        default
            Value to use with the *Constant* method.

        Returns
        -------
        :class:`colour.continuous.Signal`
            Continuous signal with NaN values filled.

        Examples
        --------
        >>> range_ = np.linspace(10, 100, 10)
        >>> signal = Signal(range_)
        >>> signal[3:7] = np.nan
        >>> print(signal)
        [[  0.  10.]
         [  1.  20.]
         [  2.  30.]
         [  3.  nan]
         [  4.  nan]
         [  5.  nan]
         [  6.  nan]
         [  7.  80.]
         [  8.  90.]
         [  9. 100.]]
        >>> print(signal.fill_nan())
        [[  0.  10.]
         [  1.  20.]
         [  2.  30.]
         [  3.  40.]
         [  4.  50.]
         [  5.  60.]
         [  6.  70.]
         [  7.  80.]
         [  8.  90.]
         [  9. 100.]]
        >>> signal[3:7] = np.nan
        >>> print(signal.fill_nan(method="Constant"))
        [[  0.  10.]
         [  1.  20.]
         [  2.  30.]
         [  3.   0.]
         [  4.   0.]
         [  5.   0.]
         [  6.   0.]
         [  7.  80.]
         [  8.  90.]
         [  9. 100.]]
        """
        pass

    @required("Pandas")
    def to_series(self) -> Series:
        """
        Convert the continuous signal to a *Pandas* :class:`pandas.Series`
        class instance.

        Returns
        -------
        :class:`pandas.Series`
            Continuous signal as a *Pandas* :class:`pandas.Series` class
            instance.

        Examples
        --------
        >>> if is_pandas_installed():
        ...     range_ = np.linspace(10, 100, 10)
        ...     signal = Signal(range_)
        ...     print(signal.to_series())  # doctest: +SKIP
        0.0     10.0
        1.0     20.0
        2.0     30.0
        3.0     40.0
        4.0     50.0
        5.0     60.0
        6.0     70.0
        7.0     80.0
        8.0     90.0
        9.0    100.0
        Name: Signal (...), dtype: float64
        """
        pass
