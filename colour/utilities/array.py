"""
Array Utilities
===============

Provide utilities for array manipulation and computational operations.

References
----------
-   :cite:`Castro2014a` : Castro, S. (2014). Numpy: Fastest way of computing
    diagonal for each row of a 2d array. Retrieved August 22, 2014, from
    http://stackoverflow.com/questions/26511401/\
numpy-fastest-way-of-computing-diagonal-for-each-row-of-a-2d-array/\
26517247#26517247
-   :cite:`Yorke2014a` : Yorke, R. (2014). Python: Change format of np.array or
    allow tolerance in in1d function. Retrieved March 27, 2015, from
    http://stackoverflow.com/a/23521245/931625
"""

from __future__ import annotations

import functools
import os
import re
import sys
import typing
from collections.abc import KeysView, ValuesView
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from operator import add, mul, pow, sub, truediv  # noqa: A004
from typing import Union, get_args, get_origin, get_type_hints

import numpy as np

try:
    import array_api_compat as xpc
except ImportError:
    xpc = None

try:
    import array_api_extra as xpx
except ImportError:
    xpx = None

from colour.constants import (
    DTYPE_COMPLEX_DEFAULT,
    DTYPE_FLOAT_DEFAULT,
    DTYPE_INT_DEFAULT,
    EPSILON,
)

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        Callable,
        DType,
        DTypeBoolean,
        DTypeReal,
        Dataclass,
        Generator,
        Literal,
        ModuleType,
        NDArray,
        NDArrayBoolean,
        NDArrayComplex,
        NDArrayFloat,
        NDArrayInt,
        Real,
        Self,
        Sequence,
        Type,
    )

from colour.hints import ArrayLike, DTypeComplex, DTypeFloat, DTypeInt, cast
from colour.utilities import (
    CACHE_REGISTRY,
    as_bool,
    attest,
    int_digest,
    is_caching_enabled,
    optional,
    runtime_warning,
    suppress_warnings,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ARRAY_API_ENABLED",
    "is_array_api_enabled",
    "set_array_api_enabled",
    "array_api_enable",
    "trace_array_namespace",
    "array_namespace",
    "is_numpy_namespace",
    "is_non_ndarray",
    "as_ndarray",
    "xp_asarray",
    "xp_astype",
    "xp_select",
    "xp_interp",
    "xp_trapezoid",
    "xp_average",
    "xp_gradient",
    "xp_resize",
    "xp_nanmean",
    "xp_median",
    "xp_round",
    "xp_radians",
    "xp_degrees",
    "xp_atleast_1d",
    "xp_atleast_2d",
    "xp_sinc",
    "xp_isclose",
    "xp_nan_to_num",
    "xp_create_diagonal",
    "xp_reshape",
    "xp_lstsq",
    "xp_isin",
    "xp_linspace",
    "xp_pad",
    "xp_unique",
    "xp_insert",
    "xp_setxor1d",
    "xp_assert_close",
    "xp_assert_equal",
    "MixinDataclassFields",
    "MixinDataclassIterable",
    "MixinDataclassArray",
    "MixinDataclassArithmetic",
    "as_array",
    "as_int",
    "as_float",
    "as_int_array",
    "as_float_array",
    "as_int_scalar",
    "as_float_scalar",
    "as_complex_array",
    "set_default_int_dtype",
    "set_default_float_dtype",
    "get_domain_range_scale",
    "set_domain_range_scale",
    "domain_range_scale",
    "get_domain_range_scale_metadata",
    "to_domain_1",
    "to_domain_10",
    "to_domain_100",
    "to_domain_degrees",
    "to_domain_int",
    "from_range_1",
    "from_range_10",
    "from_range_100",
    "from_range_degrees",
    "from_range_int",
    "is_ndarray_copy_enabled",
    "set_ndarray_copy_enable",
    "ndarray_copy_enable",
    "ndarray_copy",
    "closest_indexes",
    "closest",
    "interval",
    "is_uniform",
    "in_array",
    "tstack",
    "tsplit",
    "row_as_diagonal",
    "orient",
    "centroid",
    "fill_nan",
    "has_only_nan",
    "ndarray_write",
    "zeros",
    "ones",
    "full",
    "index_along_last_axis",
    "format_array_as_row",
]

ARRAY_API_ENABLED: bool = as_bool(os.environ.get("COLOUR_SCIENCE__ARRAY_API", "False"))
"""
Global variable storing the current *Colour* Array API dispatch enabled state.
"""


def is_array_api_enabled() -> bool:
    """
    Determine whether *Colour* Array API dispatch is enabled.

    The Array API dispatch state is controlled by the global
    *COLOUR_SCIENCE__ARRAY_API* environment variable and can be
    temporarily modified using the :func:`set_array_api_enabled` function
    or the :class:`array_api_enable` context manager.

    Returns
    -------
    :class:`bool`
        Whether *Colour* Array API dispatch is enabled.

    Examples
    --------
    >>> with array_api_enable(False):
    ...     is_array_api_enabled()
    False
    >>> with array_api_enable(True):
    ...     is_array_api_enabled()
    True
    """

    return ARRAY_API_ENABLED


def set_array_api_enabled(enable: bool) -> None:
    """
    Set the *Colour* Array API dispatch enabled state.

    Parameters
    ----------
    enable
        Whether to enable *Colour* Array API dispatch.

    Examples
    --------
    >>> with array_api_enable(True):
    ...     print(is_array_api_enabled())
    ...     set_array_api_enabled(False)
    ...     print(is_array_api_enabled())
    True
    False
    """

    global ARRAY_API_ENABLED  # noqa: PLW0603

    ARRAY_API_ENABLED = enable


class array_api_enable:
    """
    Define a context manager and decorator to temporarily set the *Colour*
    Array API dispatch enabled state.

    Parameters
    ----------
    enable
        Whether to enable or disable *Colour* Array API dispatch.
    """

    def __init__(self, enable: bool) -> None:
        self._enable = enable
        self._previous_state: bool = False

    def __enter__(self) -> Self:
        """Enter the context and set the Array API dispatch state."""

        self._previous_state = is_array_api_enabled()
        set_array_api_enabled(self._enable)

        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context and restore the previous Array API state."""

        set_array_api_enabled(self._previous_state)

    def __call__(self, function: Callable) -> Callable:
        """Decorate and call the specified function with Array API control."""

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return function(*args, **kwargs)

        return wrapper


class trace_array_namespace:
    """
    Define a context manager to trace :func:`array_namespace` calls and
    array type flow through *Colour* functions using :func:`sys.settrace`.

    When active, every function call under ``colour/`` is logged with the
    types of all array arguments (positional and keyword). Return values
    are logged with their types. Calls where multiple array backends
    coexist in the same argument list are flagged as ``MIXED``.

    The trace output is indented to reflect the call stack depth.

    Examples
    --------
    >>> import torch  # doctest: +SKIP
    >>> with array_api_enable(True), trace_array_namespace():
    ...     pass  # doctest: +SKIP
    """

    _ARRAY_TYPES: tuple = (np.ndarray, np.generic)

    def __init__(self) -> None:
        self._depth: int = 0
        self._previous_trace: Any = None

        try:
            import torch  # noqa: PLC0415

            self._ARRAY_TYPES = (*self._ARRAY_TYPES, torch.Tensor)
        except ImportError:
            pass

        try:
            import jax  # noqa: PLC0415

            self._ARRAY_TYPES = (*self._ARRAY_TYPES, jax.Array)
        except ImportError:
            pass

    def _type_label(self, obj: Any) -> str:
        """Return a short type label for the specified object."""

        cls = type(obj)
        module = cls.__module__.split(".")[0]

        if isinstance(obj, np.ndarray):
            return f"ndarray{list(obj.shape)}"

        return (
            f"{module}.{cls.__name__}{list(obj.shape) if hasattr(obj, 'shape') else ''}"
        )

    def _format_args(
        self,
        code: Any,
        local_vars: dict,
    ) -> str:
        """Format function arguments with array type annotations."""

        parts = []
        param_names = list(code.co_varnames[: code.co_argcount])

        for name in param_names:
            if name == "self":
                continue

            value = local_vars.get(name)

            if value is None:
                parts.append(f"{name}: None")
            elif isinstance(value, self._ARRAY_TYPES):
                parts.append(f"{name}: {self._type_label(value)}")
            else:
                parts.append(f"{name}: {type(value).__name__}")

        return ", ".join(parts)

    def _has_mixed_backends(self, local_vars: dict) -> bool:
        """Check whether the local variables contain mixed array backends."""

        backends = set()

        for value in local_vars.values():
            if isinstance(value, self._ARRAY_TYPES):
                if isinstance(value, (np.ndarray, np.generic)):
                    backends.add("numpy")
                else:
                    backends.add(type(value).__module__.split(".")[0])

        return len(backends) > 1

    def _is_colour_frame(self, frame: Any) -> bool:
        """Check whether the specified frame belongs to *Colour*."""

        filename = frame.f_code.co_filename or ""

        return "colour/" in filename and "/site-packages/" not in filename

    def _trace(self, frame: Any, event: str, arg: Any) -> Any:
        """Trace function for :func:`sys.settrace`."""

        if not self._is_colour_frame(frame):
            return self._trace

        if event == "call":
            code = frame.f_code
            name = code.co_name

            if name.startswith("<") or (
                name.startswith("_") and not name.startswith("__")
            ):
                return self._trace

            args_str = self._format_args(code, frame.f_locals)
            mixed = self._has_mixed_backends(frame.f_locals)
            marker = " [MIXED]" if mixed else ""

            indent = "  " * self._depth
            print(f"{indent}{name}({args_str}){marker}")  # noqa: T201

            self._depth += 1

            return self._trace

        if event == "return":
            self._depth = max(0, self._depth - 1)

            if isinstance(arg, self._ARRAY_TYPES):
                indent = "  " * self._depth
                print(f"{indent}-> {self._type_label(arg)}")  # noqa: T201

            return self._trace

        return self._trace

    def __enter__(self) -> Self:
        """Enter the context and install the trace hook."""

        import sys  # noqa: PLC0415

        self._previous_trace = sys.gettrace()
        self._depth = 0
        sys.settrace(self._trace)

        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context and restore the previous trace hook."""

        import sys  # noqa: PLC0415

        sys.settrace(self._previous_trace)


def array_namespace(*arrays: Any) -> ModuleType:
    """
    Return the array namespace for the specified arrays.

    When Array API dispatch is disabled (default), return :mod:`numpy`.
    When enabled, use :func:`array_api_compat.array_namespace` to detect
    the appropriate namespace from the input arrays.

    Parameters
    ----------
    *arrays
        Arrays to determine the namespace from.

    Returns
    -------
    :class:`types.ModuleType`
        Array namespace module.

    Examples
    --------
    >>> array_namespace(np.array([1, 2, 3]))  # doctest: +ELLIPSIS
    <module 'numpy'...>
    """

    if not is_array_api_enabled():
        return np

    arrays = tuple(
        a
        for a in arrays
        if a is not None
        and (
            hasattr(a, "__array_namespace__")
            or isinstance(a, np.ndarray)
            or (xpc is not None and xpc.is_array_api_obj(a))
        )
    )

    if not arrays:
        return np

    # When inputs mix NumPy arrays (e.g., module-level constants) with a
    # non-NumPy backend, promote to the non-NumPy backend.  Only mixed
    # non-NumPy backends (e.g., JAX + CuPy) raise a ``TypeError``.
    non_numpy = tuple(a for a in arrays if not isinstance(a, (np.ndarray, np.generic)))

    if non_numpy:
        arrays = non_numpy

    if xpc is None:  # pragma: no cover
        return np

    return xpc.array_namespace(*arrays)


def is_numpy_namespace(xp: ModuleType) -> bool:
    """
    Determine whether the specified namespace is :mod:`numpy`.

    Parameters
    ----------
    xp
        Namespace module to test.

    Returns
    -------
    :class:`bool`
        Whether the namespace is :mod:`numpy`.

    Examples
    --------
    >>> is_numpy_namespace(np)
    True
    """

    if xp is np:
        return True

    if xpc is not None:
        return xpc.is_numpy_namespace(xp)

    return False


def is_non_ndarray(a: Any) -> bool:
    """
    Determine whether the specified object is a non-*NumPy* array.

    Parameters
    ----------
    a
        Object to test.

    Returns
    -------
    :class:`bool`
        Whether the object is a non-*NumPy* array (e.g., *JAX*, *PyTorch*,
        *CuPy*).

    Examples
    --------
    >>> is_non_ndarray(np.array([1, 2, 3]))
    False
    >>> is_non_ndarray([1, 2, 3])
    False
    """

    if isinstance(a, (np.ndarray, np.generic)):
        return False

    if hasattr(a, "__array_namespace__"):
        return True

    if xpc is not None:
        return xpc.is_array_api_obj(a)

    return False


def as_ndarray(a: Any) -> np.ndarray:
    """
    Convert the specified array :math:`a` to a :class:`numpy.ndarray`.

    This function handles arrays from any backend (*JAX*, *PyTorch*, *CuPy*,
    etc.) by using the *DLPack* protocol when direct conversion is not
    possible, e.g., for device-resident arrays.

    Parameters
    ----------
    a
        Array :math:`a` to convert.

    Returns
    -------
    :class:`numpy.ndarray`
        *NumPy* array.

    Examples
    --------
    >>> as_ndarray(np.array([1, 2, 3]))
    array([1, 2, 3])
    """

    try:
        return np.asarray(a)
    except (TypeError, RuntimeError):
        try:
            return np.from_dlpack(a)
        except (RuntimeError, TypeError):
            return np.from_dlpack(a.detach().cpu())


def xp_asarray(data: ArrayLike, *, xp: ModuleType, like: Any = None) -> NDArrayFloat:
    """
    Convert the specified data to the target namespace.

    When the namespace is :mod:`numpy`, the original data is returned as a
    :class:`numpy.ndarray` without unnecessary copying. For other namespaces,
    the data is converted via ``xp.asarray``, optionally matching the device
    of a reference array *like*.

    Parameters
    ----------
    data
        Data to convert.
    xp
        Target array namespace module.
    like
        Reference array whose device to match (for backends like PyTorch
        that support multiple devices).

    Returns
    -------
    :class:`object`
        Data in the target namespace.

    Examples
    --------
    >>> xp_asarray([1, 2, 3], xp=np)
    array([1, 2, 3])
    """

    if is_numpy_namespace(xp):
        return as_array(data)

    device = getattr(like, "device", None)

    if device is not None and hasattr(device, "type"):
        # PyTorch device (has .type attribute: 'cpu', 'mps', 'cuda')
        try:
            return xp.asarray(data, device=device)
        except TypeError:
            # MPS does not support float64; downcast to float32.
            return xp.asarray(np.asarray(data, dtype=np.float32), device=device)

    return xp.asarray(data)


def xp_astype(a: ArrayLike, dtype: Any, xp: ModuleType | None = None) -> NDArray:
    """
    Array API compatible implementation of :meth:`numpy.ndarray.astype`.

    *NumPy* uses ``a.astype(dtype)`` while the *Array API* standard uses
    ``xp.astype(a, dtype)`` with backend-native dtype objects.

    Parameters
    ----------
    a
        Array to cast.
    dtype
        Target dtype (*NumPy* dtype accepted, automatically translated for
        non-*NumPy* backends).
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`object`
        Cast array.
    """

    xp = array_namespace(a) if xp is None else xp

    if is_numpy_namespace(xp):
        return a.astype(dtype)  # pyright: ignore

    try:
        xp_dtype = getattr(xp, np.dtype(dtype).name, dtype)
    except TypeError:
        # dtype is already a backend-native type (e.g., torch.float64)
        xp_dtype = dtype

    if a.dtype == xp_dtype:  # pyright: ignore
        return a  # pyright: ignore

    # NOTE: ``array_namespace(a)`` is called again to obtain the
    # ``array-api-compat`` wrapped namespace which provides ``astype`` for
    # backends (e.g., *PyTorch*) that lack a module-level ``astype``.
    return array_namespace(a).astype(a, xp_dtype)


def xp_select(
    condlist: Any,
    choicelist: Any,
    *,
    default: Any = 0,
    xp: ModuleType,
) -> NDArrayFloat:
    """
    Array API compatible implementation of :func:`numpy.select`.

    Parameters
    ----------
    condlist
        List of boolean arrays for conditions.
    choicelist
        List of arrays from which output elements are taken.
    default
        Value used when all conditions are ``False``.
    xp
        Array namespace module.

    Returns
    -------
    :class:`object`
        Array with elements from *choicelist* where *condlist* is ``True``.
    """

    if is_numpy_namespace(xp):
        return np.select(condlist, choicelist, default)

    # Find a reference array for device placement.
    like = None
    for item in (*condlist, *choicelist):
        if hasattr(item, "device"):
            like = item
            break

    condlist = [xp_asarray(c, xp=xp, like=like) for c in condlist]
    choicelist = [xp_asarray(c, xp=xp, like=like) for c in choicelist]

    result = xp.full(
        condlist[0].shape,
        fill_value=default,
        dtype=choicelist[0].dtype,
        device=getattr(like, "device", None),
    )

    for condition, choice in zip(reversed(condlist), reversed(choicelist), strict=True):
        result = xp.where(xp_astype(condition, bool, xp), choice, result)

    return result


def xp_interp(
    x: ArrayLike,
    x_data: ArrayLike,
    fp: ArrayLike,
    *,
    xp: ModuleType,
) -> NDArrayFloat:
    """
    Array API compatible implementation of :func:`numpy.interp`.

    Parameters
    ----------
    x
        x-coordinates at which to evaluate the interpolation.
    x_data
        x-coordinates of the data points.
    fp
        y-coordinates of the data points.
    xp
        Array namespace module.

    Returns
    -------
    :class:`object`
        Interpolated values.
    """

    if is_numpy_namespace(xp):
        return np.interp(x, x_data, fp)  # pyright: ignore

    if hasattr(xp, "interp"):
        return xp.interp(x, x_data, fp)

    runtime_warning(
        '"xp_interp" is falling back to "NumPy" for non-"NumPy" '
        "arrays, this will incur a performance penalty due to array "
        "conversion."
    )

    result = np.interp(as_ndarray(x), as_ndarray(x_data), as_ndarray(fp))

    device = getattr(x, "device", None)
    if device is not None and hasattr(device, "type"):
        try:
            return xp.asarray(result, device=device)
        except TypeError:
            return xp.asarray(np.asarray(result, dtype=np.float64), device=device)

    return xp.asarray(result)


def xp_trapezoid(
    y: ArrayLike,
    *,
    x: ArrayLike | None = None,
    dx: float = 1.0,
    axis: int = -1,
    xp: ModuleType,
) -> NDArrayFloat:
    """
    Array API compatible implementation of :func:`numpy.trapezoid`.

    Parameters
    ----------
    y
        y-coordinates of the function values.
    x
        x-coordinates of the function values.
    dx
        Spacing between sample points when *x* is ``None``.
    axis
        Axis along which to integrate.
    xp
        Array namespace module.

    Returns
    -------
    :class:`object`
        Approximation of the integral.
    """

    if is_numpy_namespace(xp):
        return np.trapezoid(y, x=x, dx=dx, axis=axis)  # pyright: ignore

    try:
        if x is not None:
            return xp.trapezoid(y, x=x, axis=axis)

        return xp.trapezoid(y, dx=dx, axis=axis)
    except (TypeError, AttributeError):
        pass

    runtime_warning(
        '"xp_trapezoid" is falling back to "NumPy" for non-"NumPy" '
        "arrays, this will incur a performance penalty due to array "
        "conversion."
    )

    result = np.trapezoid(
        as_ndarray(y), x=as_ndarray(x) if x is not None else None, dx=dx, axis=axis
    )

    return xp.asarray(result)


def xp_average(
    a: ArrayLike,
    *,
    axis: int | None = None,
    weights: ArrayLike | None = None,
    xp: ModuleType,
) -> NDArrayFloat:
    """
    Array API compatible implementation of :func:`numpy.average`.

    Parameters
    ----------
    a
        Array to average.
    axis
        Axis along which to average.
    weights
        Weights associated with the values in *a*.
    xp
        Array namespace module.

    Returns
    -------
    :class:`object`
        Weighted average.
    """

    if is_numpy_namespace(xp):
        return np.average(a, axis=axis, weights=weights)  # pyright: ignore

    if weights is None:
        return xp.mean(a, axis=axis)

    return xp.sum(a * weights, axis=axis) / xp.sum(weights, axis=axis)  # pyright: ignore


def xp_gradient(
    f: ArrayLike, *varargs: Any, xp: ModuleType, axis: Any = None
) -> NDArrayFloat:
    """
    Array API compatible implementation of :func:`numpy.gradient`.

    Parameters
    ----------
    f
        Array of function values.
    *varargs
        Spacing between values.
    xp
        Array namespace module.
    axis
        Axis along which to compute the gradient.

    Returns
    -------
    :class:`object`
        Gradient of *f*.
    """

    if is_numpy_namespace(xp):
        return np.gradient(f, *varargs, axis=axis)

    try:
        result = xp.gradient(f, *varargs, axis=axis)
    except (TypeError, AttributeError):
        pass
    else:
        # Some backends (e.g., torch) return a tuple of tensors.
        if isinstance(result, (tuple, list)) and len(result) == 1:
            return result[0]
        return result  # pyright: ignore

    runtime_warning(
        '"xp_gradient" is falling back to "NumPy" for non-"NumPy" '
        "arrays, this will incur a performance penalty due to array "
        "conversion."
    )

    result = np.gradient(as_ndarray(f), *(as_ndarray(v) for v in varargs), axis=axis)

    if isinstance(result, list):
        return [xp.asarray(r) for r in result]  # pyright: ignore

    return xp.asarray(result)


def xp_resize(a: ArrayLike, new_shape: Any, *, xp: ModuleType) -> NDArray:
    """
    Array API compatible implementation of :func:`numpy.resize`.

    Parameters
    ----------
    a
        Array to resize.
    new_shape
        Shape of the resized array.
    xp
        Array namespace module.

    Returns
    -------
    :class:`object`
        Resized array.
    """

    if is_numpy_namespace(xp):
        return np.resize(a, new_shape)

    if hasattr(xp, "resize"):
        return xp.resize(a, new_shape)

    # Native implementation via tile + slice for backends without resize.
    a = xp.asarray(a)
    raveled = xp.reshape(a, (-1,))
    target_size = 1
    for shape in new_shape if isinstance(new_shape, tuple) else (new_shape,):
        target_size *= shape

    if raveled.shape[0] == 0:
        return xp.zeros(new_shape, dtype=a.dtype)  # pyright: ignore

    repeats = (target_size + raveled.shape[0] - 1) // raveled.shape[0]
    tiled = xp.tile(raveled, (repeats,))[:target_size]

    return xp.reshape(tiled, new_shape)


def xp_nanmean(
    a: ArrayLike,
    *,
    axis: int | None = None,
    xp: ModuleType,
) -> NDArrayFloat:
    """
    Array API compatible implementation of :func:`numpy.nanmean`.

    Parameters
    ----------
    a
        Array containing numbers whose NaN-aware mean is desired.
    axis
        Axis along which the mean is computed.
    xp
        Array namespace module.

    Returns
    -------
    :class:`object`
        NaN-aware mean.
    """

    if is_numpy_namespace(xp):
        return np.nanmean(a, axis=axis)  # pyright: ignore

    mask = xp.isnan(a)
    zeroed = xp.where(mask, xp.asarray(0.0, dtype=a.dtype), a)  # pyright: ignore
    count = xp.sum(
        xp_astype(~mask, a.dtype, xp),  # pyright: ignore
        axis=axis,
    )

    return xp.sum(zeroed, axis=axis) / count


def xp_median(
    a: ArrayLike,
    *,
    axis: int | None = None,
    xp: ModuleType,
) -> NDArrayFloat:
    """
    Array API compatible implementation of :func:`numpy.median`.

    Parameters
    ----------
    a
        Array whose median is desired.
    axis
        Axis along which the median is computed.
    xp
        Array namespace module.

    Returns
    -------
    :class:`object`
        Median value(s).
    """

    if is_numpy_namespace(xp):
        return np.median(a, axis=axis)  # pyright: ignore

    runtime_warning(
        '"xp_median" is falling back to "NumPy" for non-"NumPy" '
        "arrays, this will incur a performance penalty due to array "
        "conversion."
    )

    result = np.median(as_ndarray(a), axis=axis)

    return xp.asarray(result)


def xp_round(
    a: ArrayLike,
    *,
    decimals: int = 0,
    xp: ModuleType,
) -> NDArrayFloat:
    """
    Array API compatible implementation of :func:`numpy.round` with *decimals*.

    The Array API standard ``xp.round`` does not accept a *decimals*
    parameter. This helper uses the backend's native ``round`` when it
    supports *decimals* (JAX, CuPy), otherwise falls back to a
    multiply-round-divide pattern using the standard ``xp.round``.

    Parameters
    ----------
    a
        Array to round.
    decimals
        Number of decimal places.
    xp
        Array namespace module.

    Returns
    -------
    :class:`object`
        Rounded array.
    """

    if is_numpy_namespace(xp):
        return np.round(a, decimals)  # pyright: ignore

    try:
        return xp.round(a, decimals)
    except TypeError:
        factor = 10**decimals
        return xp.round(a * factor) / factor


def xp_radians(a: ArrayLike) -> NDArrayFloat:
    """
    Array API compatible implementation of :func:`numpy.radians`.

    Parameters
    ----------
    a
        Angle in degrees.

    Returns
    -------
    :class:`object`
        Angle in radians.
    """

    return as_float_array(a) * (np.pi / 180)


def xp_degrees(a: ArrayLike) -> NDArrayFloat:
    """
    Array API compatible implementation of :func:`numpy.degrees`.

    Parameters
    ----------
    a
        Angle in radians.

    Returns
    -------
    :class:`object`
        Angle in degrees.
    """

    return as_float_array(a) * (180 / np.pi)


# NOTE: The following wrappers around ``array_api_extra`` functions exist for
# typing purposes. ``array_api_extra`` returns a generic ``Array`` type that
# ``pyright`` cannot reconcile with ``NDArrayFloat`` and other *Colour* type
# aliases. These thin wrappers provide properly annotated return types,
# avoiding ``cast()`` noise at every call site. They can be removed once
# ``array-api-typing`` is released and ``array_api_extra`` adopts it.


def xp_atleast_1d(a: ArrayLike, xp: ModuleType | None = None) -> NDArrayFloat:
    """
    Array API compatible implementation of :func:`numpy.atleast_1d`.

    Parameters
    ----------
    a
        Array to ensure is at least 1-D.
    xp
        Array namespace module.

    Returns
    -------
    :class:`numpy.ndarray`
        Array with ``ndim >= 1``.
    """

    if xp is None or is_numpy_namespace(xp):
        return np.atleast_1d(a)

    return xpx.atleast_nd(a, ndim=1, xp=xp)  # pyright: ignore


def xp_atleast_2d(a: ArrayLike, xp: ModuleType | None = None) -> NDArrayFloat:
    """
    Array API compatible implementation of :func:`numpy.atleast_2d`.

    Parameters
    ----------
    a
        Array to ensure is at least 2-D.
    xp
        Array namespace module.

    Returns
    -------
    :class:`numpy.ndarray`
        Array with ``ndim >= 2``.
    """

    if xp is None or is_numpy_namespace(xp):
        return np.atleast_2d(a)

    if isinstance(a, (np.ndarray, np.generic)):
        a = xp_asarray(a, xp=xp)

    return xpx.atleast_nd(a, ndim=2, xp=xp)  # pyright: ignore


def xp_sinc(a: ArrayLike, xp: ModuleType | None = None) -> NDArrayFloat:
    """
    Array API compatible implementation of :func:`numpy.sinc`.

    Parameters
    ----------
    a
        Array of values.
    xp
        Array namespace module.

    Returns
    -------
    :class:`numpy.ndarray`
        Sinc of *a*.
    """

    if xp is None or is_numpy_namespace(xp):
        return np.sinc(a)  # pyright: ignore

    return xpx.sinc(a, xp=xp)  # pyright: ignore


def xp_isclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    xp: ModuleType | None = None,
) -> NDArrayBoolean:
    """
    Array API compatible implementation of :func:`numpy.isclose`.

    Parameters
    ----------
    a
        First array.
    b
        Second array.
    rtol
        Relative tolerance.
    atol
        Absolute tolerance.
    xp
        Array namespace module.

    Returns
    -------
    :class:`numpy.ndarray`
        Boolean array of element-wise comparisons.
    """

    if xp is None or is_numpy_namespace(xp):
        return np.isclose(a, b, rtol=rtol, atol=atol)

    if isinstance(a, (np.ndarray, np.generic)):
        a = xp_asarray(a, xp=xp)
    if isinstance(b, (np.ndarray, np.generic)):
        b = xp_asarray(b, xp=xp)

    return xpx.isclose(a, b, rtol=rtol, atol=atol, xp=xp)  # pyright: ignore


def xp_nan_to_num(
    a: ArrayLike,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
    xp: ModuleType | None = None,
) -> NDArrayFloat:
    """
    Array API compatible implementation of :func:`numpy.nan_to_num`.

    Parameters
    ----------
    a
        Array to process.
    nan
        Value to replace NaN entries.
    posinf
        Value to replace positive infinity entries.
    neginf
        Value to replace negative infinity entries.
    xp
        Array namespace module.

    Returns
    -------
    :class:`numpy.ndarray`
        Array with NaN/Inf replaced.
    """

    if xp is None or is_numpy_namespace(xp):
        return np.nan_to_num(a, nan=nan, posinf=posinf, neginf=neginf)

    xp = array_namespace(a) if xp is None else xp

    result = xpx.nan_to_num(a, fill_value=nan, xp=xp)  # pyright: ignore

    if posinf is not None:
        result = xp.where(xp.isinf(a) & (a > 0), posinf, result)  # pyright: ignore

    if neginf is not None:
        result = xp.where(xp.isinf(a) & (a < 0), neginf, result)  # pyright: ignore

    return result  # pyright: ignore


def xp_create_diagonal(a: ArrayLike, xp: ModuleType | None = None) -> NDArrayFloat:
    """
    Array API compatible implementation of :func:`numpy.diagflat`.

    Parameters
    ----------
    a
        1-D array of diagonal values.
    xp
        Array namespace module.

    Returns
    -------
    :class:`numpy.ndarray`
        2-D array with *a* on the diagonal.
    """

    if xp is None or is_numpy_namespace(xp):
        return np.diagflat(a)

    return xpx.create_diagonal(a, xp=xp)  # pyright: ignore


def xp_reshape(a: ArrayLike, shape: Any, *, xp: ModuleType) -> NDArray:
    """
    Array API compatible implementation of :func:`numpy.reshape`.

    This typed wrapper exists because ``xp`` is typed as :class:`Any`, so
    ``xp.reshape(...)`` returns :class:`Any`. When the result is assigned
    back to a variable that was originally a function parameter (e.g.,
    ``a: ArrayLike``), *Pyright* reverts the variable to its declared type
    instead of narrowing it. This wrapper declares ``-> NDArrayFloat`` so
    that *Pyright* can track the narrowed type through reassignments.

    Parameters
    ----------
    a
        Input array.
    shape
        New shape.
    xp
        Array namespace module.

    Returns
    -------
    :class:`numpy.ndarray`
        Reshaped array.
    """

    return xp.reshape(a, shape)


def xp_lstsq(
    a: ArrayLike,
    b: ArrayLike,
    rcond: float | None = None,
    xp: ModuleType | None = None,
) -> NDArrayFloat:
    """
    Array API compatible implementation of :func:`numpy.linalg.lstsq`.

    Returns only the least-squares solution (the first element of the tuple
    returned by :func:`numpy.linalg.lstsq`). Backends that provide
    ``xp.linalg.lstsq`` (e.g., *JAX*, *PyTorch*) are used natively; others
    fall back to *NumPy* with a `ColourRuntimeWarning`.

    Parameters
    ----------
    a
        Coefficient matrix.
    b
        Ordinate values.
    rcond
        Cut-off ratio for small singular values (passed to *NumPy* only).
    xp
        Array namespace module.

    Returns
    -------
    :class:`numpy.ndarray`
        Least-squares solution.
    """

    if xp is None or is_numpy_namespace(xp):
        return np.linalg.lstsq(a, b, rcond=rcond)[0]  # pyright: ignore

    if hasattr(xp, "linalg") and hasattr(xp.linalg, "lstsq"):
        return xp.linalg.lstsq(a, b)[0]

    runtime_warning(
        '"xp_lstsq" is falling back to "NumPy" for non-"NumPy" '
        "arrays, this will incur a performance penalty due to array "
        "conversion."
    )

    result = np.linalg.lstsq(np.asarray(a), np.asarray(b), rcond=rcond)[0]

    return xp.asarray(result)


def xp_isin(
    element: ArrayLike,
    test_elements: ArrayLike,
    *,
    xp: ModuleType,
) -> NDArrayBoolean:
    """
    Array API compatible implementation of :func:`numpy.isin`.

    Use the backend's native ``isin`` when available (JAX, CuPy),
    otherwise fall back to NumPy.

    Parameters
    ----------
    element
        Input array.
    test_elements
        Values against which to test each element of *element*.
    xp
        Array namespace module.

    Returns
    -------
    :class:`object`
        Boolean array of the same shape as *element*.
    """

    if is_numpy_namespace(xp):
        return np.isin(element, test_elements)

    if hasattr(xp, "isin"):
        return xp.isin(element, test_elements)

    runtime_warning(
        '"xp_isin" is falling back to "NumPy" for non-"NumPy" '
        "arrays, this will incur a performance penalty due to array "
        "conversion."
    )

    result = np.isin(np.asarray(element), np.asarray(test_elements))

    return xp.asarray(result)


def xp_linspace(
    start: ArrayLike,
    stop: ArrayLike,
    *,
    num: int = 50,
    xp: ModuleType,
    **kwargs: Any,
) -> NDArrayFloat | tuple[NDArrayFloat, float]:
    """
    Array API compatible implementation of :func:`numpy.linspace` with
    extra keyword arguments such as *retstep* and *dtype*.

    The Array API standard ``xp.linspace`` does not accept *retstep*.
    This helper tries the backend's native ``linspace`` first, falling
    back to NumPy if the keyword is unsupported.

    Parameters
    ----------
    start
        Start of the interval.
    stop
        End of the interval.
    num
        Number of samples.
    xp
        Array namespace module.
    **kwargs
        Extra keyword arguments (e.g., ``retstep``, ``dtype``).

    Returns
    -------
    :class:`object`
        Array of evenly spaced values (and step size if *retstep=True*).
    """

    if is_numpy_namespace(xp):
        return np.linspace(start, stop, num, **kwargs)  # pyright: ignore

    try:
        return xp.linspace(start, stop, num, **kwargs)
    except TypeError:
        runtime_warning(
            '"xp_linspace" is falling back to "NumPy" for non-"NumPy" '
            "arrays, this will incur a performance penalty due to array "
            "conversion."
        )

        result = np.linspace(start, stop, num, **kwargs)  # pyright: ignore

        if isinstance(result, tuple):
            return xp.asarray(result[0]), result[1]

        return xp.asarray(result)


def xp_pad(
    a: ArrayLike, pad_width: Any, *args: Any, xp: ModuleType, **kwargs: Any
) -> NDArray:
    """
    Array API compatible implementation of :func:`numpy.pad`.

    Use the backend's native ``pad`` when available (JAX, CuPy),
    otherwise fall back to NumPy.

    Parameters
    ----------
    a
        Array to pad.
    pad_width
        Number of values padded to the edges of each axis.
    *args
        Positional arguments passed to the padding function.
    xp
        Array namespace module.
    **kwargs
        Keyword arguments passed to the padding function.

    Returns
    -------
    :class:`object`
        Padded array.
    """

    if is_numpy_namespace(xp):
        return np.pad(a, pad_width, *args, **kwargs)

    if hasattr(xp, "pad"):
        return xp.pad(a, pad_width, *args, **kwargs)

    runtime_warning(
        '"xp_pad" is falling back to "NumPy" for non-"NumPy" '
        "arrays, this will incur a performance penalty due to array "
        "conversion."
    )

    result = np.pad(np.asarray(a), pad_width, *args, **kwargs)

    return xp.asarray(result)


def xp_unique(
    a: ArrayLike, *, xp: ModuleType, **kwargs: Any
) -> NDArray | tuple[NDArray, ...]:
    """
    Array API compatible implementation of :func:`numpy.unique` with
    extra keyword arguments such as *return_index* and *axis*.

    The Array API standard only provides ``xp.unique_values`` and
    related functions without *return_index* or *axis* support.
    This helper tries the backend's native ``unique`` first, falling
    back to NumPy if the keywords are unsupported.

    Parameters
    ----------
    a
        Input array.
    xp
        Array namespace module.
    **kwargs
        Extra keyword arguments (e.g., ``return_index``, ``axis``).

    Returns
    -------
    :class:`object`
        Unique values (and optional indices).
    """

    if is_numpy_namespace(xp):
        return np.unique(a, **kwargs)

    if hasattr(xp, "unique"):
        try:
            return xp.unique(a, **kwargs)
        except TypeError:
            pass

    runtime_warning(
        '"xp_unique" is falling back to "NumPy" for non-"NumPy" '
        "arrays, this will incur a performance penalty due to array "
        "conversion."
    )

    result = np.unique(np.asarray(a), **kwargs)

    if isinstance(result, tuple):
        return tuple(xp.asarray(r) for r in result)

    return xp.asarray(result)


def xp_insert(
    a: ArrayLike,
    indices: ArrayLike,
    values: ArrayLike,
    *,
    xp: ModuleType,
) -> NDArray:
    """
    Array API compatible implementation of :func:`numpy.insert` for 1-D
    sorted indices.

    Parameters
    ----------
    a
        Array to insert into.
    indices
        Indices before which to insert *values*.
    values
        Values to insert.
    xp
        Array namespace module.

    Returns
    -------
    :class:`numpy.ndarray`
        Array with *values* inserted.
    """

    if is_numpy_namespace(xp):
        return np.insert(a, indices, values)  # pyright: ignore

    indices = xp.asarray(indices)
    values = xp.asarray(values)

    parts = []
    prev = 0
    for i in range(indices.shape[0]):  # pyright: ignore
        idx = int(indices[i])  # pyright: ignore
        parts.append(a[prev:idx])  # pyright: ignore
        parts.append(xp_reshape(values[i : i + 1], (1,), xp=xp))  # pyright: ignore
        prev = idx
    parts.append(a[prev:])  # pyright: ignore

    return xp.concat(parts)


def xp_setxor1d(
    a: ArrayLike,
    b: ArrayLike,
    *,
    xp: ModuleType,
) -> NDArray:
    """
    Array API compatible implementation of :func:`numpy.setxor1d`.

    Return sorted unique values that are in only one of the two input arrays.

    Parameters
    ----------
    a
        First array.
    b
        Second array.
    xp
        Array namespace module.

    Returns
    -------
    :class:`numpy.ndarray`
        Sorted symmetric difference.
    """

    if is_numpy_namespace(xp):
        return np.setxor1d(a, b)

    a_in_b = (
        xpx.isin(a, b)  # pyright: ignore
        if xpx is not None
        else np.isin(as_ndarray(a), as_ndarray(b))
    )
    b_in_a = (
        xpx.isin(b, a)  # pyright: ignore
        if xpx is not None
        else np.isin(as_ndarray(b), as_ndarray(a))
    )

    a_only = a[~xp.asarray(a_in_b)]  # pyright: ignore
    b_only = b[~xp.asarray(b_in_a)]  # pyright: ignore

    result = xp.sort(xp.concat([a_only, b_only]))

    return result[0] if isinstance(result, tuple) else result


def xp_assert_close(
    actual: ArrayLike,
    desired: ArrayLike,
    atol: float = 1e-7,
    rtol: float = 1e-7,
    err_msg: str = "",
) -> None:
    """
    Array API compatible version of :func:`numpy.testing.assert_allclose`.

    Both arrays are converted to *NumPy* via :func:`as_ndarray` before
    comparison.

    Parameters
    ----------
    actual
        Array produced by the tested function.
    desired
        Expected array.
    atol
        Absolute tolerance.
    rtol
        Relative tolerance.
    err_msg
        Error message to display on failure.
    """

    np.testing.assert_allclose(
        as_ndarray(actual),
        as_ndarray(desired),
        atol=atol,
        rtol=rtol,
        err_msg=err_msg,
    )


def xp_assert_equal(
    actual: ArrayLike,
    desired: ArrayLike,
    err_msg: str = "",
) -> None:
    """
    Array API compatible version of :func:`numpy.testing.assert_array_equal`.

    Both arrays are converted to *NumPy* via :func:`as_ndarray` before
    comparison.

    Parameters
    ----------
    actual
        Array produced by the tested function.
    desired
        Expected array.
    err_msg
        Error message to display on failure.
    """

    np.testing.assert_array_equal(
        as_ndarray(actual),
        as_ndarray(desired),
        err_msg=err_msg,
    )


class MixinDataclassFields:
    """
    Provide fields introspection for :class:`dataclass`-like classes.

    This mixin extends dataclass functionality to enable introspection
    capabilities, allowing programmatic access to field metadata and
    properties.

    Attributes
    ----------
    -   :attr:`~colour.utilities.MixinDataclassFields.fields`
    """

    @property
    def fields(self) -> tuple:
        """
        Getter for the fields of the :class:`dataclass`-like class.

        Returns
        -------
        :class:`tuple`
            :class:`dataclass`-like class fields.
        """

        return fields(self)  # pyright: ignore


class MixinDataclassIterable(MixinDataclassFields):
    """
    Provide iteration capabilities over :class:`dataclass`-like classes.

    This mixin extends dataclass functionality to enable dictionary-like
    iteration over fields, allowing access to field names, values, and
    name-value pairs through standard iteration protocols.

    Attributes
    ----------
    -   :attr:`~colour.utilities.MixinDataclassIterable.keys`
    -   :attr:`~colour.utilities.MixinDataclassIterable.values`
    -   :attr:`~colour.utilities.MixinDataclassIterable.items`

    Methods
    -------
    -   :meth:`~colour.utilities.MixinDataclassIterable.__iter__`

    Notes
    -----
    -   The :class:`colour.utilities.MixinDataclassIterable` class inherits
        the methods from the following class:

        -   :class:`colour.utilities.MixinDataclassFields`
    """

    @property
    def keys(self) -> tuple:
        """
        Getter for the :class:`dataclass`-like class keys, i.e., the field
        names.

        Returns
        -------
        :class:`tuple`
            :class:`dataclass`-like class keys.
        """

        return tuple(field for field, _value in self)

    @property
    def values(self) -> tuple:
        """
        Getter for the :class:`dataclass`-like class field values.

        Returns
        -------
        :class:`tuple`
            :class:`dataclass`-like class field values.
        """

        return tuple(value for _field, value in self)

    @property
    def items(self) -> tuple:
        """
        Getter for the :class:`dataclass`-like class items, i.e., the field
        names and values.

        Returns
        -------
        :class:`tuple`
            :class:`dataclass`-like class items.
        """

        return tuple((field, value) for field, value in self)

    def __iter__(self) -> Generator:
        """
        Yield the :class:`dataclass`-like class fields.

        Yields
        ------
        Generator
            :class:`dataclass`-like class field generator.
        """

        yield from {
            field.name: getattr(self, field.name) for field in self.fields
        }.items()


class MixinDataclassArray(MixinDataclassIterable):
    """
    Provide conversion methods for :class:`dataclass`-like classes to
    :class:`numpy.ndarray` objects.

    This mixin extends dataclass functionality to enable seamless conversion
    to NumPy arrays, facilitating numerical operations on structured data.

    Methods
    -------
    -   :meth:`~colour.utilities.MixinDataclassArray.__array__`

    Notes
    -----
    -   The :class:`colour.utilities.MixinDataclassArray` class
        inherits the methods from the following classes:

        -   :class:`colour.utilities.MixinDataclassIterable`
        -   :class:`colour.utilities.MixinDataclassFields`
    """

    def __array__(
        self, dtype: Type[DTypeReal] | None = None, copy: bool = True
    ) -> NDArray:
        """
        Implement support for :class:`dataclass`-like class conversion to
        :class:`numpy.ndarray` class.

        A field set to *None* will be filled with `np.nan` according to the
        shape of the first field not set with *None*.

        Parameters
        ----------
        dtype
            :class:`numpy.dtype` to use for conversion to `np.ndarray`,
            default to the :class:`numpy.dtype` defined by
            :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.
        copy
            Whether to return a copy of the underlying data, will always be
            `True`, irrespective of the parameter value.

        Returns
        -------
        :class:`numpy.ndarray`
            :class:`dataclass`-like class converted to
            :class:`numpy.ndarray`.
        """

        dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

        default = None
        for _field, value in self:
            if value is not None:
                default = full(as_float_array(value).shape, np.nan)
                break

        return tstack(
            cast(
                "ArrayLike",
                [
                    as_ndarray(value) if value is not None else default
                    for value in self.values
                ],
            ),
            dtype=dtype,
        )


class MixinDataclassArithmetic(MixinDataclassArray):
    """
    Provide mathematical operations for :class:`dataclass`-like classes.

    This mixin extends dataclass functionality to enable arithmetic
    operations, facilitating mathematical computations on dataclass instances
    containing array-like data.

    Methods
    -------
    -   :meth:`~colour.utilities.MixinDataclassArray.__iadd__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__add__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__isub__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__sub__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__imul__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__mul__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__idiv__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__div__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__ipow__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__pow__`
    -   :meth:`~colour.utilities.MixinDataclassArray.arithmetical_operation`

    Notes
    -----
    -   The :class:`colour.utilities.MixinDataclassArithmetic` class inherits
        the methods from the following classes:

        -   :class:`colour.utilities.MixinDataclassArray`
        -   :class:`colour.utilities.MixinDataclassIterable`
        -   :class:`colour.utilities.MixinDataclassFields`
    """

    def __add__(self, a: Any) -> Self:
        """
        Implement support for addition.

        Parameters
        ----------
        a
            Variable :math:`a` to add.

        Returns
        -------
        :class:`dataclass`
            Variable added :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "+")

    def __iadd__(self, a: Any) -> Self:
        """
        Implement support for in-place addition.

        Parameters
        ----------
        a
            Variable :math:`a` to add in-place.

        Returns
        -------
        :class:`dataclass`
            In-place variable added :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "+", True)

    def __sub__(self, a: Any) -> Self:
        """
        Implement support for subtraction.

        Parameters
        ----------
        a
            Variable :math:`a` to subtract.

        Returns
        -------
        :class:`dataclass`
            Variable subtracted :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "-")

    def __isub__(self, a: Any) -> Self:
        """
        Implement support for in-place subtraction.

        Parameters
        ----------
        a
            Variable :math:`a` to subtract in-place.

        Returns
        -------
        :class:`dataclass`
            In-place variable subtracted :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "-", True)

    def __mul__(self, a: Any) -> Self:
        """
        Implement support for multiplication.

        Parameters
        ----------
        a
            Variable :math:`a` to multiply by.

        Returns
        -------
        :class:`dataclass`
            Variable multiplied :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "*")

    def __imul__(self, a: Any) -> Self:
        """
        Implement support for in-place multiplication.

        Parameters
        ----------
        a
            Variable :math:`a` to multiply by in-place.

        Returns
        -------
        :class:`dataclass`
            In-place variable multiplied :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "*", True)

    def __div__(self, a: Any) -> Self:
        """
        Implement support for division.

        Parameters
        ----------
        a
            Variable :math:`a` to divide by.

        Returns
        -------
        :class:`dataclass`
            Variable divided :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "/")

    def __idiv__(self, a: Any) -> Self:
        """
        Implement support for in-place division.

        Parameters
        ----------
        a
            Variable :math:`a` to divide by in-place.

        Returns
        -------
        :class:`dataclass`
            In-place variable divided :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "/", True)

    __itruediv__ = __idiv__
    __truediv__ = __div__

    def __pow__(self, a: Any) -> Self:
        """
        Implement support for exponentiation.

        Parameters
        ----------
        a
            Variable :math:`a` to exponentiate by.

        Returns
        -------
        :class:`dataclass`
            Variable exponentiated :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "**")

    def __ipow__(self, a: Any) -> Self:
        """
        Implement support for in-place exponentiation.

        Parameters
        ----------
        a
            Variable :math:`a` to exponentiate by in-place.

        Returns
        -------
        :class:`dataclass`
            In-place variable exponentiated :class:`dataclass`-like
            class.
        """

        return self.arithmetical_operation(a, "**", True)

    def arithmetical_operation(
        self, a: Any, operation: str, in_place: bool = False
    ) -> Dataclass:
        """
        Perform the specified arithmetical operation with the :math:`a`
        operand on the :class:`dataclass`-like class.

        Parameters
        ----------
        a
            Operand.
        operation
            Operation to perform.
        in_place
            Operation happens in place.

        Returns
        -------
        :class:`dataclass`
            :class:`dataclass`-like class with the arithmetical operation
            performed.
        """

        callable_operation = {
            "+": add,
            "-": sub,
            "*": mul,
            "/": truediv,
            "**": pow,
        }[operation]

        if is_dataclass(a):
            a = as_float_array(a)  # pyright: ignore

        values = tsplit(callable_operation(as_float_array(self), a))
        field_values = {field: values[i] for i, field in enumerate(self.keys)}
        field_values.update({field: None for field, value in self if value is None})

        dataclass = replace(self, **field_values)  # pyright: ignore

        if in_place:
            for field in self.keys:
                setattr(self, field, getattr(dataclass, field))

            return self

        return dataclass


# NOTE : The following messages are pre-generated for performance reasons.
_ASSERTION_MESSAGE_DTYPE_INT = (
    f'"dtype" must be one of the following types: "{DTypeInt.__args__}"'
)

_ASSERTION_MESSAGE_DTYPE_FLOAT = (
    f'"dtype" must be one of the following types: "{DTypeFloat.__args__}"'
)

_ASSERTION_MESSAGE_DTYPE_COMPLEX = (
    f'"dtype" must be one of the following types: "{DTypeComplex.__args__}"'
)


def as_array(
    a: ArrayLike | KeysView | ValuesView,
    dtype: Type[DType] | None = None,
) -> NDArray:
    """
    Convert the specified variable :math:`a` to :class:`numpy.ndarray` using
    the specified :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Variable :math:`a` converted to :class:`numpy.ndarray`.

    Examples
    --------
    >>> as_array([1, 2, 3])  # doctest: +ELLIPSIS
    array([1, 2, 3]...)
    >>> as_array([1, 2, 3], dtype=DTYPE_FLOAT_DEFAULT)
    array([1., 2., 3.])
    """

    # TODO: Remove when https://github.com/numpy/numpy/issues/5718 is
    # addressed.
    if isinstance(a, (KeysView, ValuesView)):
        a = list(a)

    if is_array_api_enabled():
        # When ``a`` is a list/tuple of non-NumPy arrays, resolve the
        # namespace from the first element and use ``xp.stack`` since
        # ``xp.asarray(list)`` would fall back to NumPy.
        if isinstance(a, list) and len(a) > 0 and is_non_ndarray(a[0]):
            xp = array_namespace(a[0])

            if dtype is not None:
                dtype = getattr(xp, np.dtype(dtype).name, dtype)

            return xp.stack([xp.asarray(x) for x in a])

        xp = array_namespace(a)

        if dtype is not None and not is_numpy_namespace(xp):
            dtype = getattr(xp, np.dtype(dtype).name, dtype)

        return xp.asarray(a, dtype=dtype)

    return np.asarray(a, dtype)


@typing.overload
def as_int(a: float | DTypeFloat, dtype: Type[DTypeInt] | None = None) -> DTypeInt: ...
@typing.overload
def as_int(
    a: NDArray | Sequence[int], dtype: Type[DTypeInt] | None = None
) -> NDArrayInt: ...
@typing.overload
def as_int(
    a: ArrayLike, dtype: Type[DTypeInt] | None = None
) -> DTypeInt | NDArrayInt: ...
def as_int(a: ArrayLike, dtype: Type[DTypeInt] | None = None) -> DTypeInt | NDArrayInt:
    """
    Convert the specified variable :math:`a` to :class:`numpy.integer` using
    the specified :class:`numpy.dtype`.

    The function converts variable :math:`a` to an integer type. If variable
    :math:`a` is not a scalar or 0-dimensional array, it is converted to
    :class:`numpy.ndarray`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_INT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Variable :math:`a` converted to :class:`numpy.integer`.

    Examples
    --------
    >>> as_int(np.array(1))
    np.int64(1)
    >>> as_int(np.array([1]))  # doctest: +SKIP
    array([1])
    >>> as_int(np.arange(10))  # doctest: +SKIP
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]...)
    """

    dtype = optional(dtype, DTYPE_INT_DEFAULT)

    attest(dtype in DTypeInt.__args__, _ASSERTION_MESSAGE_DTYPE_INT)

    return dtype(a)  # pyright: ignore


@typing.overload
def as_float(
    a: float | DTypeFloat, dtype: Type[DTypeFloat] | None = None
) -> DTypeFloat: ...
@typing.overload
def as_float(
    a: NDArray | Sequence[float], dtype: Type[DTypeFloat] | None = None
) -> NDArrayFloat: ...
@typing.overload
def as_float(
    a: ArrayLike, dtype: Type[DTypeFloat] | None = None
) -> DTypeFloat | NDArrayFloat: ...
def as_float(
    a: ArrayLike, dtype: Type[DTypeFloat] | None = None
) -> DTypeFloat | NDArrayFloat:
    """
    Convert the specified variable :math:`a` to :class:`numpy.floating` using
    the specified :class:`numpy.dtype`.

    If variable :math:`a` is not a scalar or 0-dimensional, it is converted
    to :class:`numpy.ndarray`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Variable :math:`a` converted to :class:`numpy.floating`.

    Examples
    --------
    >>> as_float(np.array(1))
    np.float64(1.0)
    >>> as_float(np.array([1]))
    array([1.])
    >>> as_float(np.arange(10))
    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    attest(dtype in DTypeFloat.__args__, _ASSERTION_MESSAGE_DTYPE_FLOAT)

    if is_array_api_enabled() and not isinstance(a, np.ndarray):
        return as_float_array(a, dtype)

    # NOTE: "np.float64" reduces dimensionality:
    # >>> np.int64(np.array([[1]]))
    # array([[1]])
    # >>> np.float64(np.array([[1]]))
    # 1.0
    # See for more information https://github.com/numpy/numpy/issues/24283
    if isinstance(a, np.ndarray) and a.size == 1 and a.ndim != 0:
        return as_float_array(a, dtype)

    return dtype(a)  # pyright: ignore


def as_int_array(a: ArrayLike, dtype: Type[DTypeInt] | None = None) -> NDArrayInt:
    """
    Convert the specified variable :math:`a` to :class:`numpy.ndarray` using
    the specified integer :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_INT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Variable :math:`a` converted to integer :class:`numpy.ndarray`.

    Examples
    --------
    >>> as_int_array([1.0, 2.0, 3.0])  # doctest: +ELLIPSIS
    array([1, 2, 3]...)
    """

    dtype = optional(dtype, DTYPE_INT_DEFAULT)

    attest(dtype in DTypeInt.__args__, _ASSERTION_MESSAGE_DTYPE_INT)

    if is_array_api_enabled() and hasattr(a, "dtype") and not isinstance(a, np.ndarray):
        xp = array_namespace(a)

        xp_dtype = getattr(xp, np.dtype(dtype).name, None)

        if xp_dtype is not None and a.dtype == xp_dtype:  # pyright: ignore
            return a  # pyright: ignore

    return as_array(a, dtype)


def as_float_array(a: ArrayLike, dtype: Type[DTypeFloat] | None = None) -> NDArrayFloat:
    """
    Convert the specified variable :math:`a` to :class:`numpy.ndarray` using
    the specified floating-point :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        Floating-point :class:`numpy.dtype` to use for conversion, default
        to the :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Variable :math:`a` converted to floating-point
        :class:`numpy.ndarray`.

    Examples
    --------
    >>> as_float_array([1, 2, 3])
    array([1., 2., 3.])
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    attest(dtype in DTypeFloat.__args__, _ASSERTION_MESSAGE_DTYPE_FLOAT)

    if is_array_api_enabled() and hasattr(a, "dtype") and not isinstance(a, np.ndarray):
        xp = array_namespace(a)

        xp_dtype = getattr(xp, np.dtype(dtype).name, None)

        if xp_dtype is not None:
            if a.dtype == xp_dtype:  # pyright: ignore
                return a  # pyright: ignore

            try:
                return xp.astype(a, xp_dtype)
            except (TypeError, AttributeError):
                return xp_astype(a, xp_dtype, xp)

    return as_array(a, dtype)


def as_int_scalar(a: ArrayLike, dtype: Type[DTypeInt] | None = None) -> int:
    """
    Convert the specified variable :math:`a` to :class:`numpy.integer` using
    the specified :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_INT_DEFAULT` attribute.

    Returns
    -------
    :class:`int`
        Variable :math:`a` converted to :class:`numpy.integer`.

    Warnings
    --------
    -   The return type is effectively annotated as :class:`int` and not
        :class:`numpy.integer`.

    Examples
    --------
    >>> as_int_scalar(np.array(1))
    np.int64(1)
    """

    a = as_int_array(a, dtype)

    xp = array_namespace(a)

    a = xp_reshape(a, (), xp=xp)

    attest(a.ndim == 0, f'"{a}" cannot be converted to "int" scalar!')

    # TODO: Revisit when Numpy types are well established.
    return cast("int", as_int(a, dtype))


def as_float_scalar(a: ArrayLike, dtype: Type[DTypeFloat] | None = None) -> float:
    """
    Convert the specified variable :math:`a` to :class:`numpy.floating` using
    the specified :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`float`
        Variable :math:`a` converted to :class:`numpy.floating`.

    Warnings
    --------
    -   The return type is effectively annotated as :class:`float` and not
        :class:`numpy.floating`.

    Examples
    --------
    >>> as_float_scalar(np.array(1))
    np.float64(1.0)
    """

    a = as_float_array(a, dtype)

    xp = array_namespace(a)

    a = xp_reshape(a, (), xp=xp)

    attest(a.ndim == 0, f'"{a}" cannot be converted to "float" scalar!')

    # TODO: Revisit when Numpy types are well established.
    return cast("float", as_float(a, dtype))


def as_complex_array(
    a: ArrayLike,
    dtype: Type[DTypeComplex] | None = None,
) -> NDArrayComplex:
    """
    Convert the specified variable :math:`a` to :class:`numpy.ndarray` using
    the specified complex :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        Complex :class:`numpy.dtype` to use for conversion, default
        to the :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_COMPLEX_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Variable :math:`a` converted to complex
        :class:`numpy.ndarray`.

    Examples
    --------
    >>> as_complex_array([1, 2, 3])
    array([1.+0.j, 2.+0.j, 3.+0.j])
    >>> as_complex_array([1 + 2j, 3 + 4j])
    array([1.+2.j, 3.+4.j])
    """

    dtype = optional(dtype, DTYPE_COMPLEX_DEFAULT)

    attest(dtype in DTypeComplex.__args__, _ASSERTION_MESSAGE_DTYPE_COMPLEX)

    return as_array(a, dtype)


def set_default_int_dtype(
    dtype: Type[DTypeInt] = DTYPE_INT_DEFAULT,
) -> None:
    """
    Set the *Colour* default :class:`numpy.integer` precision by setting
    :attr:`colour.constant.DTYPE_INT_DEFAULT` attribute with the specified
    :class:`numpy.dtype` wherever the attribute is imported.

    Parameters
    ----------
    dtype
        :class:`numpy.dtype` to set
        :attr:`colour.constant.DTYPE_INT_DEFAULT` with.

    Notes
    -----
    -   It is possible to define the integer precision at import time by
        setting the *COLOUR_SCIENCE__DEFAULT_INT_DTYPE* environment
        variable, for example `set COLOUR_SCIENCE__DEFAULT_INT_DTYPE=int32`.

    Warnings
    --------
    This definition is mostly given for consistency purposes with
    :func:`colour.utilities.set_default_float_dtype` definition but contrary
    to the latter, changing *integer* precision will almost certainly
    completely break *Colour*. With great power comes great responsibility.

    Examples
    --------
    >>> as_int_array(np.ones(3)).dtype  # doctest: +SKIP
    dtype('int64')
    >>> set_default_int_dtype(np.int32)  # doctest: +SKIP
    >>> as_int_array(np.ones(3)).dtype  # doctest: +SKIP
    dtype('int32')
    >>> set_default_int_dtype(np.int64)
    >>> as_int_array(np.ones(3)).dtype  # doctest: +SKIP
    dtype('int64')
    """

    # TODO: Investigate behaviour on Windows.
    with suppress_warnings(colour_usage_warnings=True):
        for module in sys.modules.values():
            if not hasattr(module, "DTYPE_INT_DEFAULT"):
                continue

            module.DTYPE_INT_DEFAULT = dtype  # pyright: ignore

    CACHE_REGISTRY.clear_all_caches()


def set_default_float_dtype(
    dtype: Type[DTypeFloat] = DTYPE_FLOAT_DEFAULT,
) -> None:
    """
    Set the *Colour* default :class:`numpy.floating` precision by setting
    :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute with the
    specified :class:`numpy.dtype` wherever the attribute is imported.

    Parameters
    ----------
    dtype
        :class:`numpy.dtype` to set
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` with.

    Notes
    -----
    -   It is possible to define the *float* precision at import time by
        setting the *COLOUR_SCIENCE__DEFAULT_FLOAT_DTYPE* environment
        variable, for example
        `set COLOUR_SCIENCE__DEFAULT_FLOAT_DTYPE=float32`.
    -   Some definition returning a single-scalar ndarray might not
        honour the specified *float* precision:
        https://github.com/numpy/numpy/issues/16353

    Warnings
    --------
    Changing *float* precision might result in various *Colour*
    functionality breaking entirely:
    https://github.com/numpy/numpy/issues/6860. With great power comes
    great responsibility.

    Examples
    --------
    >>> as_float_array(np.ones(3)).dtype
    dtype('float64')
    >>> set_default_float_dtype(np.float16)  # doctest: +SKIP
    >>> as_float_array(np.ones(3)).dtype  # doctest: +SKIP
    dtype('float16')
    >>> set_default_float_dtype(np.float64)
    >>> as_float_array(np.ones(3)).dtype
    dtype('float64')
    """

    with suppress_warnings(colour_usage_warnings=True):
        for module in sys.modules.values():
            if not hasattr(module, "DTYPE_FLOAT_DEFAULT"):
                continue

            module.DTYPE_FLOAT_DEFAULT = dtype  # pyright: ignore

    CACHE_REGISTRY.clear_all_caches()


# TODO: Annotate with "Union[Literal['ignore', 'reference', '1', '100'], str]"
# when Python 3.7 is dropped.
_DOMAIN_RANGE_SCALE = "reference"
"""
Global variable storing the current *Colour* domain-range scale.

_DOMAIN_RANGE_SCALE
"""


def get_domain_range_scale() -> Literal["ignore", "reference", "1", "100"] | str:
    """
    Return the current *Colour* domain-range scale.

    The following scales are available:

    -   **'Reference'**, the default *Colour* domain-range scale which
        varies depending on the referenced algorithm, e.g., [0, 1],
        [0, 10], [0, 100], [0, 255], etc...
    -   **'1'**, a domain-range scale normalised to [0, 1], it is
        important to acknowledge that this is a soft normalisation
        and it is possible to use negative out of gamut values or
        high dynamic range data exceeding 1.

    Returns
    -------
    :class:`str`
        *Colour* domain-range scale.

    Warnings
    --------
    -   The **'Ignore'** and **'100'** domain-range scales are for
        internal usage only!
    """

    return _DOMAIN_RANGE_SCALE


def set_domain_range_scale(
    scale: (
        Literal["ignore", "reference", "Ignore", "Reference", "1", "100"] | str
    ) = "reference",
) -> None:
    """
    Set the current *Colour* domain-range scale.

    The following scales are available:

    -   **'Reference'**, the default *Colour* domain-range scale which
        varies depending on the referenced algorithm, e.g., [0, 1],
        [0, 10], [0, 100], [0, 255], etc...
    -   **'1'**, a domain-range scale normalised to [0, 1], it is
        important to acknowledge that this is a soft normalisation and it
        is possible to use negative out of gamut values or high dynamic
        range data exceeding 1.

    Parameters
    ----------
    scale
        *Colour* domain-range scale to set.

    Warnings
    --------
    -   The **'Ignore'** and **'100'** domain-range scales are for
        internal usage only!
    """

    global _DOMAIN_RANGE_SCALE  # noqa: PLW0603

    _DOMAIN_RANGE_SCALE = validate_method(
        str(scale),
        ("ignore", "reference", "1", "100"),
        '"{0}" scale is invalid, it must be one of {1}!',
    )


class domain_range_scale:
    """
    Define a context manager and decorator to temporarily set the *Colour*
    domain-range scale.

    The following scales are available:

    -   **'Reference'**, the default *Colour* domain-range scale which
        varies depending on the referenced algorithm, e.g., [0, 1],
        [0, 10], [0, 100], [0, 255], etc...
    -   **'1'**, a domain-range scale normalised to [0, 1], it is
        important to acknowledge that this is a soft normalisation and it
        is possible to use negative out of gamut values or high dynamic
        range data exceeding 1.

    Parameters
    ----------
    scale
        *Colour* domain-range scale to set.

    Warnings
    --------
    -   The **'Ignore'** and **'100'** domain-range scales are for
        internal usage only!

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("1"):
    ...     to_domain_1(1)
    array(1.)
    >>> with domain_range_scale("Reference"):
    ...     from_range_1(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     to_domain_1(1)
    array(1.)
    >>> with domain_range_scale("1"):
    ...     from_range_1(1)
    array(1.)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     to_domain_1(1)
    array(0.01)
    >>> with domain_range_scale("100"):
    ...     from_range_1(1)
    array(100.)
    """

    def __init__(
        self,
        scale: (
            Literal["ignore", "reference", "Ignore", "Reference", "1", "100"] | str
        ),
    ) -> None:
        self._scale = scale
        self._previous_scale = get_domain_range_scale()

    def __enter__(self) -> Self:
        """Set the new domain-range scale upon entering the context manager."""

        set_domain_range_scale(self._scale)

        return self

    def __exit__(self, *args: Any) -> None:
        """
        Restore the previous domain-range scale upon exiting the context
        manager.
        """

        set_domain_range_scale(self._previous_scale)

    def __call__(self, function: Callable) -> Any:
        """
        Call the wrapped definition with domain-range scale management.
        """

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return function(*args, **kwargs)

        return wrapper


_CACHE_DOMAIN_RANGE_SCALE_METADATA: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_DOMAIN_RANGE_SCALE_METADATA"
)


def get_domain_range_scale_metadata(function: Callable) -> dict[str, Any]:
    """
    Extract domain-range scale metadata from function type hints.

    Extracts scale factors from PEP 593 ``Annotated`` type hints on function
    parameters and return values. This metadata indicates which scale factors
    to use when converting between 'Reference' and '1' modes.

    Parameters
    ----------
    function
        Function to extract metadata from.

    Returns
    -------
    :class:`dict`
        Dictionary with keys:

        - ``domain``: Dict mapping parameter names to their scale factors
        - ``range``: Scale factor for return value (int, tuple, or None)

    Examples
    --------
    >>> from colour.hints import Annotated, ArrayLike, NDArrayFloat
    >>> def example_function(
    ...     XYZ: Domain1,
    ...     illuminant: ArrayLike = None,
    ... ) -> Range100:
    ...     pass
    >>> metadata = get_domain_range_scale_metadata(example_function)
    >>> metadata["domain"]
    {'XYZ': 1}
    >>> metadata["range"]
    100
    """

    # Unwrap functools.partial to get the underlying function
    if hasattr(function, "func"):
        function = function.func  # pyright: ignore

    cache_key = id(function)

    if is_caching_enabled() and cache_key in _CACHE_DOMAIN_RANGE_SCALE_METADATA:
        return _CACHE_DOMAIN_RANGE_SCALE_METADATA[cache_key]

    metadata: dict[str, Any] = {"domain": {}, "range": None}

    def extract_scale_from_hint(hint: Any) -> Any | None:
        """
        Extract scale metadata from a type hint, handling Union types.

        Parameters
        ----------
        hint
            Type hint to extract scale from.

        Returns
        -------
        :class:`int` | :class:`tuple` | :class:`None`
            Scale metadata if found, None otherwise.
        """

        # Direct Annotated type with __metadata__
        if hasattr(hint, "__metadata__") and hint.__metadata__:
            return next(iter(hint.__metadata__))

        # Union type: check if any arg is Annotated
        origin = get_origin(hint)
        if origin is Union:
            for arg in get_args(hint):
                if hasattr(arg, "__metadata__") and arg.__metadata__:
                    return next(iter(arg.__metadata__))

        return None

    try:
        hints = get_type_hints(function, include_extras=True)
        # Process hints from get_type_hints (actual types with __metadata__)
        for parameter_name, hint in hints.items():
            scale = extract_scale_from_hint(hint)
            if scale is not None:
                if parameter_name == "return":
                    metadata["range"] = scale
                else:
                    metadata["domain"][parameter_name] = scale
    except (AttributeError, TypeError, NameError):
        # Fallback: parse string annotations (when `from __future__ import annotations`)
        # Mapping of type alias names to their scale values
        type_alias_scales = {
            "Domain1": 1,
            "Domain10": 10,
            "Domain100": 100,
            "Domain360": 360,
            "Domain100_100_360": (100, 100, 360),
            "Range1": 1,
            "Range10": 10,
            "Range100": 100,
            "Range360": 360,
            "Range100_100_360": (100, 100, 360),
        }

        hints = getattr(function, "__annotations__", {})
        for parameter_name, hint in hints.items():
            scale = None

            # Check if hint is a type alias name
            if isinstance(hint, str) and hint in type_alias_scales:
                scale = type_alias_scales[hint]
            # Extract scale from string: "Annotated[Type, scale]" -> scale
            elif (
                isinstance(hint, str)
                and "Annotated[" in hint
                and (match := re.search(r"Annotated\[[^,]+,\s*([^\]]+)\]", hint))
            ):
                scale_string = match.group(1).strip()
                # Evaluate scale (could be int, tuple, etc.)
                try:
                    scale = eval(scale_string)  # noqa: S307
                except (SyntaxError, NameError, ValueError):
                    scale = scale_string

            if scale is not None:
                if parameter_name == "return":
                    metadata["range"] = scale
                else:
                    metadata["domain"][parameter_name] = scale

    if is_caching_enabled():
        _CACHE_DOMAIN_RANGE_SCALE_METADATA[cache_key] = metadata

    return metadata


def to_domain_1(
    a: ArrayLike,
    scale_factor: ArrayLike = 100,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` to domain **'1'**.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'** or **'1'**, the
        definition is almost entirely by-passed and will conveniently
        convert array :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is divided
        by ``scale_factor``, typically 100.

    Parameters
    ----------
    a
        Array :math:`a` to scale to domain **'1'**.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray`
        if some axes need different scaling to be brought to domain **'1'**.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled to domain **'1'**.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     to_domain_1(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     to_domain_1(1)
    array(1.)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     to_domain_1(1)
    array(0.01)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = ndarray_copy(as_float_array(a, dtype))

    xp = array_namespace(a)

    if _DOMAIN_RANGE_SCALE == "100":
        a = as_float_array(a / xp_asarray(scale_factor, xp=xp, like=a), dtype)

    return a


def to_domain_10(
    a: ArrayLike,
    scale_factor: ArrayLike = 10,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` to domain **'10'**, used by the
    *Munsell Renotation System*.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the definition
        is almost entirely by-passed and will conveniently convert array
        :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        multiplied by ``scale_factor``, typically 10.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        divided by ``scale_factor``, typically 10.

    Parameters
    ----------
    a
        Array :math:`a` to scale to domain **'10'**.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray`
        if some axes need different scaling to be brought to domain
        **'10'**.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled to domain **'10'**.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     to_domain_10(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     to_domain_10(1)
    array(10.)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     to_domain_10(1)
    array(0.1)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = ndarray_copy(as_float_array(a, dtype))

    xp = array_namespace(a)

    if _DOMAIN_RANGE_SCALE == "1":
        a = as_float_array(a * xp_asarray(scale_factor, xp=xp, like=a), dtype)

    if _DOMAIN_RANGE_SCALE == "100":
        a = as_float_array(a / xp_asarray(scale_factor, xp=xp, like=a), dtype)

    return a


def to_domain_100(
    a: ArrayLike,
    scale_factor: ArrayLike = 100,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` to domain **'100'**.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'** or **'100'**
        (currently unsupported private value only used for unit tests), the
        definition is almost entirely by-passed and will conveniently
        convert array :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        multiplied by ``scale_factor``, typically 100.

    Parameters
    ----------
    a
        Array :math:`a` to scale to domain **'100'**.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray`
        if some axes need different scaling to be brought to domain
        **'100'**.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled to domain **'100'**.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     to_domain_100(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     to_domain_100(1)
    array(100.)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     to_domain_100(1)
    array(1.)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = ndarray_copy(as_float_array(a, dtype))

    xp = array_namespace(a)

    if _DOMAIN_RANGE_SCALE == "1":
        a = as_float_array(a * xp_asarray(scale_factor, xp=xp, like=a), dtype)

    return a


def to_domain_degrees(
    a: ArrayLike,
    scale_factor: ArrayLike = 360,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` to degrees domain.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the definition
        is almost entirely by-passed and will conveniently convert array
        :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        multiplied by ``scale_factor``, typically 360.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        multiplied by ``scale_factor`` / 100, typically 360 / 100.

    Parameters
    ----------
    a
        Array :math:`a` to scale to degrees domain.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray`
        if some axes need different scaling to be brought to degrees domain.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled to degrees domain.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     to_domain_degrees(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     to_domain_degrees(1)
    array(360.)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     to_domain_degrees(1)
    array(3.6)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = ndarray_copy(as_float_array(a, dtype))

    xp = array_namespace(a)

    if _DOMAIN_RANGE_SCALE == "1":
        a = as_float_array(a * xp_asarray(scale_factor, xp=xp, like=a), dtype)

    if _DOMAIN_RANGE_SCALE == "100":
        a = as_float_array(a * xp_asarray(scale_factor, xp=xp, like=a) / 100, dtype)

    return a


def to_domain_int(
    a: ArrayLike,
    bit_depth: ArrayLike = 8,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` to integer domain.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the definition
        is almost entirely by-passed and will conveniently convert array
        :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        multiplied by :math:`2^{bit\\_depth} - 1`.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        multiplied by :math:`2^{bit\\_depth} - 1`.

    Parameters
    ----------
    a
        Array :math:`a` to scale to integer domain.
    bit_depth
        Bit-depth, usually *int* but can be a :class:`numpy.ndarray` if
        some axis need different scaling to be brought to integer domain.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled to integer domain.

    Notes
    -----
    -   To avoid precision issues and rounding, the scaling is performed
        on *float* numbers.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     to_domain_int(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     to_domain_int(1)
    array(255.)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     to_domain_int(1)
    array(2.55)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = ndarray_copy(as_float_array(a, dtype))

    maximum_code_value = 2**bit_depth - 1  # pyright: ignore
    if _DOMAIN_RANGE_SCALE == "1":
        a = as_float_array(a * maximum_code_value, dtype)

    if _DOMAIN_RANGE_SCALE == "100":
        a = as_float_array(a * maximum_code_value / 100, dtype)

    return a


def from_range_1(
    a: ArrayLike,
    scale_factor: ArrayLike = 100,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` from range **'1'**.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'** or **'1'**, the
        definition is entirely by-passed.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        multiplied by ``scale_factor``, typically 100.

    Parameters
    ----------
    a
        Array :math:`a` to scale from range **'1'**.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray`
        if some axis need different scaling to be brought from range
        **'1'**.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled from range **'1'**.

    Warnings
    --------
    The scale conversion of variable :math:`a` happens in-place, i.e.,
    :math:`a` will be mutated!

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     from_range_1(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     from_range_1(1)
    array(1.)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     from_range_1(1)
    array(100.)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = as_float_array(a, dtype)

    xp = array_namespace(a)

    if _DOMAIN_RANGE_SCALE == "100":
        a = as_float_array(a * xp_asarray(scale_factor, xp=xp, like=a), dtype)

    return a


def from_range_10(
    a: ArrayLike,
    scale_factor: ArrayLike = 10,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` from range **'10'**, used by the
    *Munsell Renotation System*.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the definition
        is entirely by-passed.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        divided by ``scale_factor``, typically 10.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        multiplied by ``scale_factor``, typically 10.

    Parameters
    ----------
    a
        Array :math:`a` to scale from range **'10'**.
    scale_factor
        Scale factor, usually *numeric* but can be a
        :class:`numpy.ndarray` if some axis need different scaling to be
        brought from range **'10'**.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled from range **'10'**.

    Warnings
    --------
    The scale conversion of variable :math:`a` happens in-place, i.e.,
    :math:`a` will be mutated!

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     from_range_10(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     from_range_10(1)
    array(0.1)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     from_range_10(1)
    array(10.)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = as_float_array(a, dtype)

    xp = array_namespace(a)

    if _DOMAIN_RANGE_SCALE == "1":
        a = as_float_array(a / xp_asarray(scale_factor, xp=xp, like=a), dtype)

    if _DOMAIN_RANGE_SCALE == "100":
        a = as_float_array(a * xp_asarray(scale_factor, xp=xp, like=a), dtype)

    return a


def from_range_100(
    a: ArrayLike,
    scale_factor: ArrayLike = 100,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` from range **'100'**.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'** or **'100'**
        (currently unsupported private value only used for unit tests), the
        definition is entirely by-passed.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        divided by ``scale_factor``, typically 100.

    Parameters
    ----------
    a
        Array :math:`a` to scale from range **'100'**.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray`
        if some axes require different scaling to be brought from range
        **'100'**.
    dtype
        Data type used for the conversion to :class:`numpy.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled from range **'100'**.

    Warnings
    --------
    The scale conversion of variable :math:`a` happens in-place, i.e.,
    :math:`a` will be mutated!

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     from_range_100(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     from_range_100(1)
    array(0.01)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     from_range_100(1)
    array(1.)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = as_float_array(a, dtype)

    xp = array_namespace(a)

    if _DOMAIN_RANGE_SCALE == "1":
        a = as_float_array(a / xp_asarray(scale_factor, xp=xp, like=a), dtype)

    return a


def from_range_degrees(
    a: ArrayLike,
    scale_factor: ArrayLike = 360,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` from degrees range.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the definition
        is entirely by-passed.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        divided by ``scale_factor``, typically 360.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        divided by ``scale_factor`` / 100, typically 360 / 100.

    Parameters
    ----------
    a
        Array :math:`a` to scale from degrees range.
    scale_factor
        Scale factor, usually *numeric* but can be a
        :class:`numpy.ndarray` if some axes need different scaling to be
        brought from degrees range.
    dtype
        Data type used for the conversion to :class:`numpy.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled from degrees range.

    Warnings
    --------
    The scale conversion of variable :math:`a` happens in-place, i.e.,
    :math:`a` will be mutated!

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     from_range_degrees(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     from_range_degrees(1)  # doctest: +ELLIPSIS
    array(0.0027777...)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     from_range_degrees(1)  # doctest: +ELLIPSIS
    array(0.2777777...)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = as_float_array(a, dtype)

    xp = array_namespace(a)

    if _DOMAIN_RANGE_SCALE == "1":
        a = as_float_array(a / xp_asarray(scale_factor, xp=xp, like=a), dtype)

    if _DOMAIN_RANGE_SCALE == "100":
        a = as_float_array(a / (xp_asarray(scale_factor, xp=xp, like=a) / 100), dtype)

    return a


def from_range_int(
    a: ArrayLike,
    bit_depth: ArrayLike = 8,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` from integer range.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the definition
        is entirely by-passed.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        converted to :class:`np.ndarray` and divided by
        :math:`2^{bit\\_depth} - 1`.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        converted to :class:`np.ndarray` and divided by
        :math:`2^{bit\\_depth} - 1`.

    Parameters
    ----------
    a
        Array :math:`a` to scale from integer range.
    bit_depth
        Bit-depth, usually *int* but can be a :class:`numpy.ndarray` if
        some axes need different scaling to be brought from integer range.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled from integer range.

    Warnings
    --------
    The scale conversion of variable :math:`a` happens in-place, i.e.,
    :math:`a` will be mutated!

    Notes
    -----
    -   To avoid precision issues and rounding, the scaling is performed on
        *float* numbers.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     from_range_int(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     from_range_int(1)  # doctest: +ELLIPSIS
    array(0.0039215...)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     from_range_int(1)  # doctest: +ELLIPSIS
    array(0.3921568...)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = as_float_array(a, dtype)

    maximum_code_value = 2**bit_depth - 1  # pyright: ignore
    if _DOMAIN_RANGE_SCALE == "1":
        a = as_float_array(a / maximum_code_value, dtype)

    if _DOMAIN_RANGE_SCALE == "100":
        a = as_float_array(a / (maximum_code_value / 100), dtype)

    return a


_NDARRAY_COPY_ENABLED: bool = True
"""
Global variable storing the current *Colour* state for
:class:`numpy.ndarray` copy.
"""


def is_ndarray_copy_enabled() -> bool:
    """
    Determine whether *Colour* :class:`numpy.ndarray` copy is enabled.

    Various API objects return a copy of their internal
    :class:`numpy.ndarray` for safety purposes, but this can be a slow
    operation impacting performance.

    Returns
    -------
    :class:`bool`
        Whether *Colour* :class:`numpy.ndarray` copy is enabled.

    Examples
    --------
    >>> with ndarray_copy_enable(False):
    ...     is_ndarray_copy_enabled()
    False
    >>> with ndarray_copy_enable(True):
    ...     is_ndarray_copy_enabled()
    True
    """

    return _NDARRAY_COPY_ENABLED


def set_ndarray_copy_enable(enable: bool) -> None:
    """
    Set the *Colour* :class:`numpy.ndarray` copy enabled state.

    Parameters
    ----------
    enable
        Whether to enable *Colour* :class:`numpy.ndarray` copy.

    Examples
    --------
    >>> with ndarray_copy_enable(is_ndarray_copy_enabled()):
    ...     print(is_ndarray_copy_enabled())
    ...     set_ndarray_copy_enable(False)
    ...     print(is_ndarray_copy_enabled())
    True
    False
    """

    global _NDARRAY_COPY_ENABLED  # noqa: PLW0603

    _NDARRAY_COPY_ENABLED = enable


class ndarray_copy_enable:
    """
    Define a context manager and decorator to temporarily set the *Colour*
    :class:`numpy.ndarray` copy enabled state.

    Parameters
    ----------
    enable
        Whether to enable or disable *Colour* :class:`numpy.ndarray` copy.
    """

    def __init__(self, enable: bool) -> None:
        self._enable = enable
        self._previous_state = is_ndarray_copy_enabled()

    def __enter__(self) -> Self:
        """
        Set the *Colour* :class:`numpy.ndarray` copy enabled state upon
        entering the context manager.
        """

        set_ndarray_copy_enable(self._enable)

        return self

    def __exit__(self, *args: Any) -> None:
        """
        Restore the *Colour* :class:`numpy.ndarray` copy enabled state upon
        exiting the context manager.
        """

        set_ndarray_copy_enable(self._previous_state)

    def __call__(self, function: Callable) -> Callable:
        """
        Decorate and call the specified function with array copy control.

        Parameters
        ----------
        function
            Function to be decorated with array copy state management.

        Returns
        -------
        :class:`Callable`
            Decorated function that executes within the configured array copy
            state context.
        """

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return function(*args, **kwargs)

        return wrapper


def ndarray_copy(a: NDArray) -> NDArray:
    """
    Return a :class:`numpy.ndarray` copy if the relevant *Colour* state is
    enabled.

    Various API objects return a copy of their internal
    :class:`numpy.ndarray` for safety purposes, but this can be a slow
    operation impacting performance.

    Parameters
    ----------
    a
        Array :math:`a` to return a copy of.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` copy according to *Colour* state.

    Examples
    --------
    >>> a = np.linspace(0, 1, 10)
    >>> id(a) == id(ndarray_copy(a))
    False
    >>> with ndarray_copy_enable(False):
    ...     id(a) == id(ndarray_copy(a))
    True
    """

    if _NDARRAY_COPY_ENABLED:
        xp = array_namespace(a)

        if is_numpy_namespace(xp):
            return np.copy(a)
        return xp.asarray(a, copy=True)
    return a


def closest_indexes(a: ArrayLike, b: ArrayLike) -> NDArray:
    """
    Return the closest element indexes from array :math:`a` to reference array
    :math:`b` elements.

    Parameters
    ----------
    a
        Array :math:`a` to search for the closest elements.
    b
        Reference array :math:`b`.

    Returns
    -------
    :class:`numpy.ndarray`
        Closest array :math:`a` element indexes.

    Examples
    --------
    >>> a = np.array(
    ...     [
    ...         24.31357115,
    ...         63.62396289,
    ...         55.71528816,
    ...         62.70988028,
    ...         46.84480573,
    ...         25.40026416,
    ...     ]
    ... )
    >>> print(closest_indexes(a, 63))
    [3]
    >>> print(closest_indexes(a, [63, 25]))
    [3 5]
    """

    a = as_float_array(a)
    b = as_float_array(b)

    xp = array_namespace(a, b)

    a = xp_asarray(a, xp=xp, like=b)
    b = xp_asarray(b, xp=xp, like=a)

    a = xp_reshape(a, (-1,), xp=xp)[:, None]
    b = xp_reshape(b, (-1,), xp=xp)[None, :]

    return xp.abs(a - b).argmin(axis=0)


def closest(a: ArrayLike, b: ArrayLike) -> NDArray:
    """
    Return the closest array :math:`a` elements to reference array
    :math:`b` elements.

    Parameters
    ----------
    a
        Array :math:`a` to search for the closest elements.
    b
        Reference array :math:`b`.

    Returns
    -------
    :class:`numpy.ndarray`
        Closest array :math:`a` elements.

    Examples
    --------
    >>> a = np.array(
    ...     [
    ...         24.31357115,
    ...         63.62396289,
    ...         55.71528816,
    ...         62.70988028,
    ...         46.84480573,
    ...         25.40026416,
    ...     ]
    ... )
    >>> closest(a, 63)
    array([62.70988028])
    >>> closest(a, [63, 25])
    array([62.70988028, 25.40026416])
    """

    a = as_float_array(a)
    b = as_float_array(b)

    xp = array_namespace(a, b)

    a = xp_asarray(a, xp=xp, like=b)

    return a[closest_indexes(a, b)]


_CACHE_DISTRIBUTION_INTERVAL: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_DISTRIBUTION_INTERVAL"
)


def interval(distribution: ArrayLike, unique: bool = True) -> NDArray:
    """
    Return the interval size of the specified distribution.

    Parameters
    ----------
    distribution
        Distribution to retrieve the interval from.
    unique
        Whether to return unique intervals if the distribution is
        non-uniformly spaced or the complete intervals.

    Returns
    -------
    :class:`numpy.ndarray`
        Distribution interval.

    Examples
    --------
    Uniformly spaced variable:

    >>> y = np.array([1, 2, 3, 4, 5])
    >>> interval(y)
    array([1.])
    >>> interval(y, False)
    array([1., 1., 1., 1.])

    Non-uniformly spaced variable:

    >>> y = np.array([1, 2, 3, 4, 8])
    >>> interval(y)
    array([1., 4.])
    >>> interval(y, False)
    array([1., 1., 1., 4.])
    """

    distribution = as_float_array(distribution)

    xp = array_namespace(distribution)

    hash_key = hash(
        (
            int_digest(np.asarray(distribution).tobytes()),
            distribution.shape,
            unique,
        )
    )

    if is_caching_enabled() and hash_key in _CACHE_DISTRIBUTION_INTERVAL:
        return xp.asarray(_CACHE_DISTRIBUTION_INTERVAL[hash_key], copy=True)

    differences = xp.abs(distribution[1:] - distribution[:-1])

    if unique and xp.all(differences == differences[0]):
        interval_ = xp.asarray([differences[0]])
    elif unique:
        interval_ = xp.unique_values(differences)
    else:
        interval_ = differences

    _CACHE_DISTRIBUTION_INTERVAL[hash_key] = xp.asarray(interval_, copy=True)

    return interval_


def is_uniform(distribution: ArrayLike) -> bool:
    """
    Determine whether the specified distribution is uniform.

    Parameters
    ----------
    distribution
        Distribution to check for uniformity.

    Returns
    -------
    :class:`bool`
        Whether the distribution is uniform.

    Examples
    --------
    Uniformly spaced variable:

    >>> a = np.array([1, 2, 3, 4, 5])
    >>> is_uniform(a)
    True

    Non-uniformly spaced variable:

    >>> a = np.array([1, 2, 3.1415, 4, 5])
    >>> is_uniform(a)
    False
    """

    return len(interval(distribution)) == 1


def in_array(a: ArrayLike, b: ArrayLike, tolerance: Real = EPSILON) -> NDArray:
    """
    Determine whether each element of array :math:`a` is present in array
    :math:`b` within the specified tolerance.

    Parameters
    ----------
    a
        Array :math:`a` to test the elements from.
    b
        Array :math:`b` against which to test the elements of array
        :math:`a`.
    tolerance
        Tolerance value.

    Returns
    -------
    :class:`numpy.ndarray`
        Boolean array with array :math:`a` shape indicating whether each
        element of array :math:`a` is present in array :math:`b` within the
        specified tolerance.

    References
    ----------
    :cite:`Yorke2014a`

    Examples
    --------
    >>> a = np.array([0.50, 0.60])
    >>> b = np.linspace(0, 10, 101)
    >>> np.isin(a, b)
    array([ True, False])
    >>> in_array(a, b)
    array([ True,  True])
    """

    a = as_float_array(a)
    b = as_float_array(b)

    xp = array_namespace(a, b)

    d = xp.abs(xp_reshape(a, (-1,), xp=xp) - b[..., None])

    return xp_reshape(xp.any(d <= tolerance, axis=0), a.shape, xp=xp)


def tstack(
    a: ArrayLike,
    dtype: Type[DTypeBoolean] | Type[DTypeReal] | None = None,
) -> NDArray:
    """
    Stack the specified array of arrays :math:`a` along the last axis (tail)
    to produce a stacked array.

    Used to stack an array of arrays produced by the
    :func:`colour.utilities.tsplit` definition.

    Parameters
    ----------
    a
        Array of arrays :math:`a` to stack along the last axis.
    dtype
        :class:`numpy.dtype` to use for initial conversion to
        :class:`numpy.ndarray`, default to the :class:`numpy.dtype` defined
        by :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Stacked array.

    Examples
    --------
    >>> a = 0
    >>> tstack([a, a, a])
    array([0., 0., 0.])
    >>> a = np.arange(0, 6)
    >>> tstack([a, a, a])
    array([[0., 0., 0.],
           [1., 1., 1.],
           [2., 2., 2.],
           [3., 3., 3.],
           [4., 4., 4.],
           [5., 5., 5.]])
    >>> a = np.reshape(a, (1, 6))
    >>> tstack([a, a, a])
    array([[[0., 0., 0.],
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
            [4., 4., 4.],
            [5., 5., 5.]]])
    >>> a = np.reshape(a, (1, 1, 6))
    >>> tstack([a, a, a])
    array([[[[0., 0., 0.],
             [1., 1., 1.],
             [2., 2., 2.],
             [3., 3., 3.],
             [4., 4., 4.],
             [5., 5., 5.]]]])
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    if (
        is_array_api_enabled()
        and isinstance(a, (list, tuple))
        and a
        and hasattr(a[0], "shape")
        and not isinstance(a[0], (np.ndarray, np.generic))
    ):
        xp = array_namespace(a[0])

        a = [xp_asarray(x, xp=xp, like=a[0]) for x in a]

        return xp.concat([x[..., None] for x in a], axis=-1)

    a = as_array(a, dtype)

    xp = array_namespace(a)

    return xp.concat([x[..., None] for x in a], axis=-1)


def tsplit(
    a: ArrayLike,
    dtype: Type[DTypeBoolean] | Type[DTypeReal] | None = None,
) -> NDArray:
    """
    Split the specified stacked array :math:`a` along the last axis (tail)
    to produce an array of arrays.

    Used to split a stacked array produced by the :func:`colour.utilities.tstack`
    definition.

    Parameters
    ----------
    a
        Stacked array :math:`a` to split.
    dtype
        :class:`numpy.dtype` to use for initial conversion to
        :class:`numpy.ndarray`, default to the :class:`numpy.dtype` defined
        by :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of arrays.

    Examples
    --------
    >>> a = np.array([0, 0, 0])
    >>> tsplit(a)
    array([0., 0., 0.])
    >>> a = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
    >>> tsplit(a)
    array([[0., 1., 2., 3., 4., 5.],
           [0., 1., 2., 3., 4., 5.],
           [0., 1., 2., 3., 4., 5.]])
    >>> a = np.array(
    ...     [
    ...         [
    ...             [0, 0, 0],
    ...             [1, 1, 1],
    ...             [2, 2, 2],
    ...             [3, 3, 3],
    ...             [4, 4, 4],
    ...             [5, 5, 5],
    ...         ]
    ...     ]
    ... )
    >>> tsplit(a)
    array([[[0., 1., 2., 3., 4., 5.]],
    <BLANKLINE>
           [[0., 1., 2., 3., 4., 5.]],
    <BLANKLINE>
           [[0., 1., 2., 3., 4., 5.]]])
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = as_array(a, dtype)

    xp = array_namespace(a)

    return xp.stack([a[..., x] for x in range(a.shape[-1])])


def row_as_diagonal(a: ArrayLike) -> NDArray:
    """
    Return the rows of the specified array :math:`a` as diagonal matrices.

    Parameters
    ----------
    a
        Array :math:`a` to return the rows of as diagonal matrices.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` rows as diagonal matrices.

    References
    ----------
    :cite:`Castro2014a`

    Examples
    --------
    >>> a = np.array(
    ...     [
    ...         [0.25891593, 0.07299478, 0.36586996],
    ...         [0.30851087, 0.37131459, 0.16274825],
    ...         [0.71061831, 0.67718718, 0.09562581],
    ...         [0.71588836, 0.76772047, 0.15476079],
    ...         [0.92985142, 0.22263399, 0.88027331],
    ...     ]
    ... )
    >>> row_as_diagonal(a)
    array([[[0.25891593, 0.        , 0.        ],
            [0.        , 0.07299478, 0.        ],
            [0.        , 0.        , 0.36586996]],
    <BLANKLINE>
           [[0.30851087, 0.        , 0.        ],
            [0.        , 0.37131459, 0.        ],
            [0.        , 0.        , 0.16274825]],
    <BLANKLINE>
           [[0.71061831, 0.        , 0.        ],
            [0.        , 0.67718718, 0.        ],
            [0.        , 0.        , 0.09562581]],
    <BLANKLINE>
           [[0.71588836, 0.        , 0.        ],
            [0.        , 0.76772047, 0.        ],
            [0.        , 0.        , 0.15476079]],
    <BLANKLINE>
           [[0.92985142, 0.        , 0.        ],
            [0.        , 0.22263399, 0.        ],
            [0.        , 0.        , 0.88027331]]])
    """

    d = as_array(a)

    xp = array_namespace(d)

    d = xp.expand_dims(d, axis=-2)

    eye = xp.eye(d.shape[-1])
    device = getattr(d, "device", None)
    if device is not None and hasattr(device, "type"):
        eye = eye.to(device)

    return eye * d


def orient(
    a: ArrayLike,
    orientation: (
        Literal["Ignore", "Flip", "Flop", "90 CW", "90 CCW", "180"] | str
    ) = "Ignore",
) -> NDArray:
    """
    Orient the specified array :math:`a` using the specified orientation.

    Parameters
    ----------
    a
        Array :math:`a` to orient.
    orientation
        Orientation to perform.

    Returns
    -------
    :class:`numpy.ndarray`
        Oriented array.

    Examples
    --------
    >>> a = np.tile(np.arange(5), (5, 1))
    >>> a
    array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4]])
    >>> orient(a, "90 CW")
    array([[0., 0., 0., 0., 0.],
           [1., 1., 1., 1., 1.],
           [2., 2., 2., 2., 2.],
           [3., 3., 3., 3., 3.],
           [4., 4., 4., 4., 4.]])
    >>> orient(a, "Flip")
    array([[4., 3., 2., 1., 0.],
           [4., 3., 2., 1., 0.],
           [4., 3., 2., 1., 0.],
           [4., 3., 2., 1., 0.],
           [4., 3., 2., 1., 0.]])
    """

    a = as_float_array(a)

    xp = array_namespace(a)

    orientation = validate_method(
        orientation, ("Ignore", "Flip", "Flop", "90 CW", "90 CCW", "180")
    )

    oriented = a
    if orientation == "ignore":
        oriented = a
    elif orientation == "flip":
        oriented = xp.flip(a, axis=1)
    elif orientation == "flop":
        oriented = xp.flip(a, axis=0)
    elif orientation == "90 cw":
        oriented = xp.matrix_transpose(xp.flip(a, axis=0))
    elif orientation == "90 ccw":
        oriented = xp.matrix_transpose(xp.flip(a, axis=1))
    elif orientation == "180":
        oriented = xp.flip(xp.flip(a, axis=0), axis=1)

    return oriented


def centroid(a: ArrayLike) -> NDArrayInt:
    """
    Return the centroid indexes of the specified array :math:`a`.

    Parameters
    ----------
    a
        Array :math:`a` to return the centroid indexes of.

    Returns
    -------
    :class:`numpy.ndarray`
        Centroid indexes of array :math:`a`.

    Examples
    --------
    >>> a = np.tile(np.arange(0, 5), (5, 1))
    >>> centroid(a)  # doctest: +ELLIPSIS
    array([2, 3]...)
    """

    a = as_float_array(a)

    xp = array_namespace(a)

    a_s = xp.sum(a)

    ranges = [xp.arange(0, a.shape[i]) for i in range(a.ndim)]
    coordinates = xp.meshgrid(*ranges)

    a_ci = []
    for axis in coordinates:
        axis = xp.permute_dims(axis, tuple(reversed(range(axis.ndim))))  # noqa: PLW2901
        # Aligning axis for N-D arrays where N is normalised to
        # range [3, :math:`\\\infty`]
        for i in range(axis.ndim - 2, 0, -1):
            axis = xp.moveaxis(axis, i - 1, -1)  # noqa: PLW2901

        a_ci.append(xp.sum(axis * a) // a_s)

    # NOTE: Cannot use ``as_int_array`` as presence of NaN will raise a
    # ``ValueError`` exception.
    return xp_astype(xp.stack(a_ci), DTYPE_INT_DEFAULT, xp)


def fill_nan(
    a: ArrayLike,
    method: Literal["Interpolation", "Constant"] | str = "Interpolation",
    default: Real = 0,
) -> NDArray:
    """
    Fill the NaN values in the specified array :math:`a` using the specified
    method.

    Parameters
    ----------
    a
        Array :math:`a` to fill the NaNs of.
    method
        *Interpolation* method linearly interpolates through the NaN values,
        *Constant* method replaces NaN values with ``default``.
    default
        Value to use with the *Constant* method.

    Returns
    -------
    :class:`numpy.ndarray`
        NaN-filled array :math:`a`.

    Examples
    --------
    >>> a = np.array([0.1, 0.2, np.nan, 0.4, 0.5])
    >>> fill_nan(a)
    array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> fill_nan(a, method="Constant")
    array([0.1, 0.2, 0. , 0.4, 0.5])
    """

    a = as_float_array(a)

    xp = array_namespace(a)

    a = xp.asarray(a, copy=True)
    method = validate_method(method, ("Interpolation", "Constant"))

    mask = xp.isnan(a)

    if not xp.any(mask):
        return a  # pyright: ignore

    if method == "interpolation":
        indices = xp.arange(len(a))  # pyright: ignore
        valid = ~mask
        a = xp.where(
            mask,
            xp_interp(indices, indices[valid], a[valid], xp=xp),  # pyright: ignore
            a,
        )
    elif method == "constant":
        a = xp.where(mask, default, a)

    return a  # pyright: ignore


def has_only_nan(a: ArrayLike) -> bool:
    """
    Return whether the specified array :math:`a` contains only *NaN* values.

    Parameters
    ----------
    a
        Array :math:`a` to check whether it contains only *NaN* values.

    Returns
    -------
    :class:`bool`
        Whether array :math:`a` contains only *NaN* values.

    Examples
    --------
    >>> has_only_nan(None)
    True
    >>> has_only_nan([None, None])
    True
    >>> has_only_nan([True, None])
    False
    >>> has_only_nan([0.1, np.nan, 0.3])
    False
    """

    a = as_float_array(a)

    xp = array_namespace(a)

    return bool(xp.all(xp.isnan(a)))


@contextmanager
def ndarray_write(a: ArrayLike) -> Generator:
    """
    Define a context manager that temporarily sets the specified array
    :math:`a` to writeable for operations, then restores it to read-only.

    Parameters
    ----------
    a
        Array :math:`a` to operate on.

    Yields
    ------
    Generator
        Array :math:`a` made temporarily writeable.

    Examples
    --------
    >>> a = np.linspace(0, 1, 10)
    >>> a.setflags(write=False)
    >>> try:
    ...     a += 1
    ... except ValueError:
    ...     pass
    >>> with ndarray_write(a):
    ...     a += 1
    """

    a = as_float_array(a)

    a.setflags(write=True)

    try:
        yield a
    finally:
        a.setflags(write=False)


def zeros(
    shape: int | Sequence[int],
    dtype: Type[DTypeReal] | None = None,
) -> NDArray:
    """
    Create an array of zeros with the active dtype.

    Wrap :func:`np.zeros` definition to create an array with the active
    :class:`numpy.dtype` defined by the
    :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Parameters
    ----------
    shape
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of the specified shape and :class:`numpy.dtype`, filled
        with zeros.

    Examples
    --------
    >>> zeros(3)
    array([0., 0., 0.])
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    xp = array_namespace()

    return xp.zeros(shape, dtype=dtype)


def ones(
    shape: int | Sequence[int],
    dtype: Type[DTypeReal] | None = None,
) -> NDArray:
    """
    Create an array of ones with the active dtype.

    Wrap :func:`np.ones` definition to create an array with the active
    :class:`numpy.dtype` defined by the
    :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Parameters
    ----------
    shape
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of the specified shape and :class:`numpy.dtype`, filled with ones.

    Examples
    --------
    >>> ones(3)
    array([1., 1., 1.])
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    xp = array_namespace()

    return xp.ones(shape, dtype=dtype)


def full(
    shape: int | Sequence[int],
    fill_value: Real,
    dtype: Type[DTypeReal] | None = None,
) -> NDArray:
    """
    Create an array of the specified value with the active dtype.

    Wrap :func:`np.full` definition to create an array with the active
    :class:`numpy.dtype` defined by the
    :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Parameters
    ----------
    shape
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value
        Fill value.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of the specified shape and :class:`numpy.dtype`, filled with
        the specified value.

    Examples
    --------
    >>> full(3, 2.5)
    array([2.5, 2.5, 2.5])
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    xp = array_namespace()

    return xp.full(shape, fill_value, dtype=dtype)


def index_along_last_axis(a: ArrayLike, indexes: ArrayLike) -> NDArray:
    """
    Reduce the dimension of array :math:`a` by one, using an array of
    indexes to select elements from the last axis.

    Parameters
    ----------
    a
        Array :math:`a` to be indexed.
    indexes
        *Integer* array with the same shape as :math:`a` but with one
        dimension fewer, containing indices to the last dimension of
        :math:`a`. All elements must be numbers between 0 and
        :math:`m - 1`.

    Returns
    -------
    :class:`numpy.ndarray`
        Indexed array :math:`a`.

    Raises
    ------
    :class:`ValueError`
        If the array :math:`a` and ``indexes`` have incompatible shapes.
    :class:`IndexError`
        If ``indexes`` has elements outside of the allowed range of 0 to
        :math:`m - 1` or if it is not an *integer* array.

    Examples
    --------
    >>> a = np.array(
    ...     [
    ...         [
    ...             [0.3, 0.5, 6.9],
    ...             [3.3, 4.4, 1.6],
    ...             [4.4, 7.5, 2.3],
    ...             [2.3, 1.6, 7.4],
    ...         ],
    ...         [
    ...             [2.0, 5.9, 2.8],
    ...             [6.2, 4.9, 8.6],
    ...             [3.7, 9.7, 7.3],
    ...             [6.3, 4.3, 3.2],
    ...         ],
    ...         [
    ...             [0.8, 1.9, 0.7],
    ...             [5.6, 4.0, 1.7],
    ...             [6.7, 8.2, 1.7],
    ...             [1.2, 7.1, 1.4],
    ...         ],
    ...         [
    ...             [4.0, 4.8, 8.9],
    ...             [4.0, 0.3, 6.9],
    ...             [3.5, 7.1, 4.5],
    ...             [1.4, 1.9, 1.6],
    ...         ],
    ...     ]
    ... )
    >>> indexes = np.array([[2, 0, 1, 1], [2, 1, 1, 0], [0, 0, 1, 2], [0, 0, 1, 2]])
    >>> index_along_last_axis(a, indexes)
    array([[6.9, 3.3, 7.5, 1.6],
           [2.8, 4.9, 9.7, 6.3],
           [0.8, 5.6, 8.2, 1.4],
           [4. , 4. , 7.1, 1.6]])

    This function can be used to compute the result of :func:`np.min` along
    the last axis given the corresponding :func:`np.argmin` indexes.

    >>> indexes = np.argmin(a, axis=-1)
    >>> np.array_equal(index_along_last_axis(a, indexes), np.min(a, axis=-1))
    True

    In particular, this can be used to manipulate the indexes specified by
    functions like :func:`np.min` before indexing the array. For example, to
    get elements directly following the smallest elements:

    >>> index_along_last_axis(a, (indexes + 1) % 3)
    array([[0.5, 3.3, 4.4, 7.4],
           [5.9, 8.6, 9.7, 6.3],
           [0.8, 5.6, 6.7, 7.1],
           [4.8, 6.9, 7.1, 1.9]])
    """

    a = as_float_array(a)
    indexes = as_int_array(indexes)

    xp = array_namespace(a)

    if a.shape[:-1] != indexes.shape:
        error = (
            f"Array and indexes have incompatible shapes: {a.shape} and {indexes.shape}"
        )

        raise ValueError(error)

    return xp.take_along_axis(a, indexes[..., None], axis=-1).squeeze(axis=-1)


def format_array_as_row(a: ArrayLike, decimals: int = 7, separator: str = " ") -> str:
    """
    Format the specified array :math:`a` as a row.

    Parameters
    ----------
    a
        Array to format.
    decimals
        Decimal count to use when formatting as a row.
    separator
        Separator used to join the array :math:`a` items.

    Returns
    -------
    :class:`str`
        Array formatted as a row.

    Examples
    --------
    >>> format_array_as_row([1.25, 2.5, 3.75])
    '1.2500000 2.5000000 3.7500000'
    >>> format_array_as_row([1.25, 2.5, 3.75], 3)
    '1.250 2.500 3.750'
    >>> format_array_as_row([1.25, 2.5, 3.75], 3, ", ")
    '1.250, 2.500, 3.750'
    """

    a = as_float_array(a)

    xp = array_namespace(a)

    a = xp_reshape(a, (-1,), xp=xp)

    return separator.join(
        "{1:0.{0}f}".format(decimals, x)
        for x in a  # noqa: PLE1300, RUF100
    )
