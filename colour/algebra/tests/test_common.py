"""Define the unit tests for the :mod:`colour.algebra.common` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np
import pytest

from colour.algebra import (
    eigen_decomposition,
    euclidean_distance,
    get_sdiv_mode,
    is_identity,
    is_spow_enabled,
    linear_conversion,
    linstep_function,
    manhattan_distance,
    normalise_maximum,
    normalise_vector,
    sdiv,
    sdiv_mode,
    set_sdiv_mode,
    set_spow_enable,
    smoothstep_function,
    spow,
    spow_enable,
    vecmul,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    ColourRuntimeWarning,
    array_api_enable,
    ignore_numpy_errors,
    is_numpy_namespace,
    xp_asarray,
    xp_assert_close,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestGetSdivMode",
    "TestSetSdivMode",
    "TestSdivMode",
    "TestSdiv",
    "TestIsSpowEnabled",
    "TestSetSpowEnabled",
    "TestSpowEnable",
    "TestSpow",
    "TestSmoothstepFunction",
    "TestNormaliseVector",
    "TestNormaliseMaximum",
    "TestVectorDot",
    "TestEuclideanDistance",
    "TestManhattanDistance",
    "TestLinearConversion",
    "TestLinstepFunction",
    "TestIsIdentity",
    "TestEigenDecomposition",
]


class TestGetSdivMode:
    """
    Define :func:`colour.algebra.common.get_sdiv_mode` definition unit tests
    methods.
    """

    def test_get_sdiv_mode(self) -> None:
        """Test :func:`colour.algebra.common.get_sdiv_mode` definition."""

        with sdiv_mode("Numpy"):
            assert get_sdiv_mode() == "numpy"

        with sdiv_mode("Ignore"):
            assert get_sdiv_mode() == "ignore"

        with sdiv_mode("Warning"):
            assert get_sdiv_mode() == "warning"

        with sdiv_mode("Raise"):
            assert get_sdiv_mode() == "raise"

        with sdiv_mode("Ignore Zero Conversion"):
            assert get_sdiv_mode() == "ignore zero conversion"

        with sdiv_mode("Warning Zero Conversion"):
            assert get_sdiv_mode() == "warning zero conversion"

        with sdiv_mode("Ignore Limit Conversion"):
            assert get_sdiv_mode() == "ignore limit conversion"

        with sdiv_mode("Warning Limit Conversion"):
            assert get_sdiv_mode() == "warning limit conversion"


class TestSetSdivMode:
    """
    Define :func:`colour.algebra.common.set_sdiv_mode` definition unit tests
    methods.
    """

    def test_set_sdiv_mode(self) -> None:
        """Test :func:`colour.algebra.common.set_sdiv_mode` definition."""

        with sdiv_mode(get_sdiv_mode()):
            set_sdiv_mode("Numpy")
            assert get_sdiv_mode() == "numpy"

            set_sdiv_mode("Ignore")
            assert get_sdiv_mode() == "ignore"

            set_sdiv_mode("Warning")
            assert get_sdiv_mode() == "warning"

            set_sdiv_mode("Raise")
            assert get_sdiv_mode() == "raise"

            set_sdiv_mode("Ignore Zero Conversion")
            assert get_sdiv_mode() == "ignore zero conversion"

            set_sdiv_mode("Warning Zero Conversion")
            assert get_sdiv_mode() == "warning zero conversion"

            set_sdiv_mode("Ignore Limit Conversion")
            assert get_sdiv_mode() == "ignore limit conversion"

            set_sdiv_mode("Warning Limit Conversion")
            assert get_sdiv_mode() == "warning limit conversion"


class TestSdivMode:
    """
    Define :func:`colour.algebra.common.sdiv_mode` definition unit
    tests methods.
    """

    def test_sdiv_mode(self) -> None:
        """Test :func:`colour.algebra.common.sdiv_mode` definition."""

        with sdiv_mode("Raise"):
            assert get_sdiv_mode() == "raise"

        with sdiv_mode("Ignore Zero Conversion"):
            assert get_sdiv_mode() == "ignore zero conversion"

        @sdiv_mode("Raise")
        def fn_a() -> None:
            """:func:`sdiv_mode` unit tests :func:`fn_a` definition."""

            assert get_sdiv_mode() == "raise"

        fn_a()

        @sdiv_mode("Ignore Zero Conversion")
        def fn_b() -> None:
            """:func:`sdiv_mode` unit tests :func:`fn_b` definition."""

            assert get_sdiv_mode() == "ignore zero conversion"

        fn_b()


class TestSdiv:
    """
    Define :func:`colour.algebra.common.sdiv` definition unit
    tests methods.
    """

    def test_sdiv(self, xp: ModuleType) -> None:
        """Test :func:`colour.algebra.common.sdiv` definition."""

        a = xp_asarray([0, 1, 2], xp=xp)
        b = xp_asarray([2, 1, 0], xp=xp)

        if is_numpy_namespace(xp):
            with sdiv_mode("Numpy"):
                pytest.warns(RuntimeWarning, sdiv, a, b)

        with sdiv_mode("Ignore"):
            np.testing.assert_equal(np.asarray(sdiv(a, b)), np.array([0, 1, np.inf]))

        if is_numpy_namespace(xp):
            with sdiv_mode("Warning"):
                pytest.warns(RuntimeWarning, sdiv, a, b)
                np.testing.assert_equal(
                    np.asarray(sdiv(a, b)), np.array([0, 1, np.inf])
                )

        if is_numpy_namespace(xp):
            with sdiv_mode("Raise"):
                pytest.raises(FloatingPointError, sdiv, a, b)

        with sdiv_mode("Ignore Zero Conversion"):
            np.testing.assert_equal(np.asarray(sdiv(a, b)), np.array([0, 1, 0]))

        if is_numpy_namespace(xp):
            with sdiv_mode("Warning Zero Conversion"):
                pytest.warns(RuntimeWarning, sdiv, a, b)
                np.testing.assert_equal(np.asarray(sdiv(a, b)), np.array([0, 1, 0]))

        with sdiv_mode("Ignore Limit Conversion"):
            np.testing.assert_equal(
                np.asarray(sdiv(a, b)), np.nan_to_num(np.array([0, 1, np.inf]))
            )

        if is_numpy_namespace(xp):
            with sdiv_mode("Warning Limit Conversion"):
                pytest.warns(RuntimeWarning, sdiv, a, b)
                np.testing.assert_equal(
                    np.asarray(sdiv(a, b)), np.nan_to_num(np.array([0, 1, np.inf]))
                )

        with sdiv_mode("Replace With Epsilon"):
            xp_assert_close(
                sdiv(a, b),
                np.array([0, 1, 2 / np.finfo(np.double).eps]),
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

        if is_numpy_namespace(xp):
            with sdiv_mode("Warning Replace With Epsilon"):
                pytest.warns(ColourRuntimeWarning, sdiv, a, b)
                xp_assert_close(
                    sdiv(a, b),
                    np.array([0, 1, 2 / np.finfo(np.double).eps]),
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )


class TestIsSpowEnabled:
    """
    Define :func:`colour.algebra.common.is_spow_enabled` definition unit
    tests methods.
    """

    def test_is_spow_enabled(self) -> None:
        """Test :func:`colour.algebra.common.is_spow_enabled` definition."""

        with spow_enable(True):
            assert is_spow_enabled()

        with spow_enable(False):
            assert not is_spow_enabled()


class TestSetSpowEnabled:
    """
    Define :func:`colour.algebra.common.set_spow_enable` definition unit
    tests methods.
    """

    def test_set_spow_enable(self) -> None:
        """Test :func:`colour.algebra.common.set_spow_enable` definition."""

        with spow_enable(is_spow_enabled()):
            set_spow_enable(True)
            assert is_spow_enabled()

        with spow_enable(is_spow_enabled()):
            set_spow_enable(False)
            assert not is_spow_enabled()


class TestSpowEnable:
    """
    Define :func:`colour.algebra.common.spow_enable` definition unit
    tests methods.
    """

    def test_spow_enable(self) -> None:
        """Test :func:`colour.algebra.common.spow_enable` definition."""

        with spow_enable(True):
            assert is_spow_enabled()

        with spow_enable(False):
            assert not is_spow_enabled()

        @spow_enable(True)
        def fn_a() -> None:
            """:func:`spow_enable` unit tests :func:`fn_a` definition."""

            assert is_spow_enabled()

        fn_a()

        @spow_enable(False)
        def fn_b() -> None:
            """:func:`spow_enable` unit tests :func:`fn_b` definition."""

            assert not is_spow_enabled()

        fn_b()


class TestSpow:
    """
    Define :func:`colour.algebra.common.spow` definition unit
    tests methods.
    """

    def test_spow(self, xp: ModuleType) -> None:
        """Test :func:`colour.algebra.common.spow` definition."""

        assert spow(2, 2) == 4.0

        assert spow(-2, 2) == -4.0

        xp_assert_close(
            spow(xp_asarray([2, -2, -2, 0], xp=xp), xp_asarray([2, 2, 0.15, 0], xp=xp)),
            np.array([4.00000000, -4.00000000, -1.10956947, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        with spow_enable(True):
            xp_assert_close(spow(-2, 0.15), -1.10956947, atol=TOLERANCE_ABSOLUTE_TESTS)

        with spow_enable(False):
            np.testing.assert_equal(spow(-2, 0.15), np.nan)


class TestNormaliseVector:
    """
    Define :func:`colour.algebra.common.normalise_vector` definition unit
    tests methods.
    """

    def test_normalise_vector(self, xp: ModuleType) -> None:
        """Test :func:`colour.algebra.common.normalise_vector` definition."""

        xp_assert_close(
            normalise_vector(xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp)),
            np.array([0.84197033, 0.49722560, 0.20941026]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            normalise_vector(xp_asarray([0.14222010, 0.23042768, 0.10495772], xp=xp)),
            np.array([0.48971705, 0.79344877, 0.36140872]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            normalise_vector(xp_asarray([0.07818780, 0.06157201, 0.28099326], xp=xp)),
            np.array([0.26229003, 0.20655044, 0.94262445]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestNormaliseMaximum:
    """
    Define :func:`colour.algebra.common.normalise_maximum` definition unit
    tests methods.
    """

    def test_normalise_maximum(self, xp: ModuleType) -> None:
        """Test :func:`colour.algebra.common.normalise_maximum` definition."""

        xp_assert_close(
            normalise_maximum(xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp)),
            np.array([1.00000000, 0.59055003, 0.24871454]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            normalise_maximum(
                xp_asarray(
                    [
                        [0.20654008, 0.12197225, 0.05136952],
                        [0.14222010, 0.23042768, 0.10495772],
                        [0.07818780, 0.06157201, 0.28099326],
                    ],
                    xp=xp,
                )
            ),
            np.array(
                [
                    [0.73503571, 0.43407536, 0.18281406],
                    [0.50613349, 0.82004700, 0.37352398],
                    [0.27825507, 0.21912273, 1.00000000],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            normalise_maximum(
                xp_asarray(
                    [
                        [0.20654008, 0.12197225, 0.05136952],
                        [0.14222010, 0.23042768, 0.10495772],
                        [0.07818780, 0.06157201, 0.28099326],
                    ],
                    xp=xp,
                ),
                axis=-1,
            ),
            np.array(
                [
                    [1.00000000, 0.59055003, 0.24871454],
                    [0.61720059, 1.00000000, 0.45549094],
                    [0.27825507, 0.21912273, 1.00000000],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            normalise_maximum(
                xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp), factor=10
            ),
            np.array([10.00000000, 5.90550028, 2.48714535]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            normalise_maximum(
                xp_asarray([-0.11518475, -0.10080000, 0.05089373], xp=xp)
            ),
            np.array([0.00000000, 0.00000000, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            normalise_maximum(
                xp_asarray([-0.20654008, -0.12197225, 0.05136952], xp=xp), clip=False
            ),
            np.array([-4.02067374, -2.37440899, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestVectorDot:
    """
    Define :func:`colour.algebra.common.vecmul` definition unit tests
    methods.
    """

    def test_vecmul(self, xp: ModuleType) -> None:
        """Test :func:`colour.algebra.common.vecmul` definition."""

        m = np.array(
            [
                [0.7328, 0.4296, -0.1624],
                [-0.7036, 1.6975, 0.0061],
                [0.0030, 0.0136, 0.9834],
            ]
        )
        m = xp_reshape(xp.tile(xp_asarray(m, xp=xp), (6, 1)), (6, 3, 3), xp=xp)

        v = np.array([0.20654008, 0.12197225, 0.05136952])
        v = xp.tile(xp_asarray(v, xp=xp), (6, 1))

        m_xp = xp_asarray(m, xp=xp)
        v_xp = xp_asarray(v, xp=xp)

        xp_assert_close(
            vecmul(m_xp, v_xp),
            np.array(
                [
                    [0.19540944, 0.06203965, 0.05279523],
                    [0.19540944, 0.06203965, 0.05279523],
                    [0.19540944, 0.06203965, 0.05279523],
                    [0.19540944, 0.06203965, 0.05279523],
                    [0.19540944, 0.06203965, 0.05279523],
                    [0.19540944, 0.06203965, 0.05279523],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        with array_api_enable(True):
            xp_assert_close(
                vecmul(m_xp, v_xp),
                np.array(
                    [
                        [0.19540944, 0.06203965, 0.05279523],
                        [0.19540944, 0.06203965, 0.05279523],
                        [0.19540944, 0.06203965, 0.05279523],
                        [0.19540944, 0.06203965, 0.05279523],
                        [0.19540944, 0.06203965, 0.05279523],
                        [0.19540944, 0.06203965, 0.05279523],
                    ]
                ),
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )


class TestEuclideanDistance:
    """
    Define :func:`colour.algebra.common.euclidean_distance` definition unit
    tests methods.
    """

    def test_euclidean_distance(self, xp: ModuleType) -> None:
        """Test :func:`colour.algebra.common.euclidean_distance` definition."""

        xp_assert_close(
            euclidean_distance(
                xp_asarray([100.00000000, 21.57210357, 272.22819350], xp=xp),
                xp_asarray([100.00000000, 426.67945353, 72.39590835], xp=xp),
            ),
            451.71330197,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            euclidean_distance(
                xp_asarray([100.00000000, 21.57210357, 272.22819350], xp=xp),
                xp_asarray([100.00000000, 74.05216981, 276.45318193], xp=xp),
            ),
            52.64986116,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            euclidean_distance(
                xp_asarray([100.00000000, 21.57210357, 272.22819350], xp=xp),
                xp_asarray([100.00000000, 8.32281957, -73.58297716], xp=xp),
            ),
            346.06489172,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_euclidean_distance(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.algebra.common.euclidean_distance` definition
        n-dimensional arrays support.
        """

        a = xp_asarray([100.00000000, 21.57210357, 272.22819350], xp=xp)
        b = xp_asarray([100.00000000, 426.67945353, 72.39590835], xp=xp)
        distance = np.asarray(euclidean_distance(a, b))

        a = xp.tile(xp_asarray(a, xp=xp), (6, 1))
        b = xp.tile(xp_asarray(b, xp=xp), (6, 1))
        distance = xp.tile(xp_asarray(distance, xp=xp), (6,))
        xp_assert_close(
            euclidean_distance(a, b),
            distance,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_reshape(xp_asarray(a, xp=xp), (2, 3, 3), xp=xp)
        b = xp_reshape(xp_asarray(b, xp=xp), (2, 3, 3), xp=xp)
        distance = xp_reshape(xp_asarray(distance, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            euclidean_distance(a, b),
            distance,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_euclidean_distance(self) -> None:
        """
        Test :func:`colour.algebra.common.euclidean_distance` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        euclidean_distance(cases, cases)


class TestManhattanDistance:
    """
    Define :func:`colour.algebra.common.manhattan_distance` definition unit
    tests methods.
    """

    def test_manhattan_distance(self, xp: ModuleType) -> None:
        """Test :func:`colour.algebra.common.manhattan_distance` definition."""

        xp_assert_close(
            manhattan_distance(
                xp_asarray([100.00000000, 21.57210357, 272.22819350], xp=xp),
                xp_asarray([100.00000000, 426.67945353, 72.39590835], xp=xp),
            ),
            604.93963510999993,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            manhattan_distance(
                xp_asarray([100.00000000, 21.57210357, 272.22819350], xp=xp),
                xp_asarray([100.00000000, 74.05216981, 276.45318193], xp=xp),
            ),
            56.705054670000052,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            manhattan_distance(
                xp_asarray([100.00000000, 21.57210357, 272.22819350], xp=xp),
                xp_asarray([100.00000000, 8.32281957, -73.58297716], xp=xp),
            ),
            359.06045465999995,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_manhattan_distance(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.algebra.common.manhattan_distance` definition
        n-dimensional arrays support.
        """

        a = xp_asarray([100.00000000, 21.57210357, 272.22819350], xp=xp)
        b = xp_asarray([100.00000000, 426.67945353, 72.39590835], xp=xp)
        distance = np.asarray(manhattan_distance(a, b))

        a = xp.tile(xp_asarray(a, xp=xp), (6, 1))
        b = xp.tile(xp_asarray(b, xp=xp), (6, 1))
        distance = xp.tile(xp_asarray(distance, xp=xp), (6,))
        xp_assert_close(
            manhattan_distance(a, b),
            distance,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_reshape(xp_asarray(a, xp=xp), (2, 3, 3), xp=xp)
        b = xp_reshape(xp_asarray(b, xp=xp), (2, 3, 3), xp=xp)
        distance = xp_reshape(xp_asarray(distance, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            manhattan_distance(a, b),
            distance,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_manhattan_distance(self) -> None:
        """
        Test :func:`colour.algebra.common.manhattan_distance` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        manhattan_distance(cases, cases)


class TestLinearConversion:
    """
    Define :func:`colour.algebra.common.linear_conversion` definition unit
    tests methods.
    """

    def test_linear_conversion(self, xp: ModuleType) -> None:
        """Test :func:`colour.algebra.common.linear_conversion` definition."""

        xp_assert_close(
            linear_conversion(
                xp_asarray(np.linspace(0, 1, 10), xp=xp),
                xp_asarray([0, 1], xp=xp),
                xp_asarray([1, np.pi], xp=xp),
            ),
            np.array(
                [
                    1.00000000,
                    1.23795474,
                    1.47590948,
                    1.71386422,
                    1.95181896,
                    2.18977370,
                    2.42772844,
                    2.66568318,
                    2.90363791,
                    3.14159265,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestLinstepFunction:
    """
    Define :func:`colour.algebra.common.linstep_function` definition unit
    tests methods.
    """

    def test_linstep_function(self, xp: ModuleType) -> None:
        """Test :func:`colour.algebra.common.linstep_function` definition."""

        xp_assert_close(
            linstep_function(
                xp_asarray(np.linspace(0, 1, 10), xp=xp),
                xp_asarray(np.linspace(0, 1, 10), xp=xp),
                xp_asarray(np.linspace(0, 2, 10), xp=xp),
            ),
            np.array(
                [
                    0.00000000,
                    0.12345679,
                    0.27160494,
                    0.44444444,
                    0.64197531,
                    0.86419753,
                    1.11111111,
                    1.38271605,
                    1.67901235,
                    2.00000000,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            linstep_function(
                xp_asarray(np.linspace(0, 2, 10), xp=xp),
                xp_asarray(np.linspace(0.25, 0.5, 10), xp=xp),
                xp_asarray(np.linspace(0.5, 0.75, 10), xp=xp),
                clip=True,
            ),
            np.array(
                [
                    0.25000000,
                    0.33333333,
                    0.41666667,
                    0.50000000,
                    0.58333333,
                    0.63888889,
                    0.66666667,
                    0.69444444,
                    0.72222222,
                    0.75000000,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestSmoothstepFunction:
    """
    Define :func:`colour.algebra.common.smoothstep_function` definition unit
    tests methods.
    """

    def test_smoothstep_function(self, xp: ModuleType) -> None:
        """Test :func:`colour.algebra.common.smoothstep_function` definition."""

        assert smoothstep_function(0.5) == 0.5
        assert smoothstep_function(0.25) == 0.15625
        assert smoothstep_function(0.75) == 0.84375

        x = xp_asarray(np.linspace(-2, 2, 5), xp=xp)
        xp_assert_close(
            smoothstep_function(x),
            np.array([28.00000, 5.00000, 0.00000, 1.00000, -4.00000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            smoothstep_function(x, -2, 2, clip=True),
            np.array([0.00000, 0.15625, 0.50000, 0.84375, 1.00000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestIsIdentity:
    """
    Define :func:`colour.algebra.common.is_identity` definition unit tests
    methods.
    """

    def test_is_identity(self, xp: ModuleType) -> None:
        """Test :func:`colour.algebra.common.is_identity` definition."""

        assert is_identity(
            xp_reshape(xp_asarray([1, 0, 0, 0, 1, 0, 0, 0, 1], xp=xp), (3, 3), xp=xp)
        )

        assert not is_identity(
            xp_reshape(xp_asarray([1, 2, 0, 0, 1, 0, 0, 0, 1], xp=xp), (3, 3), xp=xp)
        )

        assert is_identity(xp_reshape(xp_asarray([1, 0, 0, 1], xp=xp), (2, 2), xp=xp))

        assert not is_identity(
            xp_reshape(xp_asarray([1, 2, 0, 1], xp=xp), (2, 2), xp=xp)
        )


class TestEigenDecomposition:
    """
    Define :func:`colour.algebra.common.eigen_decomposition` definition unit
    tests methods.
    """

    def test_is_identity(self, xp: ModuleType) -> None:
        """Test :func:`colour.algebra.common.eigen_decomposition` definition."""

        a = xp_asarray(np.diag([1, 2, 3]), xp=xp)

        w, v = eigen_decomposition(a)
        np.testing.assert_equal(np.asarray(w), np.array([3.0, 2.0, 1.0]))
        np.testing.assert_equal(
            np.asarray(v),
            np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        )

        w, v = eigen_decomposition(a, 1)
        np.testing.assert_equal(np.asarray(w), np.array([3.0]))
        np.testing.assert_equal(np.asarray(v), np.array([[0.0], [0.0], [1.0]]))

        w, v = eigen_decomposition(a, descending_order=False)
        np.testing.assert_equal(np.asarray(w), np.array([1.0, 2.0, 3.0]))
        np.testing.assert_equal(
            np.asarray(v),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        )

        w, v = eigen_decomposition(a, covariance_matrix=True)
        np.testing.assert_equal(np.asarray(w), np.array([9.0, 4.0, 1.0]))
        np.testing.assert_equal(
            np.asarray(v),
            np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        )

        w, v = eigen_decomposition(a, descending_order=False, covariance_matrix=True)
        np.testing.assert_equal(np.asarray(w), np.array([1.0, 4.0, 9.0]))
        np.testing.assert_equal(
            np.asarray(v),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        )
