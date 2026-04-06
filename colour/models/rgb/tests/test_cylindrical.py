"""Define the unit tests for the :mod:`colour.models.rgb.cylindrical` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.cylindrical import (
    HCL_to_RGB,
    HSL_to_RGB,
    HSV_to_RGB,
    RGB_to_HCL,
    RGB_to_HSL,
    RGB_to_HSV,
)
from colour.utilities import (
    domain_range_scale,
    ignore_numpy_errors,
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
    "TestRGB_to_HSV",
    "TestHSV_to_RGB",
    "TestRGB_to_HSL",
    "TestHSL_to_RGB",
    "TestRGB_to_HCL",
    "TestHCL_to_RGB",
]


class TestRGB_to_HSV:
    """
    Define :func:`colour.models.rgb.cylindrical.RGB_to_HSV` definition unit
    tests methods.
    """

    def test_RGB_to_HSV(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.cylindrical.RGB_to_HSV` definition."""

        xp_assert_close(
            RGB_to_HSV(xp_asarray([0.45620519, 0.03081071, 0.04091952], xp=xp)),
            np.array([0.99603944, 0.93246304, 0.45620519]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_HSV(xp_asarray([0.00000000, 0.00000000, 0.00000000], xp=xp)),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_HSV(xp_asarray([1.00000000, 1.00000000, 1.00000000], xp=xp)),
            np.array([0.00000000, 0.00000000, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_HSV(xp_asarray([0.00000000, 1.00000000, 1.00000000], xp=xp)),
            np.array([0.50000000, 1.00000000, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_RGB_to_HSV(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cylindrical.RGB_to_HSV` definition
        n-dimensional arrays support.
        """

        RGB = xp_asarray([0.45620519, 0.03081071, 0.04091952], xp=xp)
        HSV = np.asarray(RGB_to_HSV(RGB))

        RGB = xp.tile(xp_asarray(RGB, xp=xp), (6, 1))
        HSV = xp.tile(xp_asarray(HSV, xp=xp), (6, 1))
        xp_assert_close(RGB_to_HSV(RGB), HSV, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = xp_reshape(xp_asarray(RGB, xp=xp), (2, 3, 3), xp=xp)
        HSV = xp_reshape(xp_asarray(HSV, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(RGB_to_HSV(RGB), HSV, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_RGB_to_HSV(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cylindrical.RGB_to_HSV` definition
        domain and range scale support.
        """

        RGB = xp_asarray([0.45620519, 0.03081071, 0.04091952], xp=xp)
        HSV = np.asarray(RGB_to_HSV(RGB))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    RGB_to_HSV(RGB * factor),
                    HSV * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_RGB_to_HSV(self) -> None:
        """
        Test :func:`colour.models.rgb.cylindrical.RGB_to_HSV` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_HSV(cases)


class TestHSV_to_RGB:
    """
    Define :func:`colour.models.rgb.cylindrical.HSV_to_RGB` definition unit
    tests methods.
    """

    def test_HSV_to_RGB(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.cylindrical.HSV_to_RGB` definition."""

        xp_assert_close(
            HSV_to_RGB(xp_asarray([0.99603944, 0.93246304, 0.45620519], xp=xp)),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            HSV_to_RGB(xp_asarray([0.00000000, 0.00000000, 0.00000000], xp=xp)),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            HSV_to_RGB(xp_asarray([0.00000000, 0.00000000, 1.00000000], xp=xp)),
            np.array([1.00000000, 1.00000000, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            HSV_to_RGB(xp_asarray([0.50000000, 1.00000000, 1.00000000], xp=xp)),
            np.array([0.00000000, 1.00000000, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_HSV_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cylindrical.HSV_to_RGB` definition
        n-dimensional arrays support.
        """

        HSV = xp_asarray([0.99603944, 0.93246304, 0.45620519], xp=xp)
        RGB = np.asarray(HSV_to_RGB(HSV))

        HSV = xp.tile(xp_asarray(HSV, xp=xp), (6, 1))
        RGB = xp.tile(xp_asarray(RGB, xp=xp), (6, 1))
        xp_assert_close(HSV_to_RGB(HSV), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

        HSV = xp_reshape(xp_asarray(HSV, xp=xp), (2, 3, 3), xp=xp)
        RGB = xp_reshape(xp_asarray(RGB, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(HSV_to_RGB(HSV), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_HSV_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cylindrical.HSV_to_RGB` definition
        domain and range scale support.
        """

        HSV = xp_asarray([0.99603944, 0.93246304, 0.45620519], xp=xp)
        RGB = np.asarray(HSV_to_RGB(HSV))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    HSV_to_RGB(HSV * factor),
                    RGB * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_HSV_to_RGB(self) -> None:
        """
        Test :func:`colour.models.rgb.cylindrical.HSV_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        HSV_to_RGB(cases)


class TestRGB_to_HSL:
    """
    Define :func:`colour.models.rgb.cylindrical.RGB_to_HSL` definition unit
    tests methods.
    """

    def test_RGB_to_HSL(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.cylindrical.RGB_to_HSL` definition."""

        xp_assert_close(
            RGB_to_HSL(xp_asarray([0.45620519, 0.03081071, 0.04091952], xp=xp)),
            np.array([0.99603944, 0.87347144, 0.24350795]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_HSL(xp_asarray([0.00000000, 0.00000000, 0.00000000], xp=xp)),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_HSL(xp_asarray([1.00000000, 1.00000000, 1.00000000], xp=xp)),
            np.array([0.00000000, 0.00000000, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_HSL(xp_asarray([1.00000000, 0.00000000, 0.00000000], xp=xp)),
            np.array([0.00000000, 1.00000000, 0.50000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_RGB_to_HSL(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cylindrical.RGB_to_HSL` definition
        n-dimensional arrays support.
        """

        RGB = xp_asarray([0.45620519, 0.03081071, 0.04091952], xp=xp)
        HSL = np.asarray(RGB_to_HSL(RGB))

        RGB = xp.tile(xp_asarray(RGB, xp=xp), (6, 1))
        HSL = xp.tile(xp_asarray(HSL, xp=xp), (6, 1))
        xp_assert_close(RGB_to_HSL(RGB), HSL, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = xp_reshape(xp_asarray(RGB, xp=xp), (2, 3, 3), xp=xp)
        HSL = xp_reshape(xp_asarray(HSL, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(RGB_to_HSL(RGB), HSL, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_RGB_to_HSL(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cylindrical.RGB_to_HSL` definition
        domain and range scale support.
        """

        RGB = xp_asarray([0.45620519, 0.03081071, 0.04091952], xp=xp)
        HSL = np.asarray(RGB_to_HSL(RGB))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    RGB_to_HSL(RGB * factor),
                    HSL * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_RGB_to_HSL(self) -> None:
        """
        Test :func:`colour.models.rgb.cylindrical.RGB_to_HSL` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_HSL(cases)


class TestHSL_to_RGB:
    """
    Define :func:`colour.models.rgb.cylindrical.HSL_to_RGB` definition unit
    tests methods.
    """

    def test_HSL_to_RGB(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.cylindrical.HSL_to_RGB` definition."""

        xp_assert_close(
            HSL_to_RGB(xp_asarray([0.99603944, 0.87347144, 0.24350795], xp=xp)),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            HSL_to_RGB(xp_asarray([0.00000000, 0.00000000, 0.00000000], xp=xp)),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            HSL_to_RGB(xp_asarray([0.00000000, 0.00000000, 1.00000000], xp=xp)),
            np.array([1.00000000, 1.00000000, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            HSL_to_RGB(xp_asarray([0.00000000, 1.00000000, 0.50000000], xp=xp)),
            np.array([1.00000000, 0.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_HSL_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cylindrical.HSL_to_RGB` definition
        n-dimensional arrays support.
        """

        HSL = xp_asarray([0.99603944, 0.87347144, 0.24350795], xp=xp)
        RGB = np.asarray(HSL_to_RGB(HSL))

        HSL = xp.tile(xp_asarray(HSL, xp=xp), (6, 1))
        RGB = xp.tile(xp_asarray(RGB, xp=xp), (6, 1))
        xp_assert_close(HSL_to_RGB(HSL), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

        HSL = xp_reshape(xp_asarray(HSL, xp=xp), (2, 3, 3), xp=xp)
        RGB = xp_reshape(xp_asarray(RGB, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(HSL_to_RGB(HSL), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_HSL_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cylindrical.HSL_to_RGB` definition
        domain and range scale support.
        """

        HSL = xp_asarray([0.99603944, 0.87347144, 0.24350795], xp=xp)
        RGB = np.asarray(HSL_to_RGB(HSL))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    HSL_to_RGB(HSL * factor),
                    RGB * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_HSL_to_RGB(self) -> None:
        """
        Test :func:`colour.models.rgb.cylindrical.HSL_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        HSL_to_RGB(cases)


class TestRGB_to_HCL:
    """
    Define :func:`colour.models.rgb.cylindrical.RGB_to_HCL` definition unit
    tests methods.
    """

    def test_RGB_to_HCL(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.cylindrical.RGB_to_HCL` definition."""

        xp_assert_close(
            RGB_to_HCL(xp_asarray([0.45620519, 0.03081071, 0.04091952], xp=xp)),
            np.array([-0.03167854, 0.2841715, 0.22859647]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_HCL(xp_asarray([1.00000000, 2.00000000, 0.50000000], xp=xp)),
            np.array([1.83120102, 1.0075282, 1.00941024]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_HCL(xp_asarray([2.00000000, 1.00000000, 0.50000000], xp=xp)),
            np.array([0.30909841, 1.0075282, 1.00941024]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_HCL(xp_asarray([0.50000000, 1.00000000, 2.00000000], xp=xp)),
            np.array([-2.40349351, 1.0075282, 1.00941024]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_RGB_to_HCL(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cylindrical.RGB_to_HCL` definition
        n-dimensional arrays support.
        """

        RGB = xp_asarray([0.45620519, 0.03081071, 0.04091952], xp=xp)
        HCL = np.asarray(RGB_to_HCL(RGB))

        RGB = xp.tile(xp_asarray(RGB, xp=xp), (6, 1))
        HCL = xp.tile(xp_asarray(HCL, xp=xp), (6, 1))
        xp_assert_close(RGB_to_HCL(RGB), HCL, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = xp_reshape(xp_asarray(RGB, xp=xp), (2, 3, 3), xp=xp)
        HCL = xp_reshape(xp_asarray(HCL, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(RGB_to_HCL(RGB), HCL, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_RGB_to_HCL(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cylindrical.RGB_to_HCL` definition
        domain and range scale support.
        """

        RGB = xp_asarray([0.45620519, 0.03081071, 0.04091952], xp=xp)
        HCL = np.asarray(RGB_to_HCL(RGB))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    RGB_to_HCL(RGB * factor),
                    HCL * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_RGB_to_HCL(self) -> None:
        """
        Test :func:`colour.models.rgb.cylindrical.RGB_to_HCL` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_HCL(cases)


class TestHCL_to_RGB:
    """
    Define :func:`colour.models.rgb.cylindrical.HCL_to_RGB` definition unit
    tests methods.
    """

    def test_HCL_to_RGB(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.cylindrical.HCL_to_RGB` definition."""

        xp_assert_close(
            HCL_to_RGB(xp_asarray([-0.03167854, 0.28417150, 0.22859647], xp=xp)),
            np.array([0.45620333, 0.03081048, 0.04091925]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            HCL_to_RGB(xp_asarray([1.00000000, 2.00000000, 0.50000000], xp=xp)),
            np.array([0.92186029, 0.71091922, -2.26364935]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            HCL_to_RGB(xp_asarray([2.00000000, 1.00000000, 0.50000000], xp=xp)),
            np.array([-0.31368585, 1.00732462, -0.51534497]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            HCL_to_RGB(xp_asarray([0.50000000, 1.00000000, 2.00000000], xp=xp)),
            np.array([3.88095422, 3.11881934, 2.40881719]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_HCL_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cylindrical.HCL_to_RGB` definition
        n-dimensional arrays support.
        """

        HCL = xp_asarray([0.99603944, 0.87347144, 0.24350795], xp=xp)
        RGB = np.asarray(HCL_to_RGB(HCL))

        HCL = xp.tile(xp_asarray(HCL, xp=xp), (6, 1))
        RGB = xp.tile(xp_asarray(RGB, xp=xp), (6, 1))
        xp_assert_close(HCL_to_RGB(HCL), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

        HCL = xp_reshape(xp_asarray(HCL, xp=xp), (2, 3, 3), xp=xp)
        RGB = xp_reshape(xp_asarray(RGB, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(HCL_to_RGB(HCL), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_HCL_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cylindrical.HCL_to_RGB` definition
        domain and range scale support.
        """

        HCL = xp_asarray([0.99603944, 0.87347144, 0.24350795], xp=xp)
        RGB = np.asarray(HCL_to_RGB(HCL))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    HCL_to_RGB(HCL * factor),
                    RGB * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_HCL_to_RGB(self) -> None:
        """
        Test :func:`colour.models.rgb.cylindrical.HCL_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        HCL_to_RGB(cases)
