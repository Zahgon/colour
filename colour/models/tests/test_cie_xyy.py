"""Define the unit tests for the :mod:`colour.models.cie_xyy` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import (
    XYZ_to_xy,
    XYZ_to_xyY,
    xy_to_xyY,
    xy_to_XYZ,
    xyY_to_xy,
    xyY_to_XYZ,
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
    "TestXYZ_to_xyY",
    "TestxyY_to_XYZ",
    "TestxyY_to_xy",
    "Testxy_to_xyY",
    "TestXYZ_to_xy",
    "Testxy_to_XYZ",
]


class TestXYZ_to_xyY:
    """
    Define :func:`colour.models.cie_xyy.XYZ_to_xyY` definition unit tests
    methods.
    """

    def test_XYZ_to_xyY(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cie_xyy.XYZ_to_xyY` definition."""

        xp_assert_close(
            XYZ_to_xyY(xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp)),
            np.array([0.54369557, 0.32107944, 0.12197225]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_xyY(xp_asarray([0.14222010, 0.23042768, 0.10495772], xp=xp)),
            np.array([0.29777735, 0.48246446, 0.23042768]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_xyY(xp_asarray([0.07818780, 0.06157201, 0.28099326], xp=xp)),
            np.array([0.18582823, 0.14633764, 0.06157201]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_xyY(xp_asarray([0.00000000, 0.00000000, 1.00000000], xp=xp)),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_xyY(
                xp_asarray(
                    [
                        [0.20654008, 0.12197225, 0.05136952],
                        [0.00000000, 0.00000000, 0.00000000],
                        [0.00000000, 1.00000000, 0.00000000],
                    ],
                    xp=xp,
                )
            ),
            np.array(
                [
                    [0.54369557, 0.32107944, 0.12197225],
                    [0.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 1.00000000, 1.00000000],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_xyY(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_xyy.XYZ_to_xyY` definition n-dimensional
        support.
        """

        XYZ = xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp)
        xyY = np.asarray(XYZ_to_xyY(XYZ))

        XYZ = xp.tile(xp_asarray(XYZ, xp=xp), (6, 1))
        xyY = xp.tile(xp_asarray(xyY, xp=xp), (6, 1))
        xp_assert_close(XYZ_to_xyY(XYZ), xyY, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = xp_reshape(xp_asarray(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xyY = xp_reshape(xp_asarray(xyY, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(XYZ_to_xyY(XYZ), xyY, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_XYZ_to_xyY(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_xyy.XYZ_to_xyY` definition domain and
        range scale support.
        """

        XYZ = xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp)
        xyY = np.asarray(XYZ_to_xyY(XYZ))
        XYZ = xp_reshape(xp.tile(xp_asarray(XYZ, xp=xp), (6, 1)), (2, 3, 3), xp=xp)
        xyY = xp_reshape(xp.tile(xp_asarray(xyY, xp=xp), (6, 1)), (2, 3, 3), xp=xp)

        d_r = (
            ("reference", 1, 1),
            ("1", 1, 1),
            ("100", 100, np.array([1, 1, 100])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_xyY(XYZ * factor_a),
                    xyY * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_xyY(self) -> None:
        """Test :func:`colour.models.cie_xyy.XYZ_to_xyY` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_xyY(cases)


class TestxyY_to_XYZ:
    """
    Define :func:`colour.models.cie_xyy.xyY_to_XYZ` definition unit tests
    methods.
    """

    def test_xyY_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cie_xyy.xyY_to_XYZ` definition."""

        xp_assert_close(
            xyY_to_XYZ(xp_asarray([0.54369557, 0.32107944, 0.12197225], xp=xp)),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xyY_to_XYZ(xp_asarray([0.29777735, 0.48246446, 0.23042768], xp=xp)),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xyY_to_XYZ(xp_asarray([0.18582823, 0.14633764, 0.06157201], xp=xp)),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xyY_to_XYZ(xp_asarray([0.34567, 0.3585, 0.00000000], xp=xp)),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xyY_to_XYZ(
                xp_asarray(
                    [
                        [0.54369557, 0.32107944, 0.12197225],
                        [0.31270000, 0.32900000, 0.00000000],
                        [0.00000000, 1.00000000, 1.00000000],
                    ],
                    xp=xp,
                )
            ),
            np.array(
                [
                    [0.20654008, 0.12197225, 0.05136952],
                    [0.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 1.00000000, 0.00000000],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_xyY_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_xyy.xyY_to_XYZ` definition n-dimensional
        support.
        """

        xyY = xp_asarray([0.54369557, 0.32107944, 0.12197225], xp=xp)
        XYZ = np.asarray(xyY_to_XYZ(xyY))

        xyY = xp.tile(xp_asarray(xyY, xp=xp), (6, 1))
        XYZ = xp.tile(xp_asarray(XYZ, xp=xp), (6, 1))
        xp_assert_close(xyY_to_XYZ(xyY), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        xyY = xp_reshape(xp_asarray(xyY, xp=xp), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp_asarray(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(xyY_to_XYZ(xyY), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_xyY_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_xyy.xyY_to_XYZ` definition domain and
        range scale support.
        """

        xyY = xp_asarray([0.54369557, 0.32107944, 0.12197225], xp=xp)
        XYZ = np.asarray(xyY_to_XYZ(xyY))
        xyY = xp_reshape(xp.tile(xp_asarray(xyY, xp=xp), (6, 1)), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp.tile(xp_asarray(XYZ, xp=xp), (6, 1)), (2, 3, 3), xp=xp)

        d_r = (
            ("reference", 1, 1),
            ("1", 1, 1),
            ("100", np.array([1, 1, 100]), 100),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    xyY_to_XYZ(xyY * factor_a),
                    XYZ * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_xyY_to_XYZ(self) -> None:
        """Test :func:`colour.models.cie_xyy.xyY_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        xyY_to_XYZ(cases)


class TestxyY_to_xy:
    """
    Define :func:`colour.models.cie_xyy.xyY_to_xy` definition unit tests
    methods.
    """

    def test_xyY_to_xy(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cie_xyy.xyY_to_xy` definition."""

        xp_assert_close(
            xyY_to_xy(xp_asarray([0.54369557, 0.32107944, 0.12197225], xp=xp)),
            np.array([0.54369557, 0.32107944]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xyY_to_xy(xp_asarray([0.29777735, 0.48246446, 0.23042768], xp=xp)),
            np.array([0.29777735, 0.48246446]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xyY_to_xy(xp_asarray([0.18582823, 0.14633764, 0.06157201], xp=xp)),
            np.array([0.18582823, 0.14633764]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xyY_to_xy(xp_asarray([0.31270, 0.32900], xp=xp)),
            np.array([0.31270000, 0.32900000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_xyY_to_xy(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_xyy.xyY_to_xy` definition n-dimensional
        support.
        """

        xyY = xp_asarray([0.54369557, 0.32107944, 0.12197225], xp=xp)
        xy = np.asarray(xyY_to_xy(xyY))

        xyY = xp.tile(xp_asarray(xyY, xp=xp), (6, 1))
        xy = xp.tile(xp_asarray(xy, xp=xp), (6, 1))
        xp_assert_close(xyY_to_xy(xyY), xy, atol=TOLERANCE_ABSOLUTE_TESTS)

        xyY = xp_reshape(xp_asarray(xyY, xp=xp), (2, 3, 3), xp=xp)
        xy = xp_reshape(xp_asarray(xy, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(xyY_to_xy(xyY), xy, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_xyY_to_xy(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_xyy.xyY_to_xy` definition domain and
        range scale support.
        """

        xyY = xp_asarray([0.54369557, 0.32107944, 0.12197225], xp=xp)
        xy = np.asarray(xyY_to_xy(xyY))
        xyY = xp_reshape(xp.tile(xp_asarray(xyY, xp=xp), (6, 1)), (2, 3, 3), xp=xp)
        xy = xp_reshape(xp.tile(xp_asarray(xy, xp=xp), (6, 1)), (2, 3, 2), xp=xp)

        d_r = (
            ("reference", 1, 1),
            ("1", 1, 1),
            ("100", np.array([1, 1, 100]), 1),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    xyY_to_xy(xyY * factor_a),
                    xy * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_xyY_to_xy(self) -> None:
        """Test :func:`colour.models.cie_xyy.xyY_to_xy` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        xyY_to_xy(cases)


class Testxy_to_xyY:
    """
    Define :func:`colour.models.cie_xyy.xy_to_xyY` definition unit tests
    methods.
    """

    def test_xy_to_xyY(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cie_xyy.xy_to_xyY` definition."""

        xp_assert_close(
            xy_to_xyY(xp_asarray([0.54369557, 0.32107944], xp=xp)),
            np.array([0.54369557, 0.32107944, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xy_to_xyY(xp_asarray([0.29777735, 0.48246446], xp=xp)),
            np.array([0.29777735, 0.48246446, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xy_to_xyY(xp_asarray([0.18582823, 0.14633764], xp=xp)),
            np.array([0.18582823, 0.14633764, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xy_to_xyY(xp_asarray([0.31270000, 0.32900000, 1.00000000], xp=xp)),
            np.array([0.31270000, 0.32900000, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xy_to_xyY(xp_asarray([0.31270000, 0.32900000], xp=xp), 100),
            np.array([0.31270000, 0.32900000, 100.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_xy_to_xyY(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_xyy.xy_to_xyY` definition n-dimensional
        support.
        """

        xy = xp_asarray([0.54369557, 0.32107944], xp=xp)
        xyY = np.asarray(xy_to_xyY(xy))

        xy = xp.tile(xp_asarray(xy, xp=xp), (6, 1))
        xyY = xp.tile(xp_asarray(xyY, xp=xp), (6, 1))
        xp_assert_close(xy_to_xyY(xy), xyY, atol=TOLERANCE_ABSOLUTE_TESTS)

        xy = xp_reshape(xp_asarray(xy, xp=xp), (2, 3, 2), xp=xp)
        xyY = xp_reshape(xp_asarray(xyY, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(xy_to_xyY(xy), xyY, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_xy_to_xyY(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_xyy.xy_to_xyY` definition domain and
        range scale support.
        """

        xy = xp_asarray([0.54369557, 0.32107944, 0.12197225], xp=xp)
        xyY = np.asarray(xy_to_xyY(xy))
        xy = xp_reshape(xp.tile(xp_asarray(xy, xp=xp), (6, 1)), (2, 3, 3), xp=xp)
        xyY = xp_reshape(xp.tile(xp_asarray(xyY, xp=xp), (6, 1)), (2, 3, 3), xp=xp)

        d_r = (
            ("reference", 1, 1),
            ("1", 1, 1),
            (
                "100",
                np.array([1, 1, 100]),
                np.array([1, 1, 100]),
            ),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    xy_to_xyY(xy * factor_a),
                    xyY * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_xy_to_xyY(self) -> None:
        """Test :func:`colour.models.cie_xyy.xy_to_xyY` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        xy_to_xyY(cases)


class TestXYZ_to_xy:
    """
    Define :func:`colour.models.cie_xyy.XYZ_to_xy` definition unit tests
    methods.
    """

    def test_XYZ_to_xy(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cie_xyy.XYZ_to_xy` definition."""

        xp_assert_close(
            XYZ_to_xy(xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp)),
            np.array([0.54369557, 0.32107944]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_xy(xp_asarray([0.14222010, 0.23042768, 0.10495772], xp=xp)),
            np.array([0.29777735, 0.48246446]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_xy(xp_asarray([0.07818780, 0.06157201, 0.28099326], xp=xp)),
            np.array([0.18582823, 0.14633764]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_xy(xp_asarray([0.00000000, 0.00000000, 0.00000000], xp=xp)),
            np.array([0.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_xy(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_xyy.XYZ_to_xy` definition n-dimensional
        support.
        """

        XYZ = xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp)
        xy = np.asarray(XYZ_to_xy(XYZ))

        XYZ = xp.tile(xp_asarray(XYZ, xp=xp), (6, 1))
        xy = xp.tile(xp_asarray(xy, xp=xp), (6, 1))
        xp_assert_close(XYZ_to_xy(XYZ), xy, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = xp_reshape(xp_asarray(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xy = xp_reshape(xp_asarray(xy, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(XYZ_to_xy(XYZ), xy, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_XYZ_to_xy(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_xyy.XYZ_to_xy` definition domain and
        range scale support.
        """

        XYZ = xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp)
        xy = np.asarray(XYZ_to_xy(XYZ))
        XYZ = xp_reshape(xp.tile(xp_asarray(XYZ, xp=xp), (6, 1)), (2, 3, 3), xp=xp)
        xy = xp_reshape(xp.tile(xp_asarray(xy, xp=xp), (6, 1)), (2, 3, 2), xp=xp)

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_xy(XYZ * factor),
                    xy,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_xy(self) -> None:
        """Test :func:`colour.models.cie_xyy.XYZ_to_xy` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_xy(cases)


class Testxy_to_XYZ:
    """
    Define :func:`colour.models.cie_xyy.xy_to_XYZ` definition unit tests
    methods.
    """

    def test_xy_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cie_xyy.xy_to_XYZ` definition."""

        xp_assert_close(
            xy_to_XYZ(xp_asarray([0.54369557, 0.32107944], xp=xp)),
            np.array([1.69333661, 1.00000000, 0.42115742]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xy_to_XYZ(xp_asarray([0.29777735, 0.48246446], xp=xp)),
            np.array([0.61720059, 1.00000000, 0.45549094]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xy_to_XYZ(xp_asarray([0.18582823, 0.14633764], xp=xp)),
            np.array([1.26985942, 1.00000000, 4.56365245]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xy_to_XYZ(xp_asarray([0.31270000, 0.32900000], xp=xp)),
            np.array([0.95045593, 1.00000000, 1.08905775]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_xy_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_xyy.xy_to_XYZ` definition n-dimensional
        support.
        """

        xy = xp_asarray([0.54369557, 0.32107944], xp=xp)
        XYZ = np.asarray(xy_to_XYZ(xy))

        xy = xp.tile(xp_asarray(xy, xp=xp), (6, 1))
        XYZ = xp.tile(xp_asarray(XYZ, xp=xp), (6, 1))
        xp_assert_close(xy_to_XYZ(xy), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        xy = xp_reshape(xp_asarray(xy, xp=xp), (2, 3, 2), xp=xp)
        XYZ = xp_reshape(xp_asarray(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(xy_to_XYZ(xy), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_xy_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_xyy.xy_to_XYZ` definition domain and
        range scale support.
        """

        xy = xp_asarray([0.54369557, 0.32107944, 0.12197225], xp=xp)
        XYZ = np.asarray(xy_to_XYZ(xy))
        xy = xp_reshape(xp.tile(xp_asarray(xy, xp=xp), (6, 1)), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp.tile(xp_asarray(XYZ, xp=xp), (6, 1)), (2, 3, 3), xp=xp)

        d_r = (
            ("reference", 1, 1),
            ("1", 1, 1),
            ("100", np.array([1, 1, 100]), 100),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    xy_to_XYZ(xy * factor_a),
                    XYZ * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_xy_to_XYZ(self) -> None:
        """Test :func:`colour.models.cie_xyy.xy_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        xy_to_XYZ(cases)
