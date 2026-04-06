"""Define the unit tests for the :mod:`colour.models.igpgtg` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import IgPgTg_to_XYZ, XYZ_to_IgPgTg
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
    "TestXYZ_to_IgPgTg",
    "TestIgPgTg_to_XYZ",
]


class TestXYZ_to_IgPgTg:
    """
    Define :func:`colour.models.igpgtg.XYZ_to_IgPgTg` definition unit tests
    methods.
    """

    def test_XYZ_to_IgPgTg(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.igpgtg.XYZ_to_IgPgTg` definition."""

        xp_assert_close(
            XYZ_to_IgPgTg(xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp)),
            np.array([0.42421258, 0.18632491, 0.10689223]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_IgPgTg(xp_asarray([0.14222010, 0.23042768, 0.10495772], xp=xp)),
            np.array([0.50912820, -0.14804331, 0.11921472]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_IgPgTg(xp_asarray([0.07818780, 0.06157201, 0.28099326], xp=xp)),
            np.array([0.29095152, -0.04057508, -0.18220795]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_IgPgTg(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.igpgtg.XYZ_to_IgPgTg` definition
        n-dimensional support.
        """

        XYZ = xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp)
        IgPgTg = np.asarray(XYZ_to_IgPgTg(XYZ))

        XYZ = xp.tile(xp_asarray(XYZ, xp=xp), (6, 1))
        IgPgTg = xp.tile(xp_asarray(IgPgTg, xp=xp), (6, 1))
        xp_assert_close(XYZ_to_IgPgTg(XYZ), IgPgTg, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = xp_reshape(xp_asarray(XYZ, xp=xp), (2, 3, 3), xp=xp)
        IgPgTg = xp_reshape(xp_asarray(IgPgTg, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(XYZ_to_IgPgTg(XYZ), IgPgTg, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_XYZ_to_IgPgTg(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.igpgtg.XYZ_to_IgPgTg` definition domain and
        range scale support.
        """

        XYZ = xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp)
        IgPgTg = np.asarray(XYZ_to_IgPgTg(XYZ))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_IgPgTg(XYZ * factor),
                    IgPgTg * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_IgPgTg(self) -> None:
        """
        Test :func:`colour.models.igpgtg.XYZ_to_IgPgTg` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_IgPgTg(cases)


class TestIgPgTg_to_XYZ:
    """
    Define :func:`colour.models.igpgtg.IgPgTg_to_XYZ` definition unit tests
    methods.
    """

    def test_IgPgTg_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.igpgtg.IgPgTg_to_XYZ` definition."""

        xp_assert_close(
            IgPgTg_to_XYZ(xp_asarray([0.42421258, 0.18632491, 0.10689223], xp=xp)),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            IgPgTg_to_XYZ(xp_asarray([0.50912820, -0.14804331, 0.11921472], xp=xp)),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            IgPgTg_to_XYZ(xp_asarray([0.29095152, -0.04057508, -0.18220795], xp=xp)),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_IgPgTg_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.igpgtg.IgPgTg_to_XYZ` definition
        n-dimensional support.
        """

        IgPgTg = xp_asarray([0.42421258, 0.18632491, 0.10689223], xp=xp)
        XYZ = np.asarray(IgPgTg_to_XYZ(IgPgTg))

        IgPgTg = xp.tile(xp_asarray(IgPgTg, xp=xp), (6, 1))
        XYZ = xp.tile(xp_asarray(XYZ, xp=xp), (6, 1))
        xp_assert_close(IgPgTg_to_XYZ(IgPgTg), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        IgPgTg = xp_reshape(xp_asarray(IgPgTg, xp=xp), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp_asarray(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(IgPgTg_to_XYZ(IgPgTg), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_IgPgTg_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.igpgtg.IgPgTg_to_XYZ` definition domain and
        range scale support.
        """

        IgPgTg = xp_asarray([0.42421258, 0.18632491, 0.10689223], xp=xp)
        XYZ = np.asarray(IgPgTg_to_XYZ(IgPgTg))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    IgPgTg_to_XYZ(IgPgTg * factor),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_IgPgTg_to_XYZ(self) -> None:
        """
        Test :func:`colour.models.igpgtg.IgPgTg_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        IgPgTg_to_XYZ(cases)
