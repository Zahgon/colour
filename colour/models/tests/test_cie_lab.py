"""Define the unit tests for the :mod:`colour.models.cie_lab` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import Lab_to_XYZ, XYZ_to_Lab
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
    "TestXYZ_to_Lab",
    "TestLab_to_XYZ",
]


class TestXYZ_to_Lab:
    """
    Define :func:`colour.models.cie_lab.XYZ_to_Lab` definition unit tests
    methods.
    """

    def test_XYZ_to_Lab(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cie_lab.XYZ_to_Lab` definition."""

        xp_assert_close(
            XYZ_to_Lab(xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp)),
            np.array([41.52787529, 52.63858304, 26.92317922]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Lab(xp_asarray([0.14222010, 0.23042768, 0.10495772], xp=xp)),
            np.array([55.11636304, -41.08791787, 30.91825778]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Lab(xp_asarray([0.07818780, 0.06157201, 0.28099326], xp=xp)),
            np.array([29.80565520, 20.01830466, -48.34913874]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Lab(
                xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp),
                xp_asarray([0.44757, 0.40745], xp=xp),
            ),
            np.array([41.52787529, 38.48089305, -5.73295122]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Lab(
                xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp),
                xp_asarray([0.34570, 0.35850], xp=xp),
            ),
            np.array([41.52787529, 51.19354174, 19.91843098]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Lab(
                xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp),
                xp_asarray([0.34570, 0.35850, 1.00000], xp=xp),
            ),
            np.array([41.52787529, 51.19354174, 19.91843098]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_Lab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_lab.XYZ_to_Lab` definition n-dimensional
        support.
        """

        XYZ = xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp)
        illuminant = xp_asarray([0.31270, 0.32900], xp=xp)
        Lab = np.asarray(XYZ_to_Lab(XYZ, illuminant))

        XYZ = xp.tile(xp_asarray(XYZ, xp=xp), (6, 1))
        Lab = xp.tile(xp_asarray(Lab, xp=xp), (6, 1))
        xp_assert_close(XYZ_to_Lab(XYZ, illuminant), Lab, atol=TOLERANCE_ABSOLUTE_TESTS)

        illuminant = xp.tile(xp_asarray(illuminant, xp=xp), (6, 1))
        xp_assert_close(XYZ_to_Lab(XYZ, illuminant), Lab, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = xp_reshape(xp_asarray(XYZ, xp=xp), (2, 3, 3), xp=xp)
        illuminant = xp_reshape(xp_asarray(illuminant, xp=xp), (2, 3, 2), xp=xp)
        Lab = xp_reshape(xp_asarray(Lab, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(XYZ_to_Lab(XYZ, illuminant), Lab, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_XYZ_to_Lab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_lab.XYZ_to_Lab` definition
        domain and range scale support.
        """

        XYZ = xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp)
        illuminant = xp_asarray([0.31270, 0.32900], xp=xp)
        Lab = np.asarray(XYZ_to_Lab(XYZ, illuminant))

        d_r = (("reference", 1, 1), ("1", 1, 0.01), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_Lab(XYZ * factor_a, illuminant),
                    Lab * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_Lab(self) -> None:
        """Test :func:`colour.models.cie_lab.XYZ_to_Lab` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_Lab(cases, cases[..., 0:2])


class TestLab_to_XYZ:
    """
    Define :func:`colour.models.cie_lab.Lab_to_XYZ` definition unit tests
    methods.
    """

    def test_Lab_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cie_lab.Lab_to_XYZ` definition."""

        xp_assert_close(
            Lab_to_XYZ(xp_asarray([41.52787529, 52.63858304, 26.92317922], xp=xp)),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Lab_to_XYZ(xp_asarray([55.11636304, -41.08791787, 30.91825778], xp=xp)),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Lab_to_XYZ(xp_asarray([29.80565520, 20.01830466, -48.34913874], xp=xp)),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Lab_to_XYZ(
                xp_asarray([41.52787529, 38.48089305, -5.73295122], xp=xp),
                xp_asarray([0.44757, 0.40745], xp=xp),
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Lab_to_XYZ(
                xp_asarray([41.52787529, 51.19354174, 19.91843098], xp=xp),
                xp_asarray([0.34570, 0.35850], xp=xp),
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Lab_to_XYZ(
                xp_asarray([41.52787529, 51.19354174, 19.91843098], xp=xp),
                xp_asarray([0.34570, 0.35850, 1.00000], xp=xp),
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_Lab_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_lab.Lab_to_XYZ` definition n-dimensional
        support.
        """

        Lab = xp_asarray([41.52787529, 52.63858304, 26.92317922], xp=xp)
        illuminant = xp_asarray([0.31270, 0.32900], xp=xp)
        XYZ = np.asarray(Lab_to_XYZ(Lab, illuminant))

        Lab = xp.tile(xp_asarray(Lab, xp=xp), (6, 1))
        XYZ = xp.tile(xp_asarray(XYZ, xp=xp), (6, 1))
        xp_assert_close(Lab_to_XYZ(Lab, illuminant), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        illuminant = xp.tile(xp_asarray(illuminant, xp=xp), (6, 1))
        xp_assert_close(Lab_to_XYZ(Lab, illuminant), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        Lab = xp_reshape(xp_asarray(Lab, xp=xp), (2, 3, 3), xp=xp)
        illuminant = xp_reshape(xp_asarray(illuminant, xp=xp), (2, 3, 2), xp=xp)
        XYZ = xp_reshape(xp_asarray(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(Lab_to_XYZ(Lab, illuminant), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_Lab_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_lab.Lab_to_XYZ` definition
        domain and range scale support.
        """

        Lab = xp_asarray([41.52787529, 52.63858304, 26.92317922], xp=xp)
        illuminant = xp_asarray([0.31270, 0.32900], xp=xp)
        XYZ = np.asarray(Lab_to_XYZ(Lab, illuminant))

        d_r = (("reference", 1, 1), ("1", 0.01, 1), ("100", 1, 100))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    Lab_to_XYZ(Lab * factor_a, illuminant),
                    XYZ * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_Lab_to_XYZ(self) -> None:
        """Test :func:`colour.models.cie_lab.Lab_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Lab_to_XYZ(cases, cases[..., 0:2])
