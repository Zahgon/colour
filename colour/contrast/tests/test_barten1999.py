"""Define the unit tests for the :mod:`colour.contrast.barten1999` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.contrast import (
    contrast_sensitivity_function_Barten1999,
    maximum_angular_size_Barten1999,
    optical_MTF_Barten1999,
    pupil_diameter_Barten1999,
    retinal_illuminance_Barten1999,
    sigma_Barten1999,
)
from colour.utilities import (
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
    "TestOpticalMTFBarten1999",
    "TestPupilDiameterBarten1999",
    "TestSigmaBarten1999",
    "TestRetinalIlluminanceBarten1999",
    "TestMaximumAngularSizeBarten1999",
    "TestContrastSensitivityFunctionBarten1999",
]


class TestOpticalMTFBarten1999:
    """
    Define :func:`colour.contrast.barten1999.optical_MTF_Barten1999`
    definition unit tests methods.
    """

    def test_optical_MTF_Barten1999(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.contrast.barten1999.optical_MTF_Barten1999`
        definition.
        """

        xp_assert_close(
            optical_MTF_Barten1999(xp_asarray([4], xp=xp), xp_asarray([0.01], xp=xp)),
            np.array([0.968910791191297]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            optical_MTF_Barten1999(xp_asarray([8], xp=xp), xp_asarray([0.01], xp=xp)),
            np.array([0.881323136669471]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            optical_MTF_Barten1999(xp_asarray([4], xp=xp), xp_asarray([0.05], xp=xp)),
            np.array([0.454040738727245]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_optical_MTF_Barten1999(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.contrast.barten1999.optical_MTF_Barten1999`
        definition n-dimensional support.
        """

        u = xp_asarray([4, 8, 12], xp=xp)
        sigma = xp_asarray([0.01, 0.05, 0.1], xp=xp)
        M_opt = np.asarray(optical_MTF_Barten1999(u, sigma))

        u = xp.tile(xp_asarray(u, xp=xp), (6, 1))
        sigma = xp.tile(xp_asarray(sigma, xp=xp), (6, 1))
        M_opt = xp.tile(xp_asarray(M_opt, xp=xp), (6, 1))
        xp_assert_close(
            optical_MTF_Barten1999(u, sigma),
            M_opt,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        u = xp_reshape(xp_asarray(u, xp=xp), (2, 3, 3), xp=xp)
        sigma = xp_reshape(xp_asarray(sigma, xp=xp), (2, 3, 3), xp=xp)
        M_opt = xp_reshape(xp_asarray(M_opt, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            optical_MTF_Barten1999(u, sigma),
            M_opt,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_optical_MTF_Barten1999(self) -> None:
        """
        Test :func:`colour.contrast.barten1999.optical_MTF_Barten1999`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        optical_MTF_Barten1999(cases, cases)


class TestPupilDiameterBarten1999:
    """
    Define :func:`colour.contrast.barten1999.pupil_diameter_Barten1999`
    definition unit tests methods.
    """

    def test_pupil_diameter_Barten1999(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.contrast.barten1999.pupil_diameter_Barten1999`
        definition.
        """

        xp_assert_close(
            pupil_diameter_Barten1999(xp_asarray([20], xp=xp), xp_asarray([60], xp=xp)),
            np.array([3.262346170373243]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            pupil_diameter_Barten1999(
                xp_asarray([0.2], xp=xp), xp_asarray([600], xp=xp)
            ),
            np.array([3.262346170373243]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            pupil_diameter_Barten1999(
                xp_asarray([20], xp=xp),
                xp_asarray([60], xp=xp),
                xp_asarray([30], xp=xp),
            ),
            np.array([3.519054451149336]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_pupil_diameter_Barten1999(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.contrast.barten1999.pupil_diameter_Barten1999`
        definition n-dimensional support.
        """

        L = xp_asarray([0.2, 20, 100], xp=xp)
        X_0 = xp_asarray([60, 120, 240], xp=xp)
        Y_0 = xp_asarray([60, 30, 15], xp=xp)
        d = np.asarray(pupil_diameter_Barten1999(L, X_0, Y_0))

        L = xp.tile(xp_asarray(L, xp=xp), (6, 1))
        X_0 = xp.tile(xp_asarray(X_0, xp=xp), (6, 1))
        d = xp.tile(xp_asarray(d, xp=xp), (6, 1))
        xp_assert_close(
            pupil_diameter_Barten1999(L, X_0, Y_0),
            d,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        L = xp_reshape(xp_asarray(L, xp=xp), (2, 3, 3), xp=xp)
        X_0 = xp_reshape(xp_asarray(X_0, xp=xp), (2, 3, 3), xp=xp)
        d = xp_reshape(xp_asarray(d, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            pupil_diameter_Barten1999(L, X_0, Y_0),
            d,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_pupil_diameter_Barten1999(self) -> None:
        """
        Test :func:`colour.contrast.barten1999.pupil_diameter_Barten1999`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        pupil_diameter_Barten1999(cases, cases, cases)


class TestSigmaBarten1999:
    """
    Define :func:`colour.contrast.barten1999.sigma_Barten1999` definition unit
    tests methods.
    """

    def test_sigma_Barten1999(self, xp: ModuleType) -> None:
        """Test :func:`colour.contrast.barten1999.sigma_Barten1999` definition."""

        xp_assert_close(
            sigma_Barten1999(
                xp_asarray([0.5 / 60], xp=xp),
                xp_asarray([0.08 / 60], xp=xp),
                xp_asarray([2.1], xp=xp),
            ),
            np.array([0.008791157173231]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sigma_Barten1999(
                xp_asarray([0.75 / 60], xp=xp),
                xp_asarray([0.08 / 60], xp=xp),
                xp_asarray([2.1], xp=xp),
            ),
            np.array([0.012809761902549]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sigma_Barten1999(
                xp_asarray([0.5 / 60], xp=xp),
                xp_asarray([0.16 / 60], xp=xp),
                xp_asarray([2.1], xp=xp),
            ),
            np.array([0.010040141654601]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sigma_Barten1999(
                xp_asarray([0.5 / 60], xp=xp),
                xp_asarray([0.08 / 60], xp=xp),
                xp_asarray([2.5], xp=xp),
            ),
            np.array([0.008975274678558]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_sigma_Barten1999(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.contrast.barten1999.sigma_Barten1999` definition
        n-dimensional support.
        """

        sigma_0 = xp_asarray([0.25 / 60, 0.5 / 60, 0.75 / 60], xp=xp)
        C_ab = xp_asarray([0.04 / 60, 0.08 / 60, 0.16 / 60], xp=xp)
        d = xp_asarray([2.1, 2.5, 5.0], xp=xp)
        sigma = np.asarray(sigma_Barten1999(sigma_0, C_ab, d))

        sigma_0 = xp.tile(xp_asarray(sigma_0, xp=xp), (6, 1))
        C_ab = xp.tile(xp_asarray(C_ab, xp=xp), (6, 1))
        sigma = xp.tile(xp_asarray(sigma, xp=xp), (6, 1))
        xp_assert_close(
            sigma_Barten1999(sigma_0, C_ab, d),
            sigma,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        sigma_0 = xp_reshape(xp_asarray(sigma_0, xp=xp), (2, 3, 3), xp=xp)
        C_ab = xp_reshape(xp_asarray(C_ab, xp=xp), (2, 3, 3), xp=xp)
        sigma = xp_reshape(xp_asarray(sigma, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            sigma_Barten1999(sigma_0, C_ab, d),
            sigma,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_sigma_Barten1999(self) -> None:
        """
        Test :func:`colour.contrast.barten1999.sigma_Barten1999`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        sigma_Barten1999(cases, cases, cases)


class TestRetinalIlluminanceBarten1999:
    """
    Define :func:`colour.contrast.barten1999.retinal_illuminance_Barten1999`
    definition unit tests methods.
    """

    def test_retinal_illuminance_Barten1999(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.contrast.barten1999.retinal_illuminance_Barten1999`
        definition.
        """

        xp_assert_close(
            retinal_illuminance_Barten1999(
                xp_asarray([20], xp=xp), xp_asarray([2.1], xp=xp), True
            ),
            np.array([66.082316060529919]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            retinal_illuminance_Barten1999(
                xp_asarray([20], xp=xp), xp_asarray([2.5], xp=xp), True
            ),
            np.array([91.815644777503664]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            retinal_illuminance_Barten1999(
                xp_asarray([20], xp=xp), xp_asarray([2.1], xp=xp), False
            ),
            np.array([69.272118011654939]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_retinal_illuminance_Barten1999(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.contrast.barten1999.retinal_illuminance_Barten1999`
        definition n-dimensional support.
        """

        L = xp_asarray([0.2, 20, 100], xp=xp)
        d = xp_asarray([2.1, 2.5, 5.0], xp=xp)
        E = np.asarray(retinal_illuminance_Barten1999(L, d))

        L = xp.tile(xp_asarray(L, xp=xp), (6, 1))
        d = xp.tile(xp_asarray(d, xp=xp), (6, 1))
        E = xp.tile(xp_asarray(E, xp=xp), (6, 1))
        xp_assert_close(
            retinal_illuminance_Barten1999(L, d),
            E,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        L = xp_reshape(xp_asarray(L, xp=xp), (2, 3, 3), xp=xp)
        d = xp_reshape(xp_asarray(d, xp=xp), (2, 3, 3), xp=xp)
        E = xp_reshape(xp_asarray(E, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            retinal_illuminance_Barten1999(L, d),
            E,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_retinal_illuminance_Barten1999(self) -> None:
        """
        Test :func:`colour.contrast.barten1999.retinal_illuminance_Barten1999`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        retinal_illuminance_Barten1999(cases, cases)


class TestMaximumAngularSizeBarten1999:
    """
    Define :func:`colour.contrast.barten1999.maximum_angular_size_Barten1999`
    definition unit tests methods.
    """

    def test_maximum_angular_size_Barten1999(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.contrast.barten1999.\
maximum_angular_size_Barten1999` definition.
        """

        xp_assert_close(
            maximum_angular_size_Barten1999(
                xp_asarray([4], xp=xp),
                xp_asarray([60], xp=xp),
                xp_asarray([12], xp=xp),
                xp_asarray([15], xp=xp),
            ),
            np.array([3.572948005052482]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            maximum_angular_size_Barten1999(
                xp_asarray([8], xp=xp),
                xp_asarray([60], xp=xp),
                xp_asarray([12], xp=xp),
                xp_asarray([15], xp=xp),
            ),
            np.array([1.851640199545103]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            maximum_angular_size_Barten1999(
                xp_asarray([4], xp=xp),
                xp_asarray([120], xp=xp),
                xp_asarray([12], xp=xp),
                xp_asarray([15], xp=xp),
            ),
            np.array([3.577708763999663]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            maximum_angular_size_Barten1999(
                xp_asarray([4], xp=xp),
                xp_asarray([60], xp=xp),
                xp_asarray([24], xp=xp),
                xp_asarray([15], xp=xp),
            ),
            np.array([3.698001308168194]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            maximum_angular_size_Barten1999(
                xp_asarray([4], xp=xp),
                xp_asarray([60], xp=xp),
                xp_asarray([12], xp=xp),
                xp_asarray([30], xp=xp),
            ),
            np.array([6.324555320336758]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_maximum_angular_size_Barten1999(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.contrast.barten1999.\
maximum_angular_size_Barten1999` definition n-dimensional support.
        """

        u = xp_asarray([4, 8, 12], xp=xp)
        X_0 = xp_asarray([60, 120, 240], xp=xp)
        X_max = xp_asarray([12, 14, 16], xp=xp)
        N_max = xp_asarray([15, 20, 25], xp=xp)
        X = np.asarray(maximum_angular_size_Barten1999(u, X_0, X_max, N_max))

        u = xp.tile(xp_asarray(u, xp=xp), (6, 1))
        X_0 = xp.tile(xp_asarray(X_0, xp=xp), (6, 1))
        X = xp.tile(xp_asarray(X, xp=xp), (6, 1))
        xp_assert_close(
            maximum_angular_size_Barten1999(u, X_0, X_max, N_max),
            X,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        u = xp_reshape(xp_asarray(u, xp=xp), (2, 3, 3), xp=xp)
        X_0 = xp_reshape(xp_asarray(X_0, xp=xp), (2, 3, 3), xp=xp)
        X = xp_reshape(xp_asarray(X, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            maximum_angular_size_Barten1999(u, X_0, X_max, N_max),
            X,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_maximum_angular_size_Barten1999(self) -> None:
        """
        Test :func:`colour.contrast.barten1999.\
maximum_angular_size_Barten1999` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        maximum_angular_size_Barten1999(cases, cases, cases, cases)


class TestContrastSensitivityFunctionBarten1999:
    """
    Define :func:`colour.contrast.barten1999.\
contrast_sensitivity_function_Barten1999` definition unit tests methods.
    """

    def test_contrast_sensitivity_function_Barten1999(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.contrast.barten1999.\
contrast_sensitivity_function_Barten1999` definition.
        """

        _a = lambda v: xp_asarray([v], xp=xp)  # noqa: E731

        xp_assert_close(
            contrast_sensitivity_function_Barten1999(
                u=_a(4),
                sigma=_a(0.01),
                E=_a(65),
                X_0=_a(60),
                X_max=_a(12),
                Y_0=_a(60),
                Y_max=_a(12),
                p=_a(1.2e6),
            ),
            np.array([352.761342126727020]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            contrast_sensitivity_function_Barten1999(
                u=_a(8),
                sigma=_a(0.01),
                E=_a(65),
                X_0=_a(60),
                X_max=_a(12),
                Y_0=_a(60),
                Y_max=_a(12),
                p=_a(1.2e6),
            ),
            np.array([177.706338840717340]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            contrast_sensitivity_function_Barten1999(
                u=_a(4),
                sigma=_a(0.02),
                E=_a(65),
                X_0=_a(60),
                X_max=_a(12),
                Y_0=_a(60),
                Y_max=_a(12),
                p=_a(1.2e6),
            ),
            np.array([320.872401634215750]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            contrast_sensitivity_function_Barten1999(
                u=_a(4),
                sigma=_a(0.01),
                E=_a(130),
                X_0=_a(60),
                X_max=_a(12),
                Y_0=_a(60),
                Y_max=_a(12),
                p=_a(1.2e6),
            ),
            np.array([455.171315756946400]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            contrast_sensitivity_function_Barten1999(
                u=_a(4),
                sigma=_a(0.01),
                E=_a(65),
                X_0=_a(120),
                X_max=_a(12),
                Y_0=_a(60),
                Y_max=_a(12),
                p=_a(1.2e6),
            ),
            np.array([352.996281545740660]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            contrast_sensitivity_function_Barten1999(
                u=_a(4),
                sigma=_a(0.01),
                E=_a(65),
                X_0=_a(60),
                X_max=_a(24),
                Y_0=_a(60),
                Y_max=_a(12),
                p=_a(1.2e6),
            ),
            np.array([358.881580104493650]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            contrast_sensitivity_function_Barten1999(
                u=_a(4),
                sigma=_a(0.01),
                E=_a(65),
                X_0=_a(240),
                X_max=_a(12),
                Y_0=_a(60),
                Y_max=_a(12),
                p=_a(1.2e6),
            ),
            np.asarray(
                contrast_sensitivity_function_Barten1999(
                    u=_a(4),
                    sigma=_a(0.01),
                    E=_a(65),
                    X_0=_a(60),
                    X_max=_a(12),
                    Y_0=_a(240),
                    Y_max=_a(12),
                    p=_a(1.2e6),
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            contrast_sensitivity_function_Barten1999(
                u=_a(4),
                sigma=_a(0.01),
                E=_a(65),
                X_0=_a(60),
                X_max=_a(12),
                Y_0=_a(60),
                Y_max=_a(12),
                p=_a(1.4e6),
            ),
            np.array([374.791328640476140]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_contrast_sensitivity_function_Barten1999(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.contrast.barten1999.\
contrast_sensitivity_function_Barten1999` definition n-dimensional support.
        """

        u = xp_asarray([4, 8, 12], xp=xp)
        sigma = xp_asarray([0.01, 0.02, 0.04], xp=xp)
        E = xp_asarray([0.65, 90, 1500], xp=xp)
        X_0 = xp_asarray([60, 120, 240], xp=xp)
        S = np.asarray(
            contrast_sensitivity_function_Barten1999(u=u, sigma=sigma, E=E, X_0=X_0)
        )

        u = xp.tile(xp_asarray(u, xp=xp), (6, 1))
        E = xp.tile(xp_asarray(E, xp=xp), (6, 1))
        S = xp.tile(xp_asarray(S, xp=xp), (6, 1))
        xp_assert_close(
            contrast_sensitivity_function_Barten1999(u=u, sigma=sigma, E=E, X_0=X_0),
            S,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        u = xp_reshape(xp_asarray(u, xp=xp), (2, 3, 3), xp=xp)
        E = xp_reshape(xp_asarray(E, xp=xp), (2, 3, 3), xp=xp)
        S = xp_reshape(xp_asarray(S, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            contrast_sensitivity_function_Barten1999(u=u, sigma=sigma, E=E, X_0=X_0),
            S,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_contrast_sensitivity_function_Barten1999(self) -> None:
        """
        Test :func:`colour.contrast.barten1999.\
contrast_sensitivity_function_Barten1999` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        contrast_sensitivity_function_Barten1999(
            u=cases, sigma=cases, E=cases, X_0=cases
        )
