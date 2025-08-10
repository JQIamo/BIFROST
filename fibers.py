"""
BIFROST provides a set of classes, functions, and data for the simulation of
Si-Ge binary glass optical fibers.

Patrick Banner
Dept of Physics, Swarthmore College
"""

import numpy as np
from scipy import optimize as opt
import numpy.typing as npt
import typing

# Constants
# -------------------------------------------------------------------------------------------------
pi = np.pi
"""Pi constant"""
C_c = 299792458
"""Speed of light in vacuum (m/s)"""

# Material properties for silica and germania
# -------------------------------------------------------------------------------------------------
_SellmeierCoeffs = {'SiO2': np.array(
                            [[[1.10127, -4.94251e-5, 5.27414e-7, -1.59700e-9, 1.75949e-12],
                              [-8.906e-2, 9.0873e-6, -6.53638e-8, 7.77072e-11, 6.84605e-14]],
                             [[1.78752e-5, 4.76391e-5, -4.49019e-7, 1.44546e-9, -1.57223e-12],
                              [2.97562e-1, -8.59578e-4, 6.59069e-6, -1.09482e-8, 7.85145e-13]],
                             [[7.93552e-1, -1.27815e-3, 1.84595e-5, -9.20275e-8, 1.48829e-10],
                              [9.34454, -70.9788e-3, 1.01968e-4, -5.07660e-7, 8.21348e-10]]]
                            ),
                    'GeO2': np.array(
                            [[0.80686642, 0.068972606],
                             [0.71815848, 0.15396605],
                             [0.85416831, 11.841931]]
                            ),
                    'F': np.array(
                         [[[-61.25, 0.2565],
                           [-23.0, 0.101]],
                          [[73.9, -1.836],
                           [10.7, -0.005]],
                          [[233.5, -5.82],
                           [1090.5, -24.695]]]
                         )
                    }
"""
Sellmeier coefficients for silica, germania, and fluorine-doped silica.
These constants are intended for use in formula Bi*w0^2/(w0^2 - Ci^2).
Overall structure for the list is [Bi, Ci], with Bi in in 1/um^2, Ci is in um.
For silica, each list is a list of five coefficients representing the T^n
coefficients, for temperature variation. For germania, they are single coeffs
measured at 24°C, and the calculating method will add on the thermo-optic
coefficients for the change with temperature. For fluorine, they are the linear
and quadratic coefficients in the molar fraction of fluorine, to be added
to the pure silica number.
"""

_CTE = {'SiO2': 5.4e-7, 'GeO2': 10e-6}
"""Coefficient of thermal expansion (1/°C or 1/K) for silica and germania"""

_SofteningTemperature = {'SiO2': 1100, 'GeO2': 300}
"""Softening temperatures (°C) for silica and germania"""

_PoissonRatio = {'SiO2': 0.170, 'GeO2': 0.212}
"""Poisson's ratios for silica and germania"""

_PhotoelasticConstants = {'SiO2': [0.121, 0.270], 'GeO2': [0.130, 0.288]}
"""Photoelastic constants for silica and germania. [p11, p12] """

_YoungModulus = {'SiO2': 74e9, 'GeO2': 45.5e9}
""" Young's modulus for silica and germania (Pa) """

# Validation utility methods
# -------------------------------------------------------------------------------------------------
def _validatePositive(val):
    """  Validate that the argument is a number-like type and is positive.  """
    if not isinstance(val, int | float | np.int32 | np.float64 | np.intc | np.int64):
        raise TypeError("Number expected; this is a" + str(type(val)))
    if not (val > 0):
        raise ValueError("Value should be greater than zero.")
    return val


def _validateNonnegative(val):
    """  Validate that the argument is a number-like type and is non-negative.  """
    if not isinstance(val, int | float | np.int32 | np.float64 | np.intc | np.int64):
        raise TypeError("Number expected.")
    if not (val >= 0):
        raise ValueError("Value should be greater than zero.")
    return val


def _validateFractions(frac):
    """  Validate that the argument is a number-like type and is between 0 and 1 inclusive.  """
    if not isinstance(frac, float | np.float64 | int | np.int32 | np.int64):
        raise TypeError("Number expected.")
    if not ((0 <= frac) and (frac <= 1)):
        raise ValueError("Fraction should be between 0 and 1. Value is {:.3f}.".format(frac))
    return frac

# Methods for calculating properties of silica-germania binary glasses
# -------------------------------------------------------------------------------------------------
def epsilonToEccSq(epsilon: float, signFlag: int = 1) -> float:
    """
    Convert epsilon (the ratio of the semimajor to semiminor axes of an ellipse)
    to eccentricity squared.

    Parameters
    ----------
    epsilon: float
        The ratio of the semimajor to semiminor axes of an ellipse.
    signFlag: {1, -1}
        If 1, the returned value is always positive. If -1, the returned value is
        negative if epsilon < 1, which can be useful for compact calculations.

    Returns
    -------
    float
        The eccentricity squared of the ellipse, which is defined as
        :math:`e^2 = 1 - (b/a)^2`, where :math:`b` is the semiminor axis and
        :math:`a` is the semimajor axis.
    """
    if (epsilon >= 1):
        return 1-(1/epsilon**2)
    else:
        return signFlag*(1-epsilon**2)


def _calcN_Ge(w0: float, T0: float) -> float:
    """ Get the refractive index of germania at a given temperature and wavelength. """
    # We have Sellmeier coefficients and a formula for the thermo-optic coefficient
    # So we'll just add the two together
    wc = w0*1e6
    Tc = T0 + 273.15  # Unit conversions
    n0 = np.sqrt(1 + np.sum(np.array([_SellmeierCoeffs['GeO2'][i][0]*wc**2/(wc**2 - _SellmeierCoeffs['GeO2'][i][1]**2) for i in range(len(_SellmeierCoeffs['GeO2']))])))
    Delta_n0 = 6.2153e-13/4*(Tc**4 - (24+273.15)**4) - 5.3387e-10/3*(Tc**3 - (24+273.15)**3) + 1.6654e-7/2*(Tc**2 - (24+273.15)**2)
    return n0 + Delta_n0


def _calcNs(w0: float, T0: float, m0: float, m1: float) -> typing.Tuple[float, float]:
    """
    This method calculates the refractive indices for the core and
    cladding of a silica-germania binary glass fiber at a given
    wavelength and temperature.

    Parameters
    ----------
    w0: float
        Wavelength in meters (m).
    T0: float
        Temperature in degrees Celsius (°C).
    m0: float
        Molar fraction of dopant in the core. Dopant is germania if m0
        is positive, and fluorine if m0 is negative.
    m1: float
        Molar fraction of dopant in the cladding. Dopant is germania if m1
        is positive, and fluorine if m1 is negative.

    Returns
    -------
    tuple
        A tuple containing the refractive indices of the core (n0) and cladding (n1).
    """
    wc = w0*1e6
    Tc = T0 + 273.15   # Unit conversions
    Tpows = np.array([Tc**i for i in [0, 1, 2, 3, 4]])
    nGe = _calcN_Ge(w0, T0)
    sc0 = np.zeros((3, 2))
    sc1 = np.zeros((3, 2))
    for i in range(3):
        for j in range(2):
            sc0[i][j] = np.dot(_SellmeierCoeffs['SiO2'][i][j], Tpows)
            sc1[i][j] = sc0[i][j] + (m1 < 0)*(_SellmeierCoeffs['F'][i][j][0]*np.abs(m1)**2 + _SellmeierCoeffs['F'][i][j][1]*np.abs(m1))
            sc0[i][j] = sc0[i][j] + (m0 < 0)*(_SellmeierCoeffs['F'][i][j][0]*np.abs(m0)**2 + _SellmeierCoeffs['F'][i][j][1]*np.abs(m0))
    n0 = np.sqrt(1 + np.sum(np.array([sc0[i][0]*wc**2/(wc**2 - sc0[i][1]**2) for i in range(len(sc0))])))
    n1 = np.sqrt(1 + np.sum(np.array([sc1[i][0]*wc**2/(wc**2 - sc1[i][1]**2) for i in range(len(sc1))])))
    if (m0 > 0):
        n0 = (1-m0)*n0 + m0*nGe
    if (m1 > 0):
        n1 = (1-m1)*n1 + m1*nGe
    return n0, n1


def _fromDiffN(mProps: dict, r0: float) -> typing.Tuple[float, float]:
    """
    One way to specify the the refractive index properties of the fiber.

    Calculates the molar fraction germania of a glass given certain
    properties of the fiber. mProps must be a dictionary and keys
    must include one of n0, n1, m0, m1, neff specifying the refractive
    index of the core/cleadding, molar conentration of core/cladding,
    or effective refractive index of the mode; keys must also include ALL
    of dn, T, and w0, the fractional difference in refractive indices
    (n0-n1)/n1 between core and cladding, and the temperature (°C) and
    wavelength (m) at which n0/n1/m0/m1/neff and dn are specified.
    Parameter r0 is the radius of the core, needed for some of the choices
    of specification.

    Parameters
    ----------
    mProps: dict
        Dictionary containing the properties of the fiber. Must include
        one of n0, n1, m0, m1, or neff (the refractive index of the core and
        cladding, the molar fraction germania of the core and cladding, and
        the effective mode index), and must also include dn, T, and w0, the
        fractional difference in refractive indices (n0-n1)/n1 between core
        and cladding, and the temperature (°C) and wavelength (m) at which
        n0/n1/m0/m1/neff and dn are specified.
    r0: float
        Radius of the core (m). This is needed for some of the choices of
        specification, such as when neff is specified.

    Returns
    -------
    tuple
        A tuple containing the molar fractions of germania in the core and cladding.
    
    Raises
    ------
    Exception
        If mProps does not contain one of the required keys (n0, n1, m0, m1, neff).
    """

    w0 = mProps['w0']
    T0 = mProps['T']

    if ('n0' in mProps.keys()):
        # Core is specified
        soln = opt.fsolve(lambda m: _calcNs(w0, T0, m[0], m[1]) - np.array([mProps['n0'], mProps['n0']/(1 + mProps['dn'])]), np.array([0, 0]))
        return soln[0], soln[1]
    elif ('n1' in mProps.keys()):
        # Cladding is specified
        soln = opt.fsolve(lambda m: _calcNs(w0, T0, m[0], m[1]) - np.array([mProps['n1']*(1 + mProps['dn']), mProps['n1']]), np.array([0, 0]))
        return soln[0], soln[1]
    elif ('m0' in mProps.keys()):
        n0 = _calcNs(w0, T0, mProps['m0'], 0)[0]
        soln = opt.fsolve(lambda m: _calcNs(w0, T0, mProps['m0'], m)[1] - n0/(1 + mProps['dn']), 0)[0]
        return mProps['m0'], soln
    elif ('m1' in mProps.keys()):
        n1 = _calcNs(w0, T0, 0, mProps['m1'])[1]
        soln = opt.fsolve(lambda m: _calcNs(w0, T0, m, mProps['m1'])[0] - n1*(1 + mProps['dn']), 0)[0]
        return soln, mProps['m1']
    elif ('neff' in mProps.keys()):
        # Effective mode index is specified...
        # We use n_eff to find n0 and then use that and dn to calculate m0, m1
        def func(na):
            return na**2*((2*pi/w0)**2) - (1/r0**2)*(((1+np.sqrt(2))*r0*(2*pi/w0)*np.sqrt(na**2 - na**2/(1 + mProps['dn'])**2))/(1 + (4 + (r0*(2*pi/w0)*np.sqrt(na**2 - na**2/(1 + mProps['dn'])**2))**4)**(1/4)))**2 - (mProps['neff']*(2*pi/w0))**2
        nco = opt.fsolve(func, 1.45)[0]  # 1.45 is the initial guess
        soln = opt.fsolve(lambda m: _calcNs(w0, T0, m[0], m[1]) - np.array([nco, nco/(1 + mProps['dn'])]), np.array([0, 0]))
        return soln[0], soln[1]
    else:
        raise Exception("mProps missing an index specifier; keys should include one of n0, n1, m0, m1, or neff.")


def _calcV(r0: float, w0: float, n0: float, n1: float) -> float:
    """ Calculate the normalized frequency of the fiber. """
    return r0*(2*pi/w0)*np.sqrt(n0**2 - n1**2)


def _calcBeta(n0: float, w0: float, r0: float, v: float) -> float:
    """ Calculate the propagation constant of the fiber. """
    return np.sqrt((n0**2)*((2*pi/w0)**2) - (1/r0**2)*(((1+np.sqrt(2))*v)/(1+(4+(v**4))**(1/4)))**2)


def _calcLt(L0: float, alpha0: float, T0: float, Tref: float) -> float:
    """
    Get the length adjusted for thermal expansion.

    Parameters
    ----------
    L0: float
        Length of the fiber at reference temperature Tref (m).
    alpha0: float
        Coefficient of thermal expansion of the fiber (1/°C).
    T0: float
        Actual temperature of the fiber (°C).
    Tref: float
        Reference temperature at which L0 is measured (°C).

    Returns
    -------
    float
        Length of the fiber at temperature T0 (m).
    """
    return L0*(1+alpha0*(T0 - Tref))


def _calcCTE(m: float) -> float:
    """ Calculate coefficient of thermal expansion for a doped silica material. """
    if (m <= 0):
        return _CTE['SiO2']
    return (1-m)*_CTE['SiO2'] + m*_CTE['GeO2']


def _calcTS(m: float) -> float:
    """ Calculate softening temperature for a doped silica material. """
    if (m <= 0):
        return _SofteningTemperature['SiO2']
    return (1-m)*_SofteningTemperature['SiO2'] + m*_SofteningTemperature['GeO2']


def _calcPoissonRatio(m: float) -> float:
    """ Calculate Poisson ratio for a doped silica material."""
    if (m <= 0):
        return _PoissonRatio['SiO2']
    return (1-m)*_PoissonRatio['SiO2'] + m*_PoissonRatio['GeO2']


def _calcPhotoelasticConstants(m: float) -> float:
    """ Calculate photoelastic constants for a doped silica material. """
    if (m <= 0):
        return _PhotoelasticConstants['SiO2']
    return [(1-m)*_PhotoelasticConstants['SiO2'][0] + m*_PhotoelasticConstants['GeO2'][0],
            (1-m)*_PhotoelasticConstants['SiO2'][1] + m*_PhotoelasticConstants['GeO2'][1]]


def _calcYoungModulus(m: float) -> float:
    """ Calculate Young's modulus for a doped silica material. """
    if (m <= 0):
        return _YoungModulus['SiO2']
    return (1-m)*_YoungModulus['SiO2'] + m*_YoungModulus['GeO2']


def _calc_B_CNC(epsilon: float, n0: float, n1: float, r0: float, v: float) -> float:
    """ Birefringence due to core noncircularity. """
    return (epsilonToEccSq(epsilon, signFlag=-1)*(1 - n1**2/n0**2)**(3/2))/(r0) * (4/v**3) * (np.log(v))**3 / (1 + np.log(v))


def _calc_B_ATS(w0: float, r0: float, n0: float, beta: float, v: float, p11: float, p12: float,
                alpha0: float, alpha1: float, T0: float, TS: float, nu_p: float,
                epsilon: float) -> float:
    """ Birefringence due to asymmetric thermal stress. """
    return (2*pi/w0)*(1-((r0**2)*((n0**2)*(2*pi/w0)**2 - beta**2))/(v**2))*(0.5*(n0**3)*(p11 - p12)*(alpha1 - alpha0)*np.abs(TS - T0)/(1 - nu_p**2)*((epsilon - 1)/(epsilon + 1)))


def _calc_B_BND(w0: float, n0: float, p11: float, p12: float, nu_p: float, r0: float,
                rc: float, E: float, tf: float = 0) -> float:
    """ Birefringence due to bending. """
    if (rc == 0):
        return 0
    return (2*pi/w0)*((n0**3)/2)*(p11-p12)*(1+nu_p)*(0.5*(r0**2/rc**2) + ((2-3*nu_p)/(1-nu_p))*(r0/rc)*(tf/(pi*(r0**2)*E)))


def _calc_B_TWS(n0: float, p11: float, p12: float, tr: float) -> float:
    """ Birefringence due to twisting. """
    return (1+((n0**2)/2)*(p11-p12))*(tr)


def _calc_deltaB_CNC(epsilon: float, n0: float, n1: float, r0: float, v: float) -> npt.NDArray[np.float64]:
    """
    Calculate the refractive index changes due to core noncircularity.

    Parameters: epsilon, n0, n1, r0, v

    Returns
    -------
    np.array[3]
        An array containing the average, minimum, and maximum refractive
        index changes. The transit time through the fiber is calculated as:

        .. math::
           L_t \\frac{w_0}{2\\pi c} \\frac{1}{\\beta_0 + \\delta\\beta}
    """
    dbx = -((1 - n1**2/n0**2)**(3/2))/(r0) * (2*(np.log(v))**2/v**3) * (1 - (np.log(v)/(1+np.log(v)))*epsilonToEccSq(epsilon, signFlag=-1))
    dby = -((1 - n1**2/n0**2)**(3/2))/(r0) * (2*(np.log(v))**2/v**3) * (1 + (np.log(v)/(1+np.log(v)))*epsilonToEccSq(epsilon, signFlag=-1))
    return np.array([(dbx+dby)/2, dbx, dby])


def _calc_deltaB_ATS(w0: float, r0: float, n0: float, beta: float, v: float, p11: float,
                     p12: float, alpha0: float, alpha1: float, T0: float, TS: float, nu_p: float,
                     epsilon: float) -> npt.NDArray[np.float64]:
    """
    Calculate the refractive index changes due to asymmetric thermal stress.

    Parameters: w0, r0, n0, beta, v, p11, p12, alpha0, alpha1, T0, TS, nu_p, epsilon

    Returns
    -------
    np.array[3]
        An array containing the average, minimum, and maximum refractive
        index changes. The transit time through the fiber is calculated as

        .. math::
           L_t \\frac{w_0}{2\\pi c} \\frac{1}{\\beta_0 + \\delta\\beta}
    """
    dbx = (2*pi/w0)*(1-((r0**2)*((n0**2)*(2*pi/w0)**2 - beta**2))/(v**2))*(0.5*(n0**3)*(alpha1 - alpha0)*np.abs(TS - T0)/(1 - nu_p**2)*(1/(r0*(np.sqrt(epsilon) + 1/np.sqrt(epsilon)))))*(p11*r0*np.sqrt(epsilon) + p12*r0/np.sqrt(epsilon))
    dby = (2*pi/w0)*(1-((r0**2)*((n0**2)*(2*pi/w0)**2 - beta**2))/(v**2))*(0.5*(n0**3)*(alpha1 - alpha0)*np.abs(TS - T0)/(1 - nu_p**2)*(1/(r0*(np.sqrt(epsilon) + 1/np.sqrt(epsilon)))))*(p12*r0*np.sqrt(epsilon) + p11*r0/np.sqrt(epsilon))
    return np.array([(dbx+dby)/2, dbx, dby])


def _calc_deltaB_BND(w0: float, n0: float, p11: float, p12: float, nu_p: float, r0: float,
                     rc: float, E: float, tf: float = 0) -> npt.NDArray[np.float64]:
    """
    Calculate the refractive index changes due to bending.

    Parameters: w0, n0, p11, p12, nu_p, r0, rc, E, tf=0

    Returns
    -------
    np.array[3]
        An array containing the average, minimum, and maximum refractive
        index changes. The transit time through the fiber is calculated as:

        .. math::
           L_t \\frac{w_0}{2\\pi c} \\frac{1}{\\beta_0 + \\delta\\beta}
    """
    if (rc == 0):
        return np.array([0, 0, 0])
    dbx = (2*pi/w0)*(n0**3/4)*p11*(1+nu_p)*(r0**2/rc**2)
    dby = (2*pi/w0)*(n0**3/4)*p12*(1+nu_p)*(r0**2/rc**2)
    return np.array([(dbx+dby)/2, dbx, dby])


def _calc_J0(beta: float, B_CNC: float, B_ATS: float, B_BND: float, B_TWS: float, Lt: float) -> npt.NDArray[np.complex_]:
    """
    Calculates a Jones matrix given birefringences.

    .. note::
       For now, if a twist birefringence is given, the other birefringnces
       are ignored and the returned matrix ONLY contains the twist birefringence.
    
    Parameters
    ----------
    beta: float
        Propagation constant (1/m).
    B_CNC: float
        Birefringence due to core noncircularity (rad/m).
    B_ATS: float
        Birefringence due to asymmetric thermal stress (rad/m).
    B_BND: float
        Birefringence due to bending (rad/m).
    B_TWS: float
        Birefringence due to twisting (rad/m).
    Lt: float
        Thermally adjusted length (m).

    Returns
    -------
    np.ndarray
        A 2x2 NumPy array representing the Jones matrix.
    """
    Jbase = np.array([
                    [np.exp(1.0j*((0 + (B_CNC + B_ATS + B_BND)/2)*Lt)), 0],
                    [0, np.exp(1.0j*((0 - (B_CNC + B_ATS + B_BND)/2)*Lt))]
                    ])
    if (B_TWS != 0):
        Jbase = np.matmul(np.array([[np.cos(B_TWS*Lt/2), -np.sin(B_TWS*Lt/2)], [np.sin(B_TWS*Lt/2), np.cos(B_TWS*Lt/2)]]), Jbase)
    return Jbase

# Other miscellaneous methods
# -------------------------------------------------------------------------------------------------
def makeRotators(n0: int) -> npt.NDArray:
    """
    Makes n0 arbitrary polarization rotators.

    Returns
    -------
    np.ndarray
        An array of n0 Rotator instances with random orientations.
    """
    rotators = np.array([], dtype=object)
    alphaData = np.random.normal(loc=0.0, scale=1.0, size=(n0, 4))
    for i in range(n0):
        rotators = np.append(rotators, Rotator(alphaData[i]))
    return rotators


_randomDistDefaults = {'T0': {'mean': 25, 'scale': 2, 'dist': 'normal'},
                       'Tref': 20,
                       'm0': 0.0036,
                       'm1': 0,
                       'mProps': {},
                       'epsilon': {'mean': 1, 'scale': 0.007, 'dist': 'uniform'},
                       'r0': 4.1e-6, 'r1': 125e-6/2,
                       'rc': {'mean': 10, 'scale': 10, 'dist': 'uniform'},
                       'tr': 0,
                       'tf': 0,
                       'nPaddles': {'mean': 3, 'scale': 1, 'dist': 'uniform_int'},
                       'Ns': {'mean': 3, 'scale': 2, 'dist': 'uniform_int'},
                       'gapLs': {'mean': 0.02, 'scale': 0.005, 'dist': 'uniform'},
                       'angles': {'mean': 0, 'scale': 100*pi/180, 'dist': 'uniform'},
                       'tfs': 0,
                       'rps': {'mean': 0.05, 'scale': 0.02, 'dist': 'uniform'},
                       'L0': {'scale': 10, 'dist': 'normal'},
                       'alpha': {'mean': 0.0, 'scale': 1.0, 'dist': 'normal'}
                       }
"""
    Default parameters for generating random fiber configurations.
    Each entry can be either a single number, in which case all instances
    of the property will be set to that number, or a dictionary containing
    keys 'mean', 'scale', and 'dist' to match _getRandom() below.
    Note: 'L0' can not have 'mean' as the mean is usually determined
    by a user input.
"""


def _getRandom(n0: int | tuple, mean: float, scale: float, dist: str) -> npt.NDArray[np.float64]:
    """
    This is a utility method for assisting with the creation of random
    fiber configurations.

    Parameters
    ----------
    n0: int or tuple
        The size of the needed random numbers. Can be a single number or 2-tuple.
    mean: float
        The mean of the distribution.
    scale: float
        A scaling parameter. For the uniform distributions, specify the
        half-width; for Gaussian distributions, specify the standard deviation.
    dist: str
        A string determining the distribution; pick from:
        - uniform: A uniform distribution (mean-scale to mean+scale)
        - uniform_int: A uniform distribution of integers only
        - normal or Gaussian: A Gaussian distribution
        - normal_pos or Gaussian_pos: A Gaussian distribution cut off at zero,
        so as to be only the positive part.

    Returns
    -------
    np.ndarray
        An array of random numbers of size n0, generated according to the
        specified distribution.
    """
    if (dist == 'uniform'):
        return (np.random.random(size=n0) - 0.5)*(scale*2) + mean
    elif (dist == 'normal' or dist == 'Gaussian'):
        return np.random.normal(loc=mean, scale=scale, size=n0)
    elif (dist == 'normal_pos' or dist == 'Gaussian_pos'):
        arr = np.random.normal(loc=mean, scale=scale, size=n0)
        q = np.where([arr <= 0])[1]
        while (len(q) != 0):
            arr[q] = np.random.normal(loc=mean, scale=scale, size=len(q))
            q = np.where([arr <= 0])[1]
        return arr
    elif (dist == 'uniform_int'):
        return np.random.randint(int(mean - scale), high=int(mean + scale), size=n0)

# FiberLength class definition
# -------------------------------------------------------------------------------------------------
class FiberLength():
    """
    This class allows the simulation of Si-Ge binary glasses. It
    assumes a pure silica cladding and a core made of silica doped
    with germania.

    Attributes
    ----------
    w0: float
        Wavelength in meters (m).
    T0: float
        Temperature in degrees Celsius (°C).
    L0: float
        Length of the fiber at reference temperature Tref (m).
    r0: float
        Radius of the core (m).
    r1: float
        Radius of the cladding (m).
    epsilon: float
        Ratio of the semimajor to semiminor axes of an ellipse representing
        the core noncircularity.
    m0: float
        Molar fraction of germania in the core. Positive values indicate
        germania doping, while negative values indicate fluorine doping.
    m1: float
        Molar fraction of germania in the cladding. Positive values indicate
        germania doping, while negative values indicate fluorine doping.
    Tref: float
        Reference temperature at which L0 is measured (°C).
    rc: float
        Curvature radius of the fiber (m). This is used for calculating
        bending birefringence.
    tf: float
        Tension on the fiber (N).
    tr: float
        Fiber twist rate (radians/m). This is used for calculating twisting birefringence.
    n0: float
        Core index of refraction.
    n1: float
        Cladding index of refraction.
    v: float
        Normalized frequency.
    beta: float
        Propagation constant (1/m).
    alpha0: float
        Coefficient of thermal expansion of the core (1/°C).
    alpha1: float
        Coefficient of thermal expansion of the cladding (1/°C).
    Lt: float
        Thermally adjusted length (m).
    nu_p: float
        Poisson's ratio.
    p11, p12: float
        Photoelastic constants.
    TS: float
        Softening temperature of the core (°C).
    E: float
        Young's modulus for the core (Pa).
    B_CNC: float
        Birefringence due to core noncircularity (rad/m).
    B_ATS: float
        Birefringence due to asymmetric thermal stress (rad/m).
    B_BND: float
        Birefringence due to bending (rad/m).
    B_TWS: float
        Birefringence due to twisting (rad/m).
    J0: np.ndarray
        Total Jones matrix.
    """

    # Derived quantities
    @property
    def n0(self) -> float:
        """Core index of refraction"""
        return _calcNs(self.w0, self.T0, self.m0, self.m1)[0]

    @property
    def n1(self) -> float:
        """Cladding index of refraction"""
        return _calcNs(self.w0, self.T0, self.m0, self.m1)[1]

    @property
    def v(self) -> float:
        """Normalized frequency"""
        n0, n1 = _calcNs(self.w0, self.T0, self.m0, self.m1)
        return _calcV(self.r0, self.w0, n0, n1)

    @property
    def beta(self) -> float:
        """Propagation constant (1/m)"""
        n0, n1 = _calcNs(self.w0, self.T0, self.m0, self.m1)
        v = _calcV(self.r0, self.w0, n0, n1)
        return _calcBeta(n0, self.w0, self.r0, v)

    @property
    def alpha0(self) -> float:
        """Coefficient of thermal expansion of the core (1/°C)"""
        return _calcCTE(self.m0)

    @property
    def alpha1(self) -> float:
        """Coefficient of thermal expansion of the cladding (1/°C)"""
        return _calcCTE(self.m1)

    @property
    def Lt(self) -> float:
        """Thermally adjusted length (m)"""
        alpha0 = _calcCTE(self.m0)
        return _calcLt(self.L0, alpha0, self.T0, self.Tref)

    @property
    def nu_p(self) -> float:
        """Poisson's ratio"""
        return _calcPoissonRatio(self.m0)

    @property
    def p11(self) -> float:
        """p11, p12: Photoelastic constants of the core"""
        return _calcPhotoelasticConstants(self.m0)[0]

    @property
    def p12(self) -> float:
        """p11, p12: Photoelastic constants of the core"""
        return _calcPhotoelasticConstants(self.m0)[1]

    @property
    def TS(self) -> float:
        """Softening temperature of the core (°C)"""
        return _calcTS(self.m0)

    @property
    def E(self) -> float:
        """Young's modulus for the core (Pa)"""
        return _calcYoungModulus(self.m0)

    @property
    def B_CNC(self) -> float:
        """Birefringence due to core noncircularity (rad/m)"""
        n0, n1 = _calcNs(self.w0, self.T0, self.m0, self.m1)
        v = _calcV(self.r0, self.w0, n0, n1)
        return _calc_B_CNC(self.epsilon, n0, n1, self.r0, v)

    @property
    def B_ATS(self) -> float:
        """Birefringence due to asymmetric thermal stress (rad/m)"""
        n0, n1 = _calcNs(self.w0, self.T0, self.m0, self.m1)
        v = _calcV(self.r0, self.w0, n0, n1)
        beta = _calcBeta(n0, self.w0, self.r0, v)
        p11, p12 = _calcPhotoelasticConstants(self.m0)
        alpha0 = _calcCTE(self.m0)
        alpha1 = _calcCTE(self.m1)
        TS = _calcTS(self.m0)
        nu_p = _calcPoissonRatio(self.m0)
        return _calc_B_ATS(self.w0, self.r0, n0, beta, v, p11, p12, alpha0, alpha1, self.T0, TS, nu_p, self.epsilon)

    @property
    def B_BND(self) -> float:
        """Birefringence due to bending (rad/m)"""
        n0, n1 = _calcNs(self.w0, self.T0, self.m0, self.m1)
        p11, p12 = _calcPhotoelasticConstants(self.m0)
        nu_p = _calcPoissonRatio(self.m0)
        E = _calcYoungModulus(self.m0)
        return _calc_B_BND(self.w0, n0, p11, p12, nu_p, self.r1, self.rc, E, tf=self.tf)

    @property
    def B_TWS(self) -> float:
        """Birefringence due to twisting (rad/m)"""
        n0, n1 = _calcNs(self.w0, self.T0, self.m0, self.m1)
        p11, p12 = _calcPhotoelasticConstants(self.m0)
        return _calc_B_TWS(n0, p11, p12, self.tr)

    @property
    def J0(self) -> npt.NDArray[np.complex_]:
        """Total Jones matrix"""
        n0, n1 = _calcNs(self.w0, self.T0, self.m0, self.m1)
        v = _calcV(self.r0, self.w0, n0, n1)
        beta = _calcBeta(n0, self.w0, self.r0, v)
        p11, p12 = _calcPhotoelasticConstants(self.m0)
        alpha0 = _calcCTE(self.m0)
        alpha1 = _calcCTE(self.m1)
        Lt = _calcLt(self.L0, alpha0, self.T0, self.Tref)
        TS = _calcTS(self.m0)
        E = _calcYoungModulus(self.m0)
        nu_p = _calcPoissonRatio(self.m0)
        B_CNC = _calc_B_CNC(self.epsilon, n0, n1, self.r0, v)
        B_ATS = _calc_B_ATS(self.w0, self.r0, n0, beta, v, p11, p12, alpha0, alpha1, self.T0, TS, nu_p, self.epsilon)
        B_BND = _calc_B_BND(self.w0, n0, p11, p12, nu_p, self.r1, self.rc, E, tf=self.tf)
        B_TWS = _calc_B_TWS(n0, p11, p12, self.tr)
        return _calc_J0(beta, B_CNC, B_ATS, B_BND, B_TWS, Lt)

    def __init__(self, w0: float, T0: float, L0: float, r0: float, r1: float, epsilon: float, m0: float,
                 m1: float, Tref: float, rc: float, tf: float, tr: float,
                 mProps: dict = {}) -> None:
        """
        Initializes the FiberLength object with the given parameters.

        Parameters
        ----------
        w0: float
            Wavelength (m).
        T0: float
            Temperature (°C).
        L0: float
            Length measured at Tref (m).
        r0: float
            Radius of core (m).
        r1: float
            Outer radius of cladding (m).
        epsilon: float
            Core noncircularity, defined as a/b, where a, b are the semimajor
            and semiminor axes. Sometimes an eccentricity is defined as
            :math:`\\epsilon^2 = (r_y/r_x)^2` for :math:`r_y < r_x`. The
            parameter :math:`\\epsilon` is related as :math:`e^2 = 1 - 1/\\epsilon^2`.
            Then :math:`r_x = r0/(1-e^2)^(1/4)` and :math:`r_y = r0*(1-e^2)^(1/4)`.
        m0: float
            Core molar fraction of fluorine (if negative) or germania (if positive)
            (may be overriden by mProps).
        m1: float
            Cladding molar fraction of fluorine (if negative) or germania (if positive)
            (may be overriden by mProps).
        Tref: float
            Temperature for length reference (°C).
        rc: float
            Bending radius of curvature (m). If set to zero, treated as infinity.
        tf: float
            Axial tension force on bends (N).
        tr: float
            Twist rate (rad/m).
        mProps: :obj:`dict`, optional
            A dictionary with alternate specifications of the
            doping concentrations of the fiber; if {}, then m0,m1 will be used;
            if not {}, this will override m0, m1, and keys must include one of
            n0, n1, m0, m1, neff specifying the refractive index of the core
            or cladding, the molar concentration of the core or cladding,
            or effective refractive index of the mode; keys must also include
            ALL of dn, T, and w0, the fractional difference in refractive
            indices (n0-n1)/n1 between core and cladding, and the temperature
            (°C) and wavelength (m) at which n0, n1, m0, m1, neff and dn are
            specified. (optional, default {})

        Raises
        ------
        Exception
            If m0 is not larger than m1, or if mProps does not contain the
            required keys for initialization.
        """

        self.w0 = w0
        self.T0 = T0
        self.L0 = L0
        self.r0 = r0
        self.r1 = r1
        self.epsilon = epsilon
        if (mProps == {}):
            if (m0 <= m1):
                raise Exception("m0 needs to be larger than m1 for a physical fiber. You have m0 = {:.6f} and m1 = {:.6f}.".format(m0, m1))
            else:
                self.m0 = m0
                self.m1 = m1
        else:
            if not (np.all([k in mProps.keys() for k in ['dn', 'w0', 'T']]) and np.any([k in mProps.keys() for k in ['n0', 'n1', 'm0', 'm1', 'neff']])):
                raise Exception("Unable to initialize FiberLength because mProps doesn't contain the right keys. See FiberLength.__init__ documentation for more details.")
            else:
                a, b = _fromDiffN(mProps, r0)
                if (a <= b):
                    raise Exception("m0 needs to be larger than m1 for a physical fiber. Review your mProps dictionary.")
                else:
                    self.m0 = a
                    self.m1 = b
        self.Tref = Tref
        self.rc = rc
        self.tf = tf
        self.tr = tr

    def calcDGD(self, dw0: float = 0.1e-9) -> float:
        """
        Calculates the DGD of the fiber length using a small wavelength change.

        Parameters
        ----------
        dw0 : :obj:`float`, optional
            The small wavelength step to take (m). Default is 0.1 nm.

        Returns
        -------
        dgd : float
            DGD in s.
        """

        # Store current variables
        wb = self.w0
        Jb = self.J0

        # Get Jones matrices for ± dw0
        self.w0 = self.w0 - dw0
        Ja = self.J0
        self.w0 = self.w0 + 2*dw0
        Jc = self.J0

        # Reset values for this object
        self.w0 = wb

        # Get DGD estimates for ± dw0
        matM = np.matmul(Ja, np.linalg.inv(Jb))
        vals, vecs = np.linalg.eig(matM)
        dgdM = np.abs(np.angle(vals[0]/vals[1])/((2*pi*C_c)/self.w0**2*dw0))

        matP = np.matmul(Jc, np.linalg.inv(Jb))
        vals, vecs = np.linalg.eig(matP)
        dgdP = np.abs(np.angle(vals[0]/vals[1])/((2*pi*C_c)/self.w0**2*dw0))

        # Average and return
        dgd = (dgdM + dgdP)/2
        return dgd

    def calcBeatLength(self) -> float:
        """
        Polarization beat length of the fiber.

        Ignores twisting; only accounts for core nincircularity, asymmetric
        thermal stress, and bending birefringences.

        Returns
        -------
        float
            polarization beat length (m)
        """
        return np.abs(2*pi/(self.B_CNC + self.B_ATS + self.B_BND + self.B_TWS))

    def calcD_CD(self, dw0: float = 0.1e-9) -> float:
        """
        Chromatic dispersion of the fiber length from the propagation constant
        of the fiber mode.

        Parameters
        ----------
        dw0 : :obj:`float`, optional
            The small wavelength step to take (m).

        Returns
        -------
        float
            Group velocity dispersion parameter :math:`D_\\text{CD}` in ps/(nm*km).
        """
        # Store current variables
        wb = self.w0
        nB = self.beta/(2*pi/self.w0)

        # Get relevant parameters for ± dw0
        self.w0 = self.w0 - dw0
        nA = self.beta/(2*pi/self.w0)
        self.w0 = self.w0 + 2*dw0
        nC = self.beta/(2*pi/self.w0)

        # Reset values for this object
        self.w0 = wb

        dcd = -self.w0/C_c*(nC - 2*nB + nA)/dw0**2*1e12*1e-9*1e3
        return dcd

    def calcNGEff(self, dw0: float = 0.1e-9) -> float:
        """
        Calculate the effective group index of the fiber.

        .. math:: c/v_g = n(\\lambda) - \\lambda \\, dn/d\\lambda.

        Parameters
        ----------
        :obj:`float`, optional
            The small wavelength step to take (m).

        Returns
        -------
        float
            Effective group index.
        """
        # Store current variables
        wb = self.w0
        nb = self.beta/(2*pi/self.w0)
        # Get relevant parameters for ± dw0
        self.w0 = wb - dw0
        na = self.beta/(2*pi/self.w0)
        self.w0 = wb + dw0
        nc = self.beta/(2*pi/self.w0)
        # Reset values for this object
        self.w0 = wb
        # Calculate
        dndwP = (nc-nb)/dw0
        dndwM = (nb-na)/dw0
        dndw = (dndwP + dndwM)/2
        return nb - wb*(dndw)

    def calcPhaseDelay(self) -> npt.NDArray[np.float64]:
        """
        Calculates the time for light to propagate through the fiber.

        Returns
        -------
        np.array[3]
            average, min, max transit time through the fiber (s)
        """
        n0, n1 = _calcNs(self.w0, self.T0, self.m0, self.m1)
        v = _calcV(self.r0, self.w0, n0, n1)
        beta = _calcBeta(n0, self.w0, self.r0, v)
        p11, p12 = _calcPhotoelasticConstants(self.m0)
        alpha0 = _calcCTE(self.m0)
        alpha1 = _calcCTE(self.m1)
        TS = _calcTS(self.m0)
        nu_p = _calcPoissonRatio(self.m0)
        E = _calcYoungModulus(self.m0)
        bcnc = _calc_deltaB_CNC(self.epsilon, n0, n1, self.r0, v)
        bats = _calc_deltaB_ATS(self.w0, self.r0, n0, beta, v, p11, p12, alpha0, alpha1, self.T0, TS, nu_p, self.epsilon)
        bbnd = _calc_deltaB_BND(self.w0, n0, p11, p12, nu_p, self.r1, self.rc, E, tf=self.tf)
        return self.Lt/((2*pi*C_c/self.w0)/(np.ones(3)*self.beta + bcnc + bats + bbnd))

    def __str__(self):
        info_array = ["Fiber length, properties:",
                      "Length: {:.4f} m".format(self.L0),
                      "Operating wavelength: {:.4f} nm".format(self.w0*1e9),
                      "Operating temperature: {:.4f} °C".format(self.T0),
                      "Reference temperature: {:.4f} °C".format(self.Tref),
                      "Base radius of the core: {:.4f} um".format(self.r0*1e6),
                      "Radius of the cladding: {:.4f} um".format(self.r1*1e6),
                      "Noncircularity a/b: {:.4f}".format(self.epsilon),
                      "Molar fraction of dopant in the core: {:.4f}".format(self.m0),
                      "Molar fraction of dopant in the cladding: {:.4f}".format(self.m1),
                      "Core index of refraction: {:.5f}".format(self.n0),
                      "Cladding index of refraction: {:.5f}".format(self.n1),
                      "Mode effective index of refraction: {:.5f}".format(self.beta/(2*pi/self.w0)),
                      "Effective group index: {:.5f}".format(self.calcNGEff()),
                      "Normalized frequency V: {:.4f}".format(self.v),
                      ]
        if (self.rc == 0):
            info_array = info_array + ["Bend radius of curvature: N/A", "Axial tension on bend: N/A"]
        else:
            info_array = info_array + ["Bend radius of curvature: {:.4f} m".format(self.rc), "Axial tension on bend: {:.4f} N".format(self.tf)]
        if (self.tr == 0):
            info_array = info_array + ["Twist rate: N/A"]
        else:
            info_array = info_array + ["Twist rate: {:.4f} rad/m".format(self.tr)]
        info_array = info_array + ["Differential group delay: {:.4f} ps".format(self.calcDGD()*1e12),
                                   "Polarization beat length: {:.4f} m".format(self.calcBeatLength()),
                                   "Chromatic dispersion coefficient D_CD: {:.4f} ps/(nm^2 km)".format(self.calcD_CD())]
        sep = '\n'
        return sep.join(info_array)

# FiberPaddleSet class definition
# -------------------------------------------------------------------------------------------------
class FiberPaddleSet():
    """
    This class implements a set of fiber paddles via an array of
    FiberLength() objects.

    From the code standpoint, we alternate FiberLengths with nonzero
    twist rates and FiberLengths with nonzero bend radii representing
    the paddles. However, physically we suppose it's all one fiber, so
    all the FiberLengths will share some common properties.

    Attributes
    ----------
    w0: float
        Wavelength (m)
    T0: float
        Temperature (°C)
    Tref: float
        Reference temperature for the lengths (°C)
    r0: float
        Radius of the core (m)
    r1: float
        Outer radius of the cladding (m)
    epsilon: float
        Core noncircularity (a/b, where a, b are the semimajor and semiminor axes)
    m0, m1: float
        Doping concentration in core and cladding (molar fraction of germania)
    nPaddles: int
        Number of paddles in the set
    finalTwistBool: bool
        A Boolean to indicate whether there is a final section of twisted fiber
        after the last paddle or not.
    fibers: list[FiberLength]
        An array of FiberLength objects representing the fiber forming the paddle set.
    J0: np.ndarray
        The total Jones matrix of the entire paddle set.
    L0: float
        The total (non-thermally-adjusted) length of the fiber forming the paddle set (m).
    """

    @property
    def fibers(self) -> typing.List[FiberLength]:
        """The array of FiberLength objects representing the fiber forming the paddle set."""
        fa = []
        angs = np.concatenate(([0], self.angles))
        for i in range(self.nPaddles):
            # Twist
            fa = np.append(fa, FiberLength(self.w0, self.T0, self.gapLs[i], self.r0, self.r1, self.epsilon, self.m0, self.m1, self.Tref, 0, 0, (angs[i+1] - angs[i])/self.gapLs[i]))
            # Bend
            fa = np.append(fa, FiberLength(self.w0, self.T0, 2*pi*self.rps[i]*self.Ns[i], self.r0, self.r1, self.epsilon, self.m0, self.m1, self.Tref, self.rps[i], self.tfs[i], 0))
        # Final twist
        if (self.ftb):
            fa = np.append(fa, FiberLength(self.w0, self.T0, self.gapLs[-1], self.r0, self.r1, self.epsilon, self.m0, self.m1, self.Tref, 0, 0, (0-angs[-1])/self.gapLs[-1]))
        return fa

    @property
    def J0(self) -> npt.NDArray[np.complex_]:
        """The total Jones matrix of the entire paddle set."""
        fa = self.fibers
        Jtot = np.array([[1, 0], [0, 1]])
        for i in range(len(fa)):
            Jtot = np.matmul(fa[i].J0, Jtot)
        return Jtot

    @property
    def L0(self) -> float:
        """The total (non-thermally-adjusted) length of the fiber forming the paddle set (m)."""
        return np.sum(self.gapLs) + 2*pi*np.dot(self.rps, self.Ns)

    # We're going to let the entire set have the same fiber properties
    def __init__(self, w0: float, T0: float, r0: float, r1: float, epsilon: float, m0: float,
                 m1: float, Tref: float, nPaddles: int, rps: npt.NDArray[np.float64],
                 angles: npt.NDArray[np.float64], tfs: npt.NDArray[np.float64], Ns: npt.NDArray[np.int_],
                 gapLs: npt.NDArray[np.float64], mProps: dict = {}, finalTwistBool:bool = False) -> None:
        """
        Initialize a FiberPaddleSet.

        Parameters
        ----------
        w0: float
            Wavelength (m)
        T0: float
            Temperature (°C)
        r0: float
            Radius of the core (m)
        r1: float
            Outer radius of the cladding (m)
        epsilon: float
            Core noncircularity (a/b, where a, b are the semimajor and semiminor axes)
        m0: float
            Doping concentration in core (molar fraction of germania)
        m1: float
            Doping concentration in cladding (molar fraction of germania)
        Tref: float
            Reference temperature for the lengths (°C)
        nPaddles: int
            Number of paddles in the set
        rps: np.array[float]
            Radii of curvature for each paddle (m)
        angles: np.array[float]
            The angle of each paddle (rad)
        tfs: np.array[float]
            Tension forces on each paddle (N)
        Ns: np.array[int]
            The number of turns of fiber on each paddle
        gapLs: np.array[float]
            The lengths of each of the straight sections of fiber between the paddles,
            including one before the first paddle and, if finalTwistBool is True,
            one after the last paddle (lengths in m)
        mProps: :obj:`dict`, optional
            A dictionary with alternate specifications of the doping concentrations
            of the fiber; if {}, then m0,m1 will be used; if not {}, this will override
            m0, m1, and keys must include one of n0, n1, m0, m1, neff specifying the
            refractive index of the core or cladding, the molar concentration of the
            core or cladding, or effective refractive index of the mode; keys must also
            include ALL of dn, T, and w0, the fractional difference in refractive
            indices (n0-n1)/n1 between core and cladding, and the temperature (°C)
            and wavelength (m) at which n0, n1, m0, m1, neff and dn are
            specified. Default is {}.
        finalTwistBool: :obj:`bool`, optional
            A Boolean to indicate whether there is a final section of twisted fiber
            after the last paddle or not (default False).

        Raises
        ------
        Exception
            If nPaddles is not a positive integer, or if the lengths of rps, angles,
            Ns, and gapLs do not match nPaddles, or if m0 is not larger than m1,
            or if mProps does not contain the required keys for initialization.
        """

        # Do some error checking
        _validatePositive(nPaddles)
        if (len(rps) != nPaddles):
            raise Exception("Length of rp array doesn't match nPaddles!")
        [_validatePositive(i) for i in rps]
        if (len(angles) != nPaddles):
            raise Exception("Length of angles array doesn't match nPaddles!")
        if (len(Ns) != nPaddles):
            raise Exception("Length of Ns array doesn't match nPaddles!")
        [_validatePositive(int(i)) for i in Ns]
        if (len(gapLs) != (nPaddles+int(finalTwistBool))):
            raise Exception("Length of gapLs array doesn't match nPaddles + {:.0f}!".format(int(finalTwistBool)))
        [_validatePositive(i) for i in gapLs]

        self.w0 = w0
        self.T0 = T0
        self.r0 = r0
        self.r1 = r1
        self.epsilon = epsilon
        if (mProps == {}):
            if (m0 <= m1):
                raise Exception("m0 needs to be larger than m1 for a physical fiber. You have m0 = {:.6f} and m1 = {:.6f}.".format(m0, m1))
            else:
                self.m0 = m0
                self.m1 = m1
        else:
            if not (np.all([k in mProps.keys() for k in ['dn', 'w0', 'T']]) and np.any([k in mProps.keys() for k in ['n0', 'n1', 'm0', 'm1', 'neff']])):
                raise Exception("Unable to initialize FiberLength because mProps doesn't contain the right keys. See FiberLength.__init__ documentation for more details.")
            else:
                a, b = _fromDiffN(mProps, r0)
                if (a <= b):
                    raise Exception("m0 needs to be larger than m1 for a physical fiber. Review your mProps dictionary.")
                else:
                    self.m0 = a
                    self.m1 = b
        self.Tref = Tref
        self.nPaddles = nPaddles
        self.rps = rps
        self.angles = angles
        self.tfs = tfs
        self.Ns = Ns
        self.gapLs = gapLs
        self.finalTwistBool = finalTwistBool

    def __str__(self):
        info_array = ["Fiber paddle set, properties:",
                      "Operating wavelength: {:.4f} nm".format(self.w0*1e9),
                      "Operating temperature: {:.4f} °C".format(self.T0),
                      "Reference temperature: {:.4f} °C".format(self.Tref),
                      "Base radius of the core: {:.4f} um".format(self.r0*1e6),
                      "Radius of the cladding: {:.4f} um".format(self.r1*1e6),
                      "Noncircularity a/b: {:.4f}".format(self.epsilon),
                      "Molar fraction of dopant in the core: {:.4f}".format(self.m0),
                      "Molar fraction of dopant in the cladding: {:.4f}".format(self.m1),
                      " ",
                      "Number of paddles: {:.0f}".format(self.nPaddles),
                      "Radii of curvature of paddles (m): " + str(self.rps),
                      "Number of fiber turns on each paddle: " + str(self.Ns),
                      "Angles of paddles (m): " + str(self.angles),
                      "Tension forces on wrappings of each paddle (N): " + str(self.tfs),
                      "Lengths of fiber between each paddle (m): " + str(self.gapLs),
                      "Total fiber length: {:.4f} m".format(self.L0),
                      " ",
                      "Use object.fibers to get more details about each fiber component."
                      ]
        sep = '\n'
        return sep.join(info_array)

    def calcPhaseDelay(self) -> npt.NDArray[np.float64]:
        """
        Calculates the time for light to propagate through the fiber.

        Returns
        -------
        np.array[3]
            average, min, max transit time through the fiber (s)
        """
        fa = self.fibers
        t0 = np.array([0, 0, 0])
        for i in range(len(fa)):
            t0 = t0 + fa[i].calcPhaseDelay()
        return t0

# Rotator class definition
# -------------------------------------------------------------------------------------------------
class Rotator():
    """
    Implements an arbitrary rotator following the formalism of Czegledi et al.

    Attributes
    ----------
    alpha: np.array[4]
        A 4-vector defining the rotation, where alpha[0] is cos(theta)
        and alpha[1:] is the axis of rotation (a 3-vector).
    theta: float
        The angle of rotation (rad).
    L0: float
        The length of the rotator (m). Set to zero for compatibility purposes.
    J0: np.ndarray
        The 2x2 Jones matrix.
    """

    @property
    def theta(self) -> float:
        """The angle of rotation (rad)."""
        return np.arccos(self.alpha[0])
    @property
    def L0(self) -> float:
        """The length of the rotator (m). Set to zero. Present for compatibility purposes."""
        return 0

    @property
    def J0(self) -> npt.NDArray[np.complex_]:
        """The Jones matrix of the rotator."""
        aVec = self.alpha[1:]/np.sin(self.theta)
        if (self.theta == 0):
            aVec = np.array([0, 0, 0])
        J0 = np.cos(self.theta)*np.array([[1, 0], [0, 1]]) - 1j*np.sin(self.theta)*(aVec[0]*np.array([[1, 0], [0, -1]]) + aVec[1]*np.array([[0, 1], [1, 0]]) + aVec[2]*np.array([[0, -1j], [1j, 0]]))
        return J0

    def __init__(self, alpha: npt.NDArray[np.float64]) -> None:
        """
        Initialize a Rotator.

        Parameters
        ----------
        alpha: np.array[4]
            A 4-vector defining the rotation, where alpha[0] is cos(theta)
            and alpha[1:] is the axis of rotation (a 3-vector).
        """
        self.alpha = alpha/np.linalg.norm(alpha)

    def calcPhaseDelay(self) -> npt.NDArray[np.float64]:
        """
        Calculates the time for light to propagate through the fiber.
        This is a dummy function for compatibility purposes, as the
        rotator does not have a length.

        Returns
        -------
        np.array[3]
            [0,0,0]
        """
        return np.array([0, 0, 0])

    def __str__(self):
        return "Arbitrary rotator over angle {:.4f}° about an axis.".format(self.theta*180/pi)

# Fiber class definition
# -------------------------------------------------------------------------------------------------
class Fiber():
    """
    Implements a full optical fiber with alternating segments and hinges
    according to the "hinge model" of optical fibers.
    """

    segmentDictKeys = np.array(['L0', 'T0', 'Tref', 'epsilon', 'm0', 'm1', 'mProps', 'r0', 'r1', 'rc', 'tf', 'tr'])
    hingeDictKeys = np.array(['Ns', 'T0', 'Tref', 'angles', 'epsilon', 'finalTwistBool', 'gapLs', 'm0', 'm1', 'mProps', 'nPaddles', 'r0', 'r1', 'rps', 'tfs'])

    @property
    def arbRotStart(self) -> bool:
        """
        Boolean, whether to start the fiber with an arbitrary rotation
        (sometimes useful in simulatory applications)
        """
        return self.arbRotStart

    @arbRotStart.setter
    def arbRotStart(self, newVal: bool) -> None:
        self.toggleStartRotator(newVal)

    @property
    def addRotators(self) -> None | float | dict:
        return self.addRotators

    @addRotators.setter
    def addRotators(self, newVal: None | float | dict) -> None:
        self.toggleAddedRotators(newVal)

    @property
    def fibers(self) -> typing.List[FiberLength] | typing.Tuple[typing.List[FiberLength], typing.List[int]]:
        """The array of FiberLength and FiberPaddleSet objects constituting the fiber"""
        # The actual number of hinges
        N0h = self.N0-1 + int(self.hingeStart) + int(self.hingeEnd)
        # Take some compliance measures on the dictionaries
        # Look for optional parameters and set appropriately if not found
        ref = [[self.segmentDict, self.N0]]
        if (self.hingeType == 0):
            ref = [[self.segmentDict, self.N0], [self.hingeDict, N0h]]
        for d, n in ref:
            if ('mProps' in d.keys() and d['mProps'] != {}):
                d['m0'] = np.zeros(n)
                d['m1'] = np.zeros(n)
            else:
                d['mProps'] = {}
        if (self.hingeType == 0) and ('finalTwistBool' not in self.hingeDict.keys()):
            self.hingeDict['finalTwistBool'] = np.zeros(N0h)
        if (self.hingeType == 1) and ('alpha' not in self.hingeDict.keys()):
            self.hingeDict['alpha'] = []

        # Now both dicts should have a specific set of keys... check that that's true
        d1 = np.sort(list(self.segmentDict.keys()))
        if (len(d1) != len(self.segmentDictKeys)):
            raise Exception("Your fiber segment dictionary does not have the expected number of keys." +
                            "\nHere are the necessary keys: " + str(self.segmentDictKeys) +
                            "\nHere is your dictionary:     " + str(d1)
                            )
        elif (not all(d1 == self.segmentDictKeys)):
            raise Exception("Something in your fiber segment dictionary is specified incorrectly." +
                            "\nHere are the necessary keys: " + str(self.segmentDictKeys) +
                            "\nHere is your dictionary:     " + str(d1)
                            )
        d2 = np.sort(list(self.hingeDict.keys()))
        if (len(d2) != len(self.hingeDictKeys)) and (len(d2) != 1):
            raise Exception("Your hinge dictionary does not have the expected number of keys." +
                            "\nHere are the necessary keys: " + str(self.hingeDictKeys) + ' or [\'alpha\']' +
                            "\nHere is your dictionary:     " + str(d2)
                            )
        elif ((len(d2) == len(self.hingeDictKeys)) and (not all(d2 == self.hingeDictKeys))) or ((len(d2) == 1) and (not all(d2 == np.array(['alpha'])))):
            raise Exception("Something in your hinge dictionary is specified incorrectly." +
                            "\nHere are the necessary keys: " + str(self.hingeDictKeys) + ' or [\'alpha\']' +
                            "\nHere is your dictionary:     " + str(d2)
                            )

        # Now we're going to check each property to see if it's the correct-sized
        # array. If it's a single number or a 1×whatever array, do the conversion
        # here. Throw errors if I really can't make it fit.

        # Pre-process r0 and mProps together (because we need r0 to process mProps)
        # Do the following for both segmentDict and hingeDict
        ref = [[self.segmentDict, self.N0]]
        if (self.hingeType == 0):
            ref = [[self.segmentDict, self.N0], [self.hingeDict, N0h]]
        for d, n in ref:
            # Check if mProps is a dict and, if so, if it has the right keys
            if not isinstance(d['mProps'], dict):
                raise Exception("mProps needs to be a dictionary but is not.")
            elif not ((d['mProps'] == {}) or (np.all([k in d['mProps'].keys() for k in ['dn', 'w0', 'T']]) and np.any([k in d['mProps'].keys() for k in ['n0', 'n1', 'm0', 'm1', 'neff']]))):
                raise Exception("mProps dictionary is invalid.")
            else:
                # mProps is a valid dictionary
                if (isinstance(d['r0'], int | float | np.int32 | np.float64)) or (isinstance(d['r0'], np.ndarray) and (len(d['r0']) == 1)):
                    # r0 is a single number...
                    if (d['mProps'] == {}):
                        # mProps isn't being used, so m0 and m1 are
                        # So just standardize r0
                        d['r0'] = np.array([d['r0']]*n).flatten()
                    else:
                        # Everything is a single number and we can compute it for all fiber
                        # segments and be done
                        a, b = _fromDiffN(d['mProps'], d['r0'])
                        d['m0'] = np.array([a]*n)
                        d['m1'] = np.array([b]*n)
                        d['r0'] = np.array([d['r0']]*n).flatten()
                elif (len(d['r0']) != n):
                    raise Exception("The r0 array of the dictionary is the wrong length.")
                else:
                    # If mProps is an empty dictionary don't need to do anything else
                    if (d['mProps'] != {}):
                        # r0 is an array of the right length
                        d['m0'] = np.zeros(n)
                        d['m1'] = np.zeros(n)
                        for i in range(n):
                            a, b = _fromDiffN(d['mProps'], d['r0'][i])
                            d['m0'][i] = a
                            d['m1'][i] = b

        # Conversions for segments
        self.segmentDict.pop('mProps', None)
        for p in self.segmentDict.keys():
            if (isinstance(self.segmentDict[p], int | float | np.int32 | np.float64)) or (isinstance(self.segmentDict[p], np.ndarray) and (len(self.segmentDict[p]) == 1)):
                self.segmentDict[p] = np.array([self.segmentDict[p]]*self.N0).flatten()
            elif (len(self.segmentDict[p]) != self.N0):
                raise Exception("Array in segment dictionary with key " + str(p) + " has the wrong shape, should be 1×{:.0f} but is ".format(self.N0) + str(np.shape(self.segmentDict[p])))

        # Conversions for hinges
        if (N0h != 0):
            self.hingeDict.pop('mProps', None)
            if (self.hingeType == 0):
                if (isinstance(self.hingeDict['nPaddles'], int | float | np.int32 | np.float64)) or (isinstance(self.hingeDict['nPaddles'], np.ndarray) and (len(self.hingeDict['nPaddles']) == 1)):
                    self.hingeDict['nPaddles'] = np.array([self.hingeDict['nPaddles']]*N0h).flatten()
                elif (len(self.hingeDict['nPaddles']) != N0h):
                    raise Exception("Array in hinge dictionary with key nPaddles has the wrong shape, should be 1×{:.0f} but is ".format(N0h) + str(np.shape(self.hingeDict['nPaddles'])))
                for p in self.hingeDict.keys():
                    if (p not in ['rps', 'angles', 'tfs', 'Ns', 'gapLs']):
                        if (isinstance(self.hingeDict[p], int | float | np.int32 | np.float64)) or (isinstance(self.hingeDict[p], np.ndarray) and (len(self.hingeDict[p]) == 1)):
                            self.hingeDict[p] = np.array([self.hingeDict[p]]*N0h).flatten()
                        elif (len(self.hingeDict[p]) != N0h):
                            raise Exception("Array in hinge dictionary with key " + str(p) + " has the wrong shape, should be 1×{:.0f} but is ".format(N0h) + str(np.shape(self.hingeDict[p])))
                    else:
                        if (len(np.shape(self.hingeDict[p])) == 1):
                            if (len(self.hingeDict[p]) >= np.max(self.hingeDict['nPaddles'])):
                                self.hingeDict[p] = np.array([list(self.hingeDict[p])]*N0h)
                            else:
                                raise Exception("Array in hinge dictionary with key " + str(p) + " has the wrong shape. It should be 1×(at least {:.0f}) or {:.0f}×(at least {:.0f}) but is ".format(np.max(self.hingeDict['nPaddles']), N0h, np.max(self.hingeDict['nPaddles'])) + str(np.shape(self.hingeDict[p])))
                        elif (len(np.shape(self.hingeDict[p])) == 2) and (np.shape(self.hingeDict[p])[0] == 1):
                            if (len(self.hingeDict[p][0]) >= np.max(self.hingeDict['nPaddles'])):
                                self.hingeDict[p] = np.array([list(self.hingeDict[p][0])]*N0h)
                            else:
                                raise Exception("Array in hinge dictionary with key " + str(p) + " has the wrong shape. It should be 1×(at least {:.0f}) or {:.0f}×(at least {:.0f}) but is ".format(np.max(self.hingeDict['nPaddles']), N0h, np.max(self.hingeDict['nPaddles'])) + str(np.shape(self.hingeDict[p])))
                        elif ((np.shape(self.hingeDict[p])[0] != N0h) or (np.shape(self.hingeDict[p])[1] < np.max(self.hingeDict['nPaddles']))):
                            raise Exception("Array in hinge dictionary with key " + str(p) + " has the wrong shape. It should be 1×(at least {:.0f}) or {:.0f}×(at least {:.0f}) but is ".format(np.max(self.hingeDict['nPaddles']), N0h, np.max(self.hingeDict['nPaddles'])) + str(np.shape(self.hingeDict[p])))
            elif (self.hingeType == 1):
                if (len(np.shape(self.hingeDict['alpha'])) == 1):
                    if (len(self.hingeDict['alpha']) == 4):
                        self.hingeDict['alpha'] = np.array([list(self.hingeDict['alpha'])]*N0h)
                    else:
                        raise Exception("Alpha specs for rotator hinges should be 1×4 or {:.0f}×4, but your spec is".format(N0h) + str(np.shape(self.hingeDict['alpha'])))
                elif (len(np.shape(self.hingeDict['alpha'])) == 2) and (np.shape(self.hingeDict['alpha'])[0] == 1):
                    if (len(self.hingeDict['alpha'][0]) == 4):
                        self.hingeDict['alpha'] = np.array([list(self.hingeDict['alpha'][0])]*N0h)
                    else:
                        raise Exception("Alpha specs for rotator hinges should be 1×4 or {:.0f}×4, but your spec is".format(N0h) + str(np.shape(self.hingeDict['alpha'])))
                elif (np.shape(self.hingeDict['alpha']) != (N0h, 4)):
                    raise Exception("Alpha specs for rotator hinges should be 1×4 or {:.0f}×4, but your spec is".format(N0h) + str(np.shape(self.hingeDict['alpha'])))

        # Now make the fiber array.
        # If we aren't putting rotators in the segments, then just interleave the segments
        # and hinges properly and call it a day. If we ARE adding rotators, then make the
        # rotators and the proper-length segments and interleave those, putting a hinge
        # where appropriate.
        # Pre-make the hinge objects, whatever they are:
        hinges = np.array([], dtype=object)
        if (self.hingeType == 0):
            for i in range(N0h):
                npad = self.hingeDict['nPaddles'][i]
                ftb = self.hingeDict['finalTwistBool'][i]
                hinges = np.append(hinges, FiberPaddleSet(self.w0, self.hingeDict['T0'][i], self.hingeDict['r0'][i], self.hingeDict['r1'][i], self.hingeDict['epsilon'][i], self.hingeDict['m0'][i], self.hingeDict['m1'][i], self.hingeDict['Tref'][i], npad, self.hingeDict['rps'][i][:npad], self.hingeDict['angles'][i][:npad], self.hingeDict['tfs'][i][:npad], self.hingeDict['Ns'][i][:npad], self.hingeDict['gapLs'][i][:int(npad+int(ftb))], mProps={}, finalTwistBool=ftb))
        elif (self.hingeType == 1):
            for i in range(N0h):
                hinges = np.append(hinges, Rotator(self.hingeDict['alpha'][i]))

        # Now put it all together.
        hingeInds = np.array([], dtype=int)
        fa = np.array([], dtype=object)
        if (self.arbRotStart):
            fa = np.append(fa, self.startRotator)
        if (self.hingeStart):
            fa = np.append(fa, hinges[0])
            hingeInds = np.append(hingeInds, 0)
        for i in range(self.N0):
            # Add the segment
            if (self.addRotators is None):
                fa = np.append(fa, FiberLength(self.w0, self.segmentDict['T0'][i], self.segmentDict['L0'][i], self.segmentDict['r0'][i], self.segmentDict['r1'][i], self.segmentDict['epsilon'][i], self.segmentDict['m0'][i], self.segmentDict['m1'][i], self.segmentDict['Tref'][i], self.segmentDict['rc'][i], self.segmentDict['tf'][i], self.segmentDict['tr'][i], mProps={}))
            else:
                Ns = 0
                thisSegmentDict = {'T0': self.segmentDict['T0'][i], 'L0': self.addedRotators['L'][i], 'r0': self.segmentDict['r0'][i], 'r1': self.segmentDict['r1'][i], 'epsilon': self.segmentDict['epsilon'][i], 'm0': self.segmentDict['m0'][i], 'm1': self.segmentDict['m1'][i], 'Tref': self.segmentDict['Tref'][i], 'rc': self.segmentDict['rc'][i], 'tf': self.segmentDict['tf'][i], 'tr': self.segmentDict['tr'][i], 'mProps': {}}
                if isinstance(self.addRotators, int | float | np.int32 | np.float64):
                    Ns, _ = np.divmod(self.segmentDict['L0'][i], self.addRotators)
                elif isinstance(self.addRotators, dict):
                    Ns, _ = np.divmod(self.segmentDict['L0'][i], self.addRotators['mean'])
                fseg = Fiber(self.w0, thisSegmentDict, {'alpha': self.addedRotators['alpha'][i]}, int(Ns)+1, hingeType=1, hingeStart=False, hingeEnd=False, arbRotStart=False, addRotators=None)
                fa = np.append(fa, fseg.fibers.copy()).flatten()
                del fseg
            # Add the next hinge
            if ((i != self.N0-1) or self.hingeEnd):
                fa = np.append(fa, hinges[i + int(self.hingeStart)])
                hingeInds = np.append(hingeInds, len(fa)-1)

        if (self._printingBool):
            return fa, hingeInds
        return fa

    @property
    def J0(self) -> npt.NDArray[np.complex_]:
        """The total Jones matrix of the fiber"""
        fa = self.fibers
        Jtot = np.array([[1, 0], [0, 1]])
        for i in range(len(fa)):
            Jtot = np.matmul(fa[i].J0, Jtot)
        return Jtot

    @property
    def L0(self) -> float | typing.Tuple[typing.List[FiberLength], typing.List[int], float]:
        """The total length of the fiber"""
        fa = 0
        hingeInds = 0
        if (self._printingBool):
            fa, hingeInds = self.fibers
        else:
            fa = self.fibers
        L0 = 0
        for i in range(len(fa)):
            L0 = L0 + fa[i].L0
        if (self._printingBool):
            return fa, hingeInds, L0
        return L0

    # N0: number of long segments
    # hingeType = 0 (fiber paddle sets), 1 (arbitrary rotators)
    def __init__(self, w0: float, segmentDict: dict, hingeDict: dict, N0: int, hingeType: int = 0,
                 hingeStart: bool = True, hingeEnd: bool = True, arbRotStart: bool = False,
                 addRotators: bool=None) -> None:
        
        """
        Parameters
        ----------
        w0 : float
            Wavelength of light (m)
        segmentDict : dict
            A dictionary containing the properties of the long segments of the fiber.
            Should contain keys 'T0', 'L0', 'r0', 'r1', 'epsilon', 'm0' and 'm1' or
            'mProps', 'Tref', 'rc', 'tf', 'tr' which are each either single numbers or
            arrays of length N0 (if single number, all segments will have that number
            as their property).
        hingeDict : dict
            A dictionary containing the properties of the hinges of the fiber. If
            hingeType = 1, this dictionary only contains 'alpha', a length-4 array
            or a (4×N0h)-length array. If hingeType = 0, this dictionary will needs keys
            'T0', 'r0', 'r1', 'epsilon', 'm0' and 'm1' or 'mProps', 'Tref', 'nPaddles',
            'finalTwistBool' (which are single numbers or 1×N0h arrays) and 'rps', 'angles',
            'tfs', 'Ns', 'gapLs' (which are 1×nPaddles arrays or N0h×nPaddles arrays).
            Here N0h = N0-1 + hingeStart + hingeEnd.
        N0 : int
            The number of long segments of fiber.
        hingeType : :obj:`int`, optional
            0 means hinges are FiberPaddleSets, 1 means hinges are arbitrary Rotators
            (default 0)
        hingeStart : :obj:`bool`, optional
            Whether there's a hinge before the first fiber segment; if arbRotStart
            is True, the arbitrary rotation precedes this first hinge (default True)
        hingeEnd : :obj:`bool`, optional
            Whether there's a hinge after the last fiber segment (default True)
        arbRotStart : :obj:`bool`, optional
            Whether to start the fiber with an arbitrary rotator (default False)
        addRotators : :obj:`None`, :obj:`float`, or :obj:`dict`, optional
            A parameter to add arbitrary rotators along the fiber segments, separate
            from the hinges. Can be either None (no rotators added) or one of the
            following:
            (1) a single number, the exact distance between each rotator (in meters)
            (it will be rounded automatically to the nearest multiple of the fiber
            length);
            (2) a dictionary with 'mean', 'scale', and 'dist' to get random distances
            between rotators (see _getRandom documentation for more details).
        """
        
        self.w0 = w0
        self.N0 = N0
        self.segmentDict = segmentDict
        self.hingeDict = hingeDict
        self.hingeType = hingeType
        self.hingeStart = hingeStart
        self.hingeEnd = hingeEnd
        self.arbRotStart = arbRotStart
        self.addRotators = addRotators

        # This is a flag for internal use only
        self._printingBool = False

    def toggleStartRotator(self, newVal: bool) -> None:
        """
        Toggles the start rotator of the fiber.
        If newVal is True, the start rotator is set to an arbitrary
        rotator with a random rotation; if False, the start rotator is set
        to the identity rotator (a Rotator with alpha = [1, 0, 0, 0]).
        """
        if (newVal):
            self.arbRotStart = True
            self.startRotator = makeRotators(1)[0]
        else:
            self.arbRotStart = False
            self.startRotator = Rotator([1, 0, 0, 0])

    def toggleAddedRotators(self, newVal: None | float | dict) -> None:
        """
        Toggles the added rotators of the fiber.
        If newVal is None, no rotators are added; if it is a number,
        then that number is the distance between each rotator (in meters);
        if it is a dictionary, then the dictionary should contain 'mean',
        'scale', and 'dist' to get random distances between rotators
        (see _getRandom documentation for more details).
        """
        if (newVal is None):
            self.addedRotators = None
            self.addRotators = None
        else:
            # Gotta get the lengths right...
            Ls = 0
            if (isinstance(self.segmentDict['L0'], int | float | np.int32 | np.float64)) or (isinstance(self.segmentDict['L0'], np.ndarray) and (len(self.segmentDict['L0']) == 1)):
                Ls = np.array([self.segmentDict['L0']]*self.N0).flatten()
            elif (len(self.segmentDict['L0']) != self.N0):
                raise Exception("The numbers of segments don't match. Should be 1×{:.0f} but is ".format(self.N0) + str(np.shape(self.segmentDict['L0'])))
            else:
                Ls = self.segmentDict['L0']
            # Now make the rotators
            self.addedRotators = {'alpha': {}, 'L': {}}
            if isinstance(newVal, int | float | np.int32 | np.float64):
                for i in range(self.N0):
                    Ns, rem = np.divmod(Ls[i], newVal)
                    if (Ns == 0):
                        raise Exception("The given length is larger than the available length, so no rotators are fitting in this segment.")
                    Ns = int(Ns)
                    self.addRotators = newVal
                    self.addedRotators['alpha'][i] = _getRandom((Ns, 4), **_randomDistDefaults['alpha'])
                    self.addedRotators['L'][i] = newVal + rem/Ns
            elif isinstance(newVal, dict):
                for i in range(self.N0):
                    Ns, _ = np.divmod(Ls[i], newVal['mean'])
                    if (Ns == 0):
                        raise Exception("The given length is larger than the available length, so no rotators are fitting in this segment.")
                    Ns = int(Ns)
                    self.addRotators = newVal
                    self.addedRotators['alpha'][i] = _getRandom((Ns, 4), **_randomDistDefaults['alpha'])
                    self.addedRotators['L'][i] = _getRandom(int(Ns+1), **newVal)

    def calcDGD(self, dw0: float = 0.1e-9) -> float:
        """
        Calculates the DGD of the fiber length using a small wavelength change.

        Parameters
        ----------
        dw0 : :obj:`float`, optional
            The small wavelength step to take (m). Default is 0.1 nm.

        Returns
        -------
        dgd : float
            DGD in s.
        """

        # Store current variables
        wb = self.w0
        Jb = self.J0

        # Get Jones matrices for ± dw0
        self.w0 = self.w0 - dw0
        Ja = self.J0
        self.w0 = self.w0 + 2*dw0
        Jc = self.J0

        # Reset values for this object
        self.w0 = wb

        # Get DGD estimates for ± dw0
        matM = np.matmul(Ja, np.linalg.inv(Jb))
        vals, vecs = np.linalg.eig(matM)
        dgdM = np.abs(np.angle(vals[0]/vals[1])/((2*pi*C_c)/self.w0**2*dw0))

        matP = np.matmul(Jc, np.linalg.inv(Jb))
        vals, vecs = np.linalg.eig(matP)
        dgdP = np.abs(np.angle(vals[0]/vals[1])/((2*pi*C_c)/self.w0**2*dw0))

        # Average and return
        dgd = (dgdM + dgdP)/2
        return dgd

    def getHingeLocations(self) -> typing.List[int]:
        """
        Returns an array of the indices in self.fibers that are hinges.
        """
        self._printingBool = True
        _, hingeInds = self.fibers
        self._printingBool = False
        return hingeInds

    @classmethod
    def random(cls, w0: float, Ltot: float, N0: int, segmentDict: dict, hingeDict: dict,
               hingeType: int = 0, hingeStart: bool = True, hingeEnd: bool = True,
               arbRotStart: bool = False, addRotators: bool = None) -> 'Fiber':
        """
        Generates a random optical fiber following input specs.

        The general idea for specifying fibers here is that the dictionaries
        will contain either numbers or arrays that will be used directly,
        specs for the distribution to randomly draw from, or nothing at all,
        in which case numbers will be drawn from random defaults. For example,
        for the segment core noncircularities, if segmentDict contains an entry
        'epsilon': 1.005, then all the segments will have epsilon = 1.005; it could
        also be a N0-length array specifying the epsilon for each segment. (Properties
        specified directly must follow the rules for doing so, see e.g. FiberPaddleSet
        documentation for more details.) Or one can specify the properties of the
        distribution to be randomly drawn from; for example,
        'epsilon': {'mean': 1.005, 'scale': 0.002, 'dist': 'uniform'} draws the
        epsilons for all the segments from a uniform distribution between 1.003 and
        1.007 (see also the _getRandom() documentation). If segmentDict has NO entry
        'epsilon', then epsilons will be drawn randomly using the information in
        _randomDistDefaults.

        If the lengths are left to randomness, the method corrects the fiber segments
        to ensure that the total length of the fiber is the specified Ltot.

        Parameters
        ----------
        w0 : float
            Wavelength of light (m)
        Ltot : float
            Total length of the fiber (m)
        N0 : int
            Number of long birefringent segments
        segmentDict : dict
            The dictionary containing the info about the segments, following the above description.
        hingeDict : dict
            The dictionary containing the info about the hinges, following the above description.
        hingeType : :obj:`int`, optional
            0 means hinges are FiberPaddleSets, 1 means hinges are arbitrary Rotators
            (default 0)
        hingeStart : :obj:`bool`, optional
            Whether there's a hinge before the first fiber segment; if arbRotStart
            is True, the arbitrary rotation precedes this first hinge (default True)
        hingeEnd : :obj:`bool`, optional
            Whether there's a hinge after the last fiber segment (default True)
        arbRotStart : :obj:`bool`, optional
            Whether to start the fiber with an arbitrary rotator (default False)
        addRotators : :obj:`None`, :obj:`float`, or :obj:`dict`, optional
            A parameter to add arbitrary rotators along the fiber segments, separate
            from the hinges. Can be either None (no rotators added) or one of the
            following:
            (1) a single number, the exact distance between each rotator (in meters);
            (2) a dictionary with 'mean', 'scale', and 'dist' to get random distances
            between rotators (see _getRandom documentation for more details).

        Returns
        -------
        Fiber
            A random Fiber following the given specifications.
        """

        N0h = N0 - 1 + hingeStart + hingeEnd

        newSegmentDict = {}
        newHingeDict = {}
        hingeLength = 0

        # Start with properties that can reuse code for both segments and hinges
        ref = [[segmentDict, newSegmentDict, N0]]
        if (hingeType == 0):
            ref = [[segmentDict, newSegmentDict, N0], [hingeDict, newHingeDict, N0h]]
        for d1, d2, n in ref:
            # Have to handle the doping fraction which can be specified one of two ways
            # If neither 'm0','m1' nor 'mProps' is in there, let the random defaults handle it
            # If they're both in there, pass both along; Fiber() class logic handles it
            if (('m0' not in d1.keys()) or ('m1' not in d1.keys())) and ('mProps' in d1.keys()):
                d1['m0'] = 0
                d1['m1'] = 0
            elif (('m0' in d1.keys()) and ('m1' in d1.keys())) and ('mProps' not in d1.keys()):
                d1['mProps'] = {}
            # Now loop over each property and pass or randomize as necessary
            for prop in ['T0', 'Tref', 'epsilon', 'r0', 'r1', 'rc', 'tf', 'tr', 'm0', 'm1', 'mProps']:
                da = d1
                if (prop not in d1.keys()):
                    da = _randomDistDefaults

                if isinstance(da[prop], int | float | np.int32 | np.float64 | np.ndarray):
                    d2[prop] = da[prop]
                elif (isinstance(da[prop], dict) and all([s in da[prop].keys() for s in ['mean', 'scale', 'dist']])):
                    d2[prop] = _getRandom(n, **da[prop])
                elif (prop == 'mProps'):
                    # Just gonna assume that what's specified is fine
                    d2[prop] = da[prop]
                else:
                    raise Exception("On property " + str(prop) + ", something is incorrectly specified.")
            # Save a few lines of code by just doing this
            newHingeDict.pop('rc', None)
            newHingeDict.pop('tr', None)
            newHingeDict.pop('tf', None)

        if (hingeType == 0):
            # nPaddles is important for remaining hinge properties
            if ('nPaddles' in hingeDict.keys()):
                if isinstance(hingeDict['nPaddles'], int | float | np.int32 | np.float64 | np.ndarray):
                    newHingeDict['nPaddles'] = hingeDict['nPaddles']
                elif (isinstance(hingeDict['nPaddles'], dict) and all([s in hingeDict['nPaddles'].keys() for s in ['mean', 'scale']])):
                    hingeDict['nPaddles']['dist'] = 'uniform_int'
                    newHingeDict['nPaddles'] = _getRandom(N0h, **hingeDict['nPaddles'])
            else:
                newHingeDict['nPaddles'] = _getRandom(N0h, **_randomDistDefaults['nPaddles'])
            # Handle finalTwistBool
            if ('finalTwistBool' in hingeDict.keys()):
                newHingeDict['finalTwistBool'] = hingeDict['finalTwistBool']
            else:
                newHingeDict['finalTwistBool'] = False
            # Now do the arrayed hinge properties
            nPadMax = np.max(newHingeDict['nPaddles'])
            for prop in ['Ns', 'angles', 'tfs', 'rps', 'gapLs']:
                if (prop in hingeDict.keys()):
                    if isinstance(hingeDict[prop], int | float | np.int32 | np.float64 | np.ndarray):
                        newHingeDict[prop] = hingeDict[prop]
                    elif (isinstance(hingeDict[prop], dict) and all([s in hingeDict[prop].keys() for s in ['mean', 'scale', 'dist']])):
                        # I add 1 for gapLs in case there's a finalTwistBool = True...
                        # the later methods will just discard it if not needed
                        newHingeDict[prop] = _getRandom((N0h, nPadMax + 1*(prop == 'gapLs')), **hingeDict[prop])
                else:
                    if isinstance(_randomDistDefaults[prop], dict):
                        newHingeDict[prop] = _getRandom((N0h, nPadMax + 1*(prop == 'gapLs')), **_randomDistDefaults[prop])
                    else:
                        newHingeDict[prop] = np.ones((N0h, nPadMax + 1*(prop == 'gapLs')))*_randomDistDefaults[prop]
            # Finally, do the lengths, ensuring they add up to Ltot including the hinge lengths
            # Have to calculate the hinge lengths first...
            hingeLengthCalcs = {}
            # The arrays have to be max(nPaddles) for uniformity, but not all those
            # numbers will be used, so let's gather that info first
            if isinstance(newHingeDict['nPaddles'], int | float | np.int32 | np.float64):
                hingeLengthCalcs['nPaddles'] = np.array([newHingeDict['nPaddles']]*N0h)
            else:
                hingeLengthCalcs['nPaddles'] = newHingeDict['nPaddles']
            if isinstance(newHingeDict['finalTwistBool'], bool | int | float | np.int32 | np.float64):
                hingeLengthCalcs['finalTwistBool'] = np.array([newHingeDict['finalTwistBool']]*N0h)
            else:
                hingeLengthCalcs['finalTwistBool'] = newHingeDict['finalTwistBool']
            # Now for the actual lengths...
            for prop in ['gapLs', 'rps', 'Ns']:
                if (len(np.shape(newHingeDict[prop])) == 1):
                    hingeLengthCalcs[prop] = np.array([list(newHingeDict[prop])]*N0h)
                elif (len(np.shape(newHingeDict[prop])) == 2) and (np.shape(newHingeDict[prop])[0] == 1):
                    hingeLengthCalcs[prop] = np.array([list(newHingeDict[prop][0])]*N0h)
                else:
                    hingeLengthCalcs[prop] = newHingeDict[prop]
            # Now do the appropriate summation
            hingeLength = 2*pi*np.sum(np.array([np.sum(hingeLengthCalcs['rps'][i][:hingeLengthCalcs['nPaddles'][i]] * hingeLengthCalcs['Ns'][i][:hingeLengthCalcs['nPaddles'][i]]) for i in range(N0h)])) + np.sum(np.array([np.sum(hingeLengthCalcs['gapLs'][i][:int(hingeLengthCalcs['nPaddles'][i] + int(hingeLengthCalcs['finalTwistBool'][i]))]) for i in range(N0h)]))

        elif (hingeType == 1):
            if ('alpha' in hingeDict.keys()):
                # Must be 1×4 or N0h×4 array or dict for random generation
                if isinstance(hingeDict['alpha'], np.ndarray):
                    newHingeDict['alpha'] = hingeDict['alpha']
                elif isinstance(hingeDict['alpha'], dict):
                    newHingeDict['alpha'] = _getRandom((N0h, 4), **hingeDict['alpha'])
            else:
                newHingeDict['alpha'] = _getRandom((N0h, 4), **_randomDistDefaults['alpha'])
            hingeLength = 0

        # Now generate segment lengths...
        if ('L0' in segmentDict.keys()):
            if isinstance(segmentDict['L0'], int | float | np.int32 | np.float64):
                newSegmentDict['L0'] = np.array([segmentDict['L0']]*N0)
            elif isinstance(segmentDict['L0'], np.ndarray):
                newSegmentDict['L0'] = segmentDict['L0']
            elif (isinstance(segmentDict['L0'], dict) and all([s in segmentDict['L0'].keys() for s in ['scale', 'dist']])):
                newSegmentDict['L0'] = _getRandom(N0, (Ltot - hingeLength)/N0, **segmentDict['L0'])
        else:
            newSegmentDict['L0'] = _getRandom(N0, (Ltot - hingeLength)/N0, **_randomDistDefaults['L0'])
        segmentLength = np.sum(newSegmentDict['L0'])
        # Ensure total length compliance with Ltot
        lengthDiff = (segmentLength + hingeLength) - Ltot
        newSegmentDict['L0'] = newSegmentDict['L0'] - (lengthDiff/N0)

        return cls(w0, newSegmentDict, newHingeDict, N0, hingeType=hingeType, hingeStart=hingeStart, hingeEnd=hingeEnd, arbRotStart=arbRotStart, addRotators=addRotators)

    def calcPhaseDelay(self) -> npt.NDArray[np.float64]:
        """
        Calculates the time for light to propagate through the fiber.

        Returns
        -------
        np.array[3]
            average, min, max transit time through the fiber (s)
        """
        fa = self.fibers
        t0 = np.array([0, 0, 0])
        for i in range(len(fa)):
            t0 = t0 + fa[i].calcPhaseDelay()
        return t0

    def __str__(self) -> str:
        self._printingBool = True
        fa, hingeInds, L0 = self.L0
        hingeTypeStr = "sets of fiber paddles"
        if (self.hingeType == 1):
            hingeTypeStr = "arbitrary rotators"
        info_array = ["Optical fiber of total length {:.4f} m, containing {:.0f} separate fiber objects.".format(L0, len(fa))]
        if isinstance(self.addRotators, int | float | np.int32 | np.float64):
            info_array = info_array + ["There are arbitrary rotators, added every {:.4f} m.".format(self.addRotators)]
        elif isinstance(self.addRotators, dict):
            info_array = info_array + ["There are arbitrary rotators added at random distances with mean {:.4f} and scale {:.4f}.".format(self.addRotators['mean'], self.addRotators['scale'])]
        info_array = info_array + ["The hinges, which are " + hingeTypeStr + ", are at the following indices of the object.fibers array.", str(hingeInds)]
        self._printingBool = False
        sep = "\n"
        return sep.join(info_array)
