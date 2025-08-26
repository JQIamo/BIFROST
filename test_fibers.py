import numpy as np
from scipy import optimize as opt
import numpy.typing as npt
import typing
import unittest
import copy
import fibers   
import pytest
# also using plugin pytest-assume to permit more than one test per method
from matplotlib import pyplot as plt
from scipy import optimize as opt

DEBUG = False


class TestBifrostSMF28():

    # The goal of this class is to implement unit tests that confirm the codebase
    # can reproduce the behavior of Corning SMF28 in certain limits. The basis 
    # for this is the BIFROST paper section VIa titled validation of intrinsic 
    # fiber properties.
    #  
    # inputs: core diameter 8.2e-6 m, cladding diameter 125e-6 m
    #         core cladding refractive index difference 0.36%
    #         reference temperature 23 C
    #         molar fraction of GeO2 in core 0.03652

    @pytest.fixture(scope="class")
    def bifrost_smf28(self):
        # factory function for the generic bifrost-smf28
        def _bifrost_smf28(w0=1550e-9):
            # Initialize the fiber with SMF28-like parameters
            # w0              # Operating wavelength
            T0 = 23           # Operating temperature
            L0 = 1000         # Length of fiber
            r0 = 4.1e-6       # Radius of core
            r1 = 125e-6 / 2   # Radius of cladding
            epsilon = 1.005   # Core noncircularity
            m0 = 0.03679      # Doping concentration in core
            m1 = 0.0          # Doping concentration in cladding
            Tref = 23         # Reference temperature
            rc = 0            # Bend radius of curvature
            tf = 0            # Axial tension
            tr = 0            # Twist rate

            return fibers.FiberLength(
                w0, T0, L0, r0, r1, epsilon, m0, m1, Tref, rc, tf, tr, mProps={}
            )
        return _bifrost_smf28

    @pytest.mark.parametrize("w0, smf28_dcd_upperbound",
            [(1550e-9, 18), (1625e-9, 22)])
    def test_chromatic_dispersion_corning(self, bifrost_smf28, w0, 
            smf28_dcd_upperbound):
        # Bounds from Corning SMF28e+ datasheet  ps/(nm km)
        assert bifrost_smf28(w0).calcD_CD() < smf28_dcd_upperbound

    @pytest.mark.parametrize("w0, bifrostv1_dcd",
        [(1550e-9, 12.31), (1625e-9, 16.15)])
    def test_chromatic_dispersion_bifrostv1(self, bifrost_smf28, w0, 
            bifrostv1_dcd):
        # Expected output based on BIFROST v1 (12/2024)
        print(bifrost_smf28(w0).calcD_CD())
        assert np.isclose(bifrost_smf28(w0).calcD_CD(), bifrostv1_dcd, atol=0.1) 

    @pytest.fixture(scope="class")
    def zero_dispersion(self, bifrost_smf28):
        f = bifrost_smf28()
        w0s = np.linspace(1200, 1650, 1001)*1e-9
        dcds = np.zeros(len(w0s))
        for i in range(len(w0s)):
            f.w0 = w0s[i]
            dcds[i] = f.calcD_CD()

        def testFunc(w0, S0, lambda0): return (S0/4)*(w0 - lambda0**4/w0**3)
        popt, pcov = opt.curve_fit(testFunc, w0s, dcds, p0 = np.array([0.073e9, 1350e-9]))
        if True:
            print("Fitted values are {:.4f} ps/(nm^2 km) at about {:.2f} nm.".format(popt[0]/1e9, popt[1]*1e9))

        return popt[1], popt[0], w0s, dcds

    def test_dispersion_function(self, zero_dispersion):
            w0s = zero_dispersion[2]
            dcds = zero_dispersion[3]

            def testFunc(w0, S0, lambda0): return (S0/4)*(w0 - lambda0**4/w0**3)
            popt, pcov = opt.curve_fit(testFunc, w0s, dcds,
                p0=np.array([0.073e9, 1350e-9]))

            # put bounds on 1-sigma errors 
            assert np.sqrt(pcov[0, 0]/1e9) < 3
            assert np.sqrt(pcov[1, 1]*1e9) < 1e-5

            if DEBUG:
                print("Fitted values are {:.4f} ps/(nm^2 km) at about {:.2f} nm."
                    .format(popt[0]/1e9, popt[1]*1e9))

                fig, ax = plt.subplots()
                ax.scatter(w0s*1e9, dcds)
                ax.set_xlabel("Wavelength (nm)")
                ax.set_ylabel("Group velocity dispersion $D_{CD}$ (ps/nm$ \\cdot $km)")
                ax.plot(w0s*1e9, testFunc(w0s, popt[0], popt[1]), color='orange', \
                        linestyle='dashed')
                ax.axhline(y=0, color='gray', linestyle='dashed')
                plt.show()

    def test_zero_dispersion_bifrostv1(self, zero_dispersion):
        # Expected output based on BIFROST v1 (12/2024)
        assert np.isclose(zero_dispersion[0], 1350e-9, atol=1e-9)
        assert np.isclose(zero_dispersion[1]/1e9, 0.0749, atol=0.005)

    def test_zero_dispersion_corning(self, zero_dispersion):
        # Expected output based on Corning SMF28 datasheet
        assert zero_dispersion[0] > 1304e-9 and zero_dispersion[0] < 1324e-9  
        assert zero_dispersion[1] <= 0.092 

    @pytest.mark.parametrize("w0, neff",
            [(1310e-9, 1.4674), (1625e-9, 1.4679)])
    def test_neff_corning(self, bifrost_smf28, w0, neff):
        assert np.isclose(bifrost_smf28(w0).calcNGEff(), neff, 0.0003)

    @pytest.mark.parametrize("w0, neff",
            [(1310e-9, 1.4676), (1625e-9, 1.4680)])
    def test_neff_bifrostv1(self, bifrost_smf28, w0, neff):
        assert np.isclose(bifrost_smf28(w0).calcNGEff(), neff, 0.0005)

    def test_cutoff_wavelength_corning(self, bifrost_smf28):
        f = bifrost_smf28()
        w0s = np.linspace(1100, 1400, 1001)*1e-9
        vs = np.zeros(len(w0s))
        for i in range(len(w0s)):
            f.w0 = w0s[i]
            vs[i] = f.v

        closestToCutoffInd = np.argmin(np.abs(vs - 2.405))
        cutoff = w0s[closestToCutoffInd]*1e9

        assert cutoff <= 1260e-9

    def test_cutoff_wavelength_bifrost(self, bifrost_smf28):
        f = bifrost_smf28()
        w0s = np.linspace(1100, 1400, 1001)*1e-9
        vs = np.zeros(len(w0s))
        for i in range(len(w0s)):
            f.w0 = w0s[i]
            vs[i] = f.v

        closestToCutoffInd = np.argmin(np.abs(vs - 2.405))
        cutoff = w0s[closestToCutoffInd]*1e9

        assert np.isclose(cutoff, 1318, 1)

    def test_polarization_beat_length(self, bifrost_smf28):
        # Beat length should be positive and on the order of meters for SMF28
        bl = bifrost_smf28().calcBeatLength()
        assert isinstance(bl, float)
        assert bl > 1 and bl < 50

    # def test_spinning_pmd_corning(self, bifrost_smf28):
    #     # TODO: PMD/sqrt(km) is an intrinsic property of Corning SMF28 fiber.
    #     # In BIFROST it is obtained by adding the right amount of spinning 
    #     # (implemented as rotations) to any fiber segment. The routine that
    #     # adds rotators as a mockup for  and the addition
    #     # of rotators should be moved so that the PMD due to spinning 

    #     assert False  # intentially fail due to non-existent implementation
    #     ipmd = bifrost_smf28.intrinsic_pmd()
    #     assert isinstance(ipmd, float)
    #     assert ipmd*1e12 < 0.02  # ps/sqrt(km)  
    #     # Corning datasheet circa 2005 reports both <0.2 and <0.02. It's 
    #     # internally inconsistent.
        
if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)