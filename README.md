# BIFROST

BIFROST (Birefringence In Fiber: Research and Optical Simulation Toolkit) is a Python library that provides a set of data, methods, and classes for the simulation of polarization mode dispersion in optical fibers. Silica-based fibers whose core and/or cladding are doped with germania can be simulated.

Specifically, the ``fibers.py`` module provides the following classes (see their individual documentations for more details):

* ``FiberLength``, the base class that stores information about the fiber geometry and conditions and calculates birefringences and Jones matrices;
* ``FiberPaddleSet``, a set of fiber paddles like ThorLabs FPC563 that manually controls the polarization;
* ``Rotator``, providing arbitrary rotations; and
* ``Fiber``, an implementation of the hinge model of optical fibers that alternates hinges with long birefringent sections. This class includes the ``Fiber.random()`` method for generating random optical fibers following user specifications.

This repository also includes an example Jupyter notebook for getting started, and includes the ``test_fibers.py`` unit test module. Currently this module's tests are known to succeed and fail in the following sequence: ``......F....F..`` .

The BIFROST paper is published in *Phys Rev Applied* at DOI: [10.1103/xgqr-rlmf](https://journals.aps.org/prapplied/abstract/10.1103/xgqr-rlmf). You can also view it [on the Arxiv](https://arxiv.org/abs/2510.01212). If you use this library, please cite the paper.

**We are currently working on a significant refactoring of this library. Check back soon. In the meantime, this version of the library is fully functional.**
-Patrick, May 2026

### Installation and Usage

It is recommended to create a clean conda environment. BIFROST enforces Python version 3.12.*, so you should also enforce this when creating your environment. Then run the following:

```
pip install "bifrost @ git+https://github.com/JQIamo/bifrost.git@main"
```

To do visualization of the examples, a little extra work is required in this version, because we implement visualization with [`py_pol`](https://py-pol.readthedocs.io/en/master/) in a somewhat janky way. First, install `bifrost` with the extra `[viz]` option as follows:

```
pip install "bifrost[viz] @ git+https://github.com/JQIamo/bifrost.git@main"
```

Then, in the ``drawings.py`` file of your local install of ``py_pol``, in the ``draw_poincare()`` method, set ``add_auxiliar = True`` by default on line 523. Then use code like I show in the examples notebook.

Another annoying caveat: one of the reasons ``py_pol`` is convenient is you can pass multiple objects around at the same time, e.g. ``E1 = J0 * E0`` can represent the several Jones vectors resulting from one Jones vector ``E0`` being multiplied by several Jones matrices ``J0``. However, this fails if there are exactly two Jones matrices in ``J0.`` So be wary of trying to make direct comparisons between two Jones matrices.

After either of these installation methods, the line

```
import bifrost as bf
```

should expose `bf.FiberLength`, `bf.FiberPaddleSet`, `bf.Rotator`, and `bf.Fiber` for use, as shown in the example notebook.

### Regime of Operation

This library models step-index silica-based germania-doped optical fibers. It includes chromatic dispersion effects. At this time, the library does not model other possible dopants nor specially engineered materials such as dispersion-compensating fiber, and it does not model other index profiles, such as graded-index fibers.

At present, BIFROST models birefringence from four mechanisms:
* Core noncircularity
* Asymmetric thermal stress (due to differing coefficients of thermal expansion between core and cladding when core is noncircular)
* Bending
* Twisting
It does **not** model birefringence due to:
* Cladding noncircularity
* Non-concentric cladding and core
* External asymmetric stress (e.g. pushing on the fiber in one direction)
* Transverse electric fields
* Axial magnetic fields
The inclusion of these mechanisms in BIFROST is a direction for future work.

Based on validation work, as well as the limits of the approximations made and the validity range of the data used in BIFROST, we believe the codebase correctly computes supported contributions to birefringence in the following regime.  
* Single-mode operation, $`V<2.405`$
* The weakly guiding regime $`n_{\text{co}}-n_{\text{cl}} \ll 1`$ (which implicitly requires weak germanium doping)
* The nearly-circular-core regime, $`e^2 \ll 1`$
* Bend radii must be much larger than the cladding radius, $`R \gg r_{\text{cl}}`$
* Temperatures 200 K $`\lesssim T \lesssim`$ 300 K, limited by our model for the thermo-optic coefficient $`dn/dT`$ of bulk germania glass. Our knowledge of the Sellmeier coefficients for germania glass is only at 297 K, but in the weakly doped regime, the temperature dependence of these coefficients is dominated by that of fused silica (which we know well)
* Telecom wavelengths 1 μm $`\lesssim \lambda \lesssim`$ 2 μm. Our expression for the thermo-optic coefficient of bulk germania glass is measured at 1550 nm, but in the weakly doped regime, the core's refractive index is dominated by that of fused silica, which we know well over a broad range of wavelengths.
We do not model the temperature dependence of the coefficients of thermal expansion or the photoelastic constants $`p_{11}`$ and $`p_{12}`$ in fused silica and germania, as the variation is small within the above parameter regime.

At this time, we do not moedl polarization-dependent loss or nonlinear scattering effects. These are directions of possible future work.

### License

Copyright (C) 2025 Patrick Banner.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

### Contact

Patrick Banner, pbanner1@swarthmore.edu
