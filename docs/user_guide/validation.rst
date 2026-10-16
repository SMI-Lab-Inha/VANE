.. SPDX-License-Identifier: Apache-2.0

Validation
==========

The verification and validation record is maintained in the repository as
``VALIDATION.md`` and reproduced below. Its figures are generated from synthetic
systems by ``docs/validation/make_figures.py``.

.. The figures of VALIDATION.md use repository-root image paths, which render on
   GitHub but not from this page; the include therefore stops at the figure marker
   and the same figures follow as figure directives.

.. include:: ../../VALIDATION.md
   :parser: myst_parser.sphinx_
   :end-before: <!-- validation-figures

.. figure:: ../validation/figures/floquet_sideband_order.png
   :alt: Averaged and sideband-corrected eigenvalue error against modulation amplitude
   :width: 100%

   Figure 1. Platform branch of a four-state periodic rotor model built about an exact
   periodic orbit: distance of the averaged eigenvalue and of the sideband-corrected
   eigenvalue from the Floquet exponent, against the amplitude ``eps`` of the 3P and 6P
   content. The dashed lines have slopes 2 and 4. Reproduced by
   ``tests/test_modal/test_averaging.py::TestSidebandCorrection`` (AVG-3).

.. figure:: ../validation/figures/azimuth_count_aliasing.png
   :alt: Azimuth-mean eigenvalue error against azimuth count
   :width: 100%

   Figure 2. Worst relative eigenvalue error of the uniform ``N``-azimuth mean, over
   start azimuths, for a three-bladed rotor model with geometrically decaying content at
   3P, 6P, 9P and above. Left: against ``N``; counts that are multiples of three are
   first aliased at ``N``, the others at ``3N``. Right: the same errors against the
   first alias order ``lcm(N, 3)``. Reproduced by
   ``tests/test_modal/test_averaging.py::TestWeightKernel`` and
   ``tests/test_modal/test_averaging_error_estimate.py::TestAzimuthCountRule`` (AVG-1,
   AVG-4).

.. figure:: ../validation/figures/tangent_pinning_spectrum.png
   :alt: Eigenvalues before and after tangent pinning
   :width: 100%

   Figure 3. Eigenvalues of the azimuth mean of a periodic rotor model with a spurious
   azimuth stiffness, sampled at 36 uneven azimuths. Left: the reference mean and the
   mean after tangent pinning. Right: the change of each eigenvalue after tangent
   pinning and after zeroing the azimuth column; changes below 1e-16 1/s are drawn at
   1e-16 1/s. Reproduced by ``tests/test_modal/test_averaging.py::TestAzimuthTreatments``
   (AVG-2).

.. figure:: ../validation/figures/mbc3_coleman_split.png
   :alt: MBC3 Coleman split of an isotropic rotor
   :width: 100%

   Figure 4. Damped frequencies of the MBC3-averaged model of three identical blade
   oscillators (``w0`` = 2 rad/s, damping ratio 0.02) against rotor speed, with the
   analytical lines ``w_d - Omega``, ``w_d`` and ``w_d + Omega``. Reproduced by
   ``tests/test_io/test_mbc_transform.py::TestMbc3Mathematics`` (MBC-2).
