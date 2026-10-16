.. SPDX-License-Identifier: Apache-2.0

Theory
======

This section gives the mathematical account of each stage of VANE, in the order the
pipeline applies them. It starts from the linearized turbine model that OpenFAST writes,
passes through mode identification and the Campbell diagram, and ends with the reduced,
scheduled state-space model and the Kalman filter built on it. Each page covers one
stage and can be read on its own. The equations name the functions that compute them,
so each result can be traced to the code.

Readers who need only the identified modes and the Campbell diagram can stop after
:doc:`campbell_uncertainty`. The pages on reduction, observability, LPV models and
estimation describe the state-space export built on those results. Evidence that the
implementation reproduces analytical and reference results is collected in
:doc:`../user_guide/validation`.

.. _theory-pipeline:

The pipeline in brief
---------------------

1. **Linearization and the multi-blade transform** (:doc:`linearization_mbc`). OpenFAST
   linearizes the turbine about a periodic steady operating point at several rotor
   azimuths. The multi-blade (Coleman) transform moves the blade degrees of freedom into
   the fixed frame, and the transformed matrices are averaged over the azimuths. For a
   rotor with identical blades on a symmetric support, the result is a time-invariant
   model.
2. **Mode identification** (:doc:`modal_identification`). The eigenvalues of the
   averaged model give the natural frequencies and damping ratios. The eigenvectors give
   the mode shapes. Left and right eigenvectors together give a participation measure
   that does not depend on the units of the states, and that measure names each mode
   (tower, platform, blade flap or edge, collective, regressive or progressive). Modes
   are then linked across operating points into the lines of a Campbell diagram, and
   crossings and veerings are reported for review.
3. **Floquet analysis** (:doc:`floquet`). When the rotor or its support is not
   symmetric, the transformed model stays periodic in azimuth, and averaging becomes an
   approximation. Floquet theory treats the periodic model exactly and provides the
   reference against which the averaging error is measured.
4. **Campbell diagram and uncertainty** (:doc:`campbell_uncertainty`). Mode lines are
   compared with the rotor harmonics to locate resonances. Each mode receives a
   confidence built from its numerical conditioning, its azimuth sensitivity and its
   tracking quality.
5. **Realization, reduction and observability** (:doc:`reduction_observability`). The
   model is written in real modal coordinates and reduced to the modes that matter for
   the chosen inputs and outputs, by modal selection or balanced truncation. The
   observability of the retained states from a sensor set is quantified, and sensors
   and process noise are chosen from it.
6. **LPV models and estimation** (:doc:`lpv_estimation`). Reduced models at many
   operating points are joined into a family scheduled on wind speed or rotor speed.
   A Kalman filter runs on that family, and its consistency is checked on recorded or
   simulated data.

Notation
--------

The symbols are shared across the pages. :math:`N_b` is the number of blades (three
unless stated), :math:`\Omega` the rotor speed (rad/s), :math:`\psi` the rotor azimuth,
:math:`T = 2\pi/\Omega` the rotor period, and :math:`j=\sqrt{-1}`. The state-space model
is :math:`(\mathbf A, \mathbf B, \mathbf C, \mathbf D)` with state :math:`\mathbf x`,
input :math:`\mathbf u`, and output :math:`\mathbf y`. VANE assembles the state in the
layout :math:`\mathbf x = [\,\mathbf q_2;\ \dot{\mathbf q}_2;\ \mathbf q_1\,]`
(second-order displacements, their rates, then first-order states); see
:doc:`../user_guide/coordinate_systems`. An eigenvalue is written
:math:`\lambda = \sigma + j\omega` with :math:`\sigma = \operatorname{Re}\lambda`. The
natural frequency is :math:`\omega_n = |\lambda|` and the damping ratio is
:math:`\zeta = -\sigma/|\lambda|`. Continuous-time quantities carry no subscript; a
discrete-time model with step :math:`\Delta t` is marked :math:`(\cdot)_d`.

.. toctree::
   :maxdepth: 2

   linearization_mbc
   modal_identification
   floquet
   campbell_uncertainty
   reduction_observability
   lpv_estimation
   references
