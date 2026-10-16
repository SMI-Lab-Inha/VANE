.. SPDX-License-Identifier: Apache-2.0

API reference
=============

The reference is generated from the NumPy-style docstrings in the source. Every public
module is listed once, grouped by subpackage. Modules and functions whose names start
with an underscore are private and are not shown.

Each subpackage ``__init__`` re-exports the principal public names of its modules, so
``from vane.modal import compute_modes`` and
``from vane.modal.eigensolver import compute_modes`` refer to the same object.
:mod:`vane.run` re-exports every public name of its modules. Elsewhere, a helper or
constant that a subpackage does not re-export is imported from its defining module
(for example ``from vane.modal.averaging import azimuth_mean``). The entries below are
listed under the defining module.

Package root
------------

.. automodule:: vane
   :no-members:

The package root re-exports the most common entry points, so they can be imported
as ``from vane import ...``: :class:`~vane.pipeline.ModalPipeline`,
:class:`~vane.pipeline.PipelineResult`, :func:`~vane.study.run_study`,
:func:`~vane.study.discover_operating_points`, :class:`~vane.study.StudyResult`,
:class:`~vane.study.Provenance`, and the extension protocols
:class:`~vane.plugins.Tracker` and :class:`~vane.plugins.Labeler`. Each is documented
in its own module below.

Pipeline
--------

.. automodule:: vane.pipeline
   :members:

Study and provenance
--------------------

.. automodule:: vane.study
   :members:

Extension protocols
-------------------

.. automodule:: vane.plugins
   :members:

Command-line interface
----------------------

The ``vane`` console command wraps the Python API. Run ``vane --help`` or
``vane <command> --help`` for the options of each command.

.. automodule:: vane.cli
   :members:

I/O
---

.. automodule:: vane.io
   :no-members:

.. automodule:: vane.io.lin_reader
   :members:

.. automodule:: vane.io.fst_reader
   :members:

.. automodule:: vane.io.fst_writer
   :members:

.. automodule:: vane.io.mbc_transform
   :members:

.. automodule:: vane.io.state_selection
   :members:

.. automodule:: vane.io.openfast_compat
   :members:

.. automodule:: vane.io.wamit_qtf
   :members:

Configuration
-------------

.. automodule:: vane.config
   :no-members:

.. automodule:: vane.config.dof_map
   :members:

.. automodule:: vane.config.output_map
   :members:

.. automodule:: vane.config.system_config
   :members:

Modal analysis
--------------

.. automodule:: vane.modal
   :no-members:

.. automodule:: vane.modal.eigensolver
   :members:

.. automodule:: vane.modal.mac
   :members:

.. automodule:: vane.modal.participation
   :members:

.. automodule:: vane.modal.biorthogonal
   :members:

.. automodule:: vane.modal.identifier
   :members:

.. automodule:: vane.modal.labeler
   :members:

.. automodule:: vane.modal.branch
   :members:

.. automodule:: vane.modal.uncertainty
   :members:

.. automodule:: vane.modal.confidence_calibration
   :members:

.. automodule:: vane.modal.calibration
   :members:

.. automodule:: vane.modal.floquet
   :members:

Azimuth averaging
-----------------

See :doc:`../user_guide/azimuth_averaging` for the workflow.

.. automodule:: vane.modal.averaging
   :members:

.. automodule:: vane.modal.averaging_error_estimate
   :members:

Campbell diagram
----------------

.. automodule:: vane.campbell
   :no-members:

.. automodule:: vane.campbell.builder
   :members:

.. automodule:: vane.campbell.excitation
   :members:

Driving OpenFAST
----------------

.. automodule:: vane.run
   :no-members:

The :mod:`vane.run` subpackage produces linearization data: it configures a deck for
each operating point, runs the OpenFAST executable, and collects the ``.lin`` output.
:func:`~vane.run.auto_sweep.run_operating_sweep` runs a complete sweep from an
operating curve (see :doc:`../user_guide/linearization_sweep`); the lower-level
driver, operating-point, trim and steady-state functions are public as well. Every
public name of the modules below, including helpers such as
:func:`~vane.run.auto_config.curve_power_coefficient` and
:func:`~vane.run.auto_config.rotor_radius` and constants such as
:data:`~vane.run.auto_sweep.FLOATING_TRIM_PERIODS`, can also be imported from
:mod:`vane.run` directly (``from vane.run import curve_power_coefficient``).

Linearization sweep
~~~~~~~~~~~~~~~~~~~

.. automodule:: vane.run.auto_sweep
   :members:

.. automodule:: vane.run.auto_config
   :members:

.. automodule:: vane.run.operating_points
   :members:

.. automodule:: vane.run.driver
   :members:

.. automodule:: vane.run.lin_config
   :members:

.. automodule:: vane.run.preflight
   :members:

.. automodule:: vane.run.trim_tuning
   :members:

.. automodule:: vane.run.trim_verification
   :members:

.. automodule:: vane.run.acceptance
   :members:

.. automodule:: vane.run.steady_state
   :members:

.. automodule:: vane.run.floating_auto
   :members:

.. automodule:: vane.run.radiation_fit
   :members:

.. automodule:: vane.run.runner
   :members:
   :exclude-members: TRIM_NOT_CONVERGED, MOTORING, ROTOR_OVERSPEED, DIVERGENCE, INITIALIZATION_ERROR, TIMEOUT, INCOMPLETE_LINEARIZATION, NON_PERIODIC_SWEEP, NON_FINITE_LINEARIZATION, CASE_ERROR, UNKNOWN

.. automodule:: vane.run.executables
   :members:

.. automodule:: vane.run.deck_files
   :members:

System identification
---------------------

.. automodule:: vane.sysid
   :no-members:

State space and reduced-order models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vane.sysid.state_space
   :members:

.. automodule:: vane.sysid.modal_to_ss
   :members:

.. automodule:: vane.sysid.rom
   :members:

.. automodule:: vane.sysid.balanced
   :members:

.. automodule:: vane.sysid.rom_family
   :members:

.. automodule:: vane.sysid.physical_lpv
   :members:

.. automodule:: vane.sysid.serialize
   :members:

Periodic systems
~~~~~~~~~~~~~~~~

.. automodule:: vane.sysid.periodic
   :members:

Loads and frequency response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vane.sysid.qoi
   :members:

.. automodule:: vane.sysid.spectral_transfer
   :members:

Estimation
~~~~~~~~~~

.. automodule:: vane.sysid.kalman_interface
   :members:

.. automodule:: vane.sysid.kalman_filter
   :members:

.. automodule:: vane.sysid.covariance_init
   :members:

.. automodule:: vane.sysid.process_noise
   :members:

.. automodule:: vane.sysid.augmentation
   :members:

.. automodule:: vane.sysid.observability
   :members:

.. automodule:: vane.sysid.lpv
   :members:

.. py:module:: vane.sysid.assessment

Steady-state estimator assessment: model-conditional reconstruction-error bounds.
Given the discrete model, the process-noise covariance and the measurement-noise
covariance, :func:`~vane.sysid.assessment.steady_state_estimator` solves the discrete
algebraic Riccati equation for the steady-state Kalman filter and returns the gain, the
predicted and filtered error covariances, and per-state and per-load error standard
deviations. :func:`~vane.sysid.assessment.assess_load` adds a pass or flag result
against a tolerance for each load.

Each reported standard deviation is exact for the supplied model and noise
covariances. It is not a guarantee about the physical turbine: it holds only to the
extent that the linearized model is accurate at the operating point, the supplied
covariances are the true ones, the noises are zero-mean, white, Gaussian and mutually
independent, and the sensors are calibrated, unbiased and make the pair detectable.
Validate against nonlinear time-domain simulation or measured data before relying on
it in operation.

.. autoclass:: vane.sysid.assessment.EstimatorAssessment

.. autoclass:: vane.sysid.assessment.LoadAssessment

.. autofunction:: vane.sysid.assessment.steady_state_estimator

.. autofunction:: vane.sysid.assessment.assess_load

Visualization
-------------

.. automodule:: vane.viz
   :no-members:

.. automodule:: vane.viz.campbell_plot
   :members:

.. automodule:: vane.viz.damping_plot
   :members:

.. automodule:: vane.viz.modeshape_static
   :members:

.. automodule:: vane.viz.modeshape_3d
   :members:

.. automodule:: vane.viz.modeshape_phasor
   :members:

.. automodule:: vane.viz.modeshape_fowt
   :members:

.. automodule:: vane.viz.qoi_plot
   :members:

.. automodule:: vane.viz.diagnostics
   :members:

.. automodule:: vane.viz.report
   :members:

.. automodule:: vane.viz.vtk_export
   :members:

.. py:module:: vane.viz.palette

Colour palettes for the figures. VANE's figures use the standard line colours defined
in :mod:`vane.viz` (black, red, blue, green, magenta, orange, cyan; see
:doc:`../user_guide/visualization`); this module re-exports them and also provides, as
an option for colour-vision deficiency, the Okabe-Ito qualitative palette (Okabe and
Ito, 2008) and a helper that applies it to a figure.

.. py:data:: vane.viz.palette.OKABE_ITO
   :type: tuple[str, ...]

   The eight Okabe-Ito colours as hex strings, with black last so that the chromatic
   hues cycle first.

.. autofunction:: vane.viz.palette.colorblind_colorway

.. autofunction:: vane.viz.palette.apply_colorblind_palette

.. autofunction:: vane.viz.palette.apply_standard_palette

Export
------

.. automodule:: vane.export
   :members:

Validation and benchmarking
---------------------------

.. automodule:: vane.validation
   :members:

.. automodule:: vane.benchmark
   :members:

Utilities
---------

.. automodule:: vane.utils
   :no-members:

.. automodule:: vane.utils.paths
   :members:

.. automodule:: vane.utils.fs_retry
   :members:
