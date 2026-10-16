.. SPDX-License-Identifier: Apache-2.0

Linearization restrictions
==========================

OpenFAST rejects many module configurations when ``Linearize = True``. Most of these
failures happen at initialization, after the deck has been copied and OpenFAST has been
launched; a few (the CalcSteady azimuth step) happen only after a long trim. VANE
guards against them in two places:

* :func:`~vane.run.preflight.validate_deck` reports every restriction a deck violates,
  all at once and before any OpenFAST process starts, as one
  :class:`~vane.run.preflight.DeckValidationError`. It judges the *effective*
  configuration, that is, the deck after the
  :class:`~vane.run.lin_config.LinearizationSettings` edits. The sweep checks the
  template once before any case is copied and the driver checks each configured case
  again before it runs.
* :func:`~vane.run.lin_config.configure_linearization` fixes a restriction
  automatically where a safe fix exists and a settings flag asks for it (for example
  ``wave_mod=0`` or ``trim_controller=True``).

The tables below list the linearization restrictions in the OpenFAST source: the glue
code's input validation and module-coupling setup, and each module's own linearization
checks. They cover OpenFAST 4.0.5, 4.2.1 and 5.0.0. The *VANE check* column says
whether :func:`~vane.run.preflight.validate_deck` reports the restriction; the
*Auto-fix* column names the setting or the automatic edit that resolves it.

Module switches are respected throughout. When the primary deck switches a module off
(``CompAero``, ``CompServo``, ``CompSeaSt``, ``CompHydro``, ``CompInflow`` = 0, or
``CompElast``/``CompMooring`` selecting another module), that module's input file is
inert. It is neither validated nor edited, and it may stay shared outside the case
directory. Switch dependencies are respected too: for example, HydroDyn ignores
``RdtnMod``/``ExctnMod`` and switches the second-order forces off unless ``PotMod = 1``.

Glue code and module selection
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 16 32

   * - Restriction
     - OpenFAST message (abridged)
     - VANE check
     - Auto-fix
   * - ``CompElast`` must not be 3 (Simplified-ElastoDyn)
     - "SED cannot perform linearization analysis."
     - Yes
     - None (choose ElastoDyn or BeamDyn)
   * - ``CompAero`` must not be 1 (AeroDisk)
     - "Linearization is not implemented for the AeroDisk module."
     - Yes
     - None
   * - ``CompAero`` must not be 3 (ExtLoads)
     - v5: "Module ExtLd not supported in linearization"; v4 has no ExtLoads
       linearization
     - Yes
     - None
   * - ``CompInflow`` must not be 2 (ExternalInflow)
     - "Linearization is not implemented for the ExternalInflow coupling."
     - Not needed
     - Always: ``CompInflow`` is set to 1 for a positive wind speed and 0 otherwise
   * - ``CompMooring`` must not be 2 (FEAMooring) or 4 (OrcaFlex)
     - "Linearization is not implemented for the FEAMooring mooring module." / v5:
       "Module Orca not supported in linearization"
     - Yes
     - None
   * - ``CompIce`` must be 0 (IceFloe, IceDyn)
     - "Linearization is not implemented for any of the ice loading modules."
     - Yes
     - None
   * - ``CompSoil`` must be 0 (SoilDyn, v5)
     - "SoilDyn cannot perform linearization analysis."
     - Yes
     - None
   * - ``CompSub`` must not be 2 (ExtPtfm_MCKF) — **v4 only**
     - "Linearization is not implemented for the ExtPtfm_MCKF substructure module."
     - Not checked: v5 linearizes ExtPtfm, and the deck does not name its OpenFAST
       version
     - None
   * - ``NLinTimes >= 1``; ``LinTimes`` non-negative and increasing; ``TrimCase`` 1, 2
       or 3; ``TrimTol`` positive; ``Twr_Kdmp``/``Bld_Kdmp`` non-negative;
       ``LinInputs``/``LinOutputs`` 0, 1 or 2; ``TrimGain`` positive
     - "NLinTimes must be at least 1 for linearization analysis." and similar
     - Yes (``LinearizationSettings`` construction)
     - Always: written from the settings
   * - ``LinOutJac`` needs ``LinInputs = LinOutputs = 2``
     - "LinOutJac can be used only when LinInputs=LinOutputs=2." (then ignored)
     - Yes (``full_jacobians`` validation)
     - Always: written from the settings
   * - ``WrVTK = 3`` (mode shapes) needs ``Linearize``
     - "WrVTK cannot be 3 (mode shapes) when Linearize is false."
     - Not needed
     - Always: ``WrVTK = 0``
   * - The last ``LinTimes`` entry must not exceed ``TMax``
     - "Tmax is less than the last linearization time." (then silently skipped)
     - Yes (explicit ``linearization_times``)
     - CalcSteady runs get a ``TMax`` of at least 600 s (``MIN_CALC_STEADY_TMAX``) unless
       ``sim_time`` is set
   * - CalcSteady: the rotor may turn at most ``360/NLinTimes`` degrees per time step
     - "The rotor is spinning too fast. The time step or NLinTimes is too large when
       CalcSteady=true."
     - Yes (on a configured deck: ``DT``, ``NLinTimes``, ElastoDyn ``RotSpeed``)
     - None (reduce ``DT`` or ``n_lin_times``)
   * - CalcSteady with ``GenDOF`` off or zero rotor speed silently drops the trim
       (and, at zero speed, uses a single linearization)
     - None (no error)
     - Not an error
     - ``GenDOF`` is switched on for a trimmed rotating point unless the user sets it
       explicitly; a standstill point is configured as a single snapshot
   * - MHK turbine linearization is incomplete — **v4 warning**
     - "Linearization is not fully implemented for an MHK turbine ..."
     - Not checked (warning only; removed in v5)
     - None

InflowWind
----------

.. list-table::
   :header-rows: 1
   :widths: 22 30 16 32

   * - Restriction
     - OpenFAST message (abridged)
     - VANE check
     - Auto-fix
   * - ``WindType`` 1 (steady) or 2 (uniform)
     - "InflowWind can perform linearization on only steady or uniform wind types."
     - Not needed
     - Always: ``WindType = 1``
   * - Uniform-wind files without upflow
     - "Upflow in uniform wind files must be 0 for linearization analysis in
       InflowWind."
     - Not needed
     - Always: steady wind (``WindType = 1``) does not read the file
   * - ``SensorType = 0`` (no lidar)
     - "InflowWind can not perform linearization with the lidar module enabled."
     - Yes
     - None
   * - ``VFlowAng = 0`` (no upflow)
     - "VFlowAngle (upflow) must be 0 for linearization analysis in InflowWind."
     - Yes
     - None
   * - Cubic velocity interpolation
     - Disabled by OpenFAST with a notice, not an error
     - Not needed
     - None

ServoDyn
--------

.. list-table::
   :header-rows: 1
   :widths: 22 30 16 32

   * - Restriction
     - OpenFAST message (abridged)
     - VANE check
     - Auto-fix
   * - ``PCMode = 0``
     - "PCMode must be 0 for linearization."
     - Yes (the controller-library value 5 and every other value)
     - ``trim_controller``
   * - ``VSContrl`` 0 or 1
     - "VSContrl must be 0 or 1 for linearization."
     - Yes (the controller-library value 5 and every other value)
     - ``trim_controller`` (``VSContrl = 1``)
   * - ``GenModel`` 1 or 2
     - "GenModel must be 1 or 2 for linearization."
     - Yes
     - ``trim_controller`` (another value becomes 1; unused with
       ``VSContrl = 1``)
   * - ``GenTiStr`` and ``GenTiStp`` True
     - "GenTiStr must be TRUE for linearization." / "GenTiStp must be TRUE ..."
     - Yes
     - ``trim_controller`` (timed start/stop placed relative to ``TMax``)
   * - ``HSSBrMode = 0``
     - "HSSBrMode must be 0 for linearization."
     - Yes (the controller-library value 5 and every other value)
     - ``trim_controller``
   * - ``YCMode = 0``
     - "YCMode must be 0 for linearization."
     - Yes (the controller-library value 5 and every other value)
     - ``trim_controller``
   * - ``AfCmode``/``CCmode`` not 5 (airfoil and cable control)
     - None (OpenFAST checks only the four switches above)
     - Yes: the controller-library routing (5) is reported like the others, since
       the library would run in the trim but is not part of the linear model
     - ``trim_controller`` (both become 0)
   * - Structural control (StC); airfoil and cable control other than 5
     - No linearization restriction (StC states are linearized)
     - Not needed
     - None

ElastoDyn and BeamDyn
---------------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 16 32

   * - Restriction
     - OpenFAST message (abridged)
     - VANE check
     - Auto-fix
   * - ElastoDyn
     - No linearization-specific restriction
     - Not needed
     - ``YawDOF``/``DrTrDOF`` are disabled by default for trim convergence, and the
       blade and tower flexibility DOFs are enabled so a rigid-bladed deck still
       linearizes its blade modes (VANE choices, not OpenFAST rules)
   * - BeamDyn dynamic solve
     - "Static analysis cannot be used for BeamDyn linearization."
     - Not needed
     - Always: the glue code forces the dynamic solve
   * - BeamDyn ``UsePitchAct = False`` — **v4** (v5 has no pitch actuator input)
     - "Pitch actuator model cannot currently be used for linearization in BeamDyn."
     - Yes
     - None

AeroDyn
-------

.. list-table::
   :header-rows: 1
   :widths: 22 30 16 32

   * - Restriction
     - OpenFAST message (abridged)
     - VANE check
     - Auto-fix
   * - ``Wake_Mod`` 0 or 1 (no OLAF free vortex wake)
     - "Wake_Mod must be 0 or 1 for linearization."; for OLAF also "dXdu
       linearization is not supported for FVW model"
     - Yes
     - None (OLAF has its own inputs; switching to BEMT changes the model)
   * - ``UA_Mod`` 0, 4, 5 or 6
     - "UA_Mod must be 0, 4, 5, or 6 for linearization."
     - Yes
     - ``unsteady_aero_mode``
   * - ``DBEMT_Mod`` -1, 0 or 3
     - "DBEMT_Mod must be -1 (frozen), 0 (none), or 3 ... for linearization."
     - Yes
     - ``dynamic_wake_mode``
   * - ``NacelleDrag = False``
     - "Nacelle drag cannot currently be used for linearization."
     - Yes
     - None
   * - Legacy-format decks: ``AFAeroMod = 1``, ``WakeMod`` 0 or 1
     - Same checks as above, under the pre-v4 keywords
     - Yes
     - ``unsteady_aero_mode=0`` and ``dynamic_wake_mode`` -1/0
   * - Unsteady aerodynamics (``UA_Mod`` 4, 5 or 6) must end inboard of the blade-tip
       node when BEMT (``Wake_Mod = 1``) uses ``TipLoss = True``: ``UAEndRad < 1``.
       AeroDyn switches UA off only where the node's radius fraction is *greater* than
       ``UAEndRad`` (a missing line reads as 1), and the tip-loss constant is exactly
       zero at the last node, where BEMT then fixes the inductions (axial 1,
       tangential 0)
     - None. The run converges and writes every ``.lin`` file, but the tip node's UA
       states are NaN or infinite (in tests on the IEA 22 MW rotor at 5, 6, 10 and
       20 m/s)
     - Yes (pre-flight); the driver also rejects any ``.lin`` file with a NaN or
       infinity (``non_finite_linearization``, not retried)
     - ``ua_end_radius`` (e.g. ``0.99``; the deck must carry a ``UAEndRad`` line)
   * - Valid BEMT operating point at each linearization time (defined inflow angle,
       non-zero relative velocity)
     - "Current operating point does not contain a valid value of phi." and similar
     - Not checkable before the run
     - None
   * - Tower influence, aeroacoustics, tail fin, buoyancy
     - No linearization restriction
     - Not needed
     - None

SeaState
--------

.. list-table::
   :header-rows: 1
   :widths: 22 30 16 32

   * - Restriction
     - OpenFAST message (abridged)
     - VANE check
     - Auto-fix
   * - ``WaveMod = 0`` (still water)
     - "Still water conditions must be used for linearization. Set WaveMod=0."
     - Yes
     - ``wave_mod=0``
   * - ``WaveDirMod = 0`` (not reset by ``WaveMod = 0``)
     - "No directional spreading must be used for linearization."
     - Yes
     - ``wave_mod=0``
   * - ``ConstWaveMod = 0``
     - "Constrained wave conditions cannot be used for linearization."
     - Yes
     - ``wave_mod=0``
   * - ``WvDiffQTF``/``WvSumQTF`` False
     - "Cannot use full difference-frequency 2nd-order wave kinematics for
       linearization."
     - Yes
     - ``disable_second_order_hydro``
   * - Still water needs ``WaveStMod`` 0 or 1 (v4.2.1+)
     - "WaveStMod must be 0 or 1 when WaveMod = 0."
     - Yes
     - ``wave_mod=0`` (another value becomes 0)
   * - Still water needs ``MCFD = 0``
     - "The MacCamy-Fuchs diffraction model is not compatible with WaveMod = 0 or 6."
     - Yes
     - ``wave_mod=0``
   * - Still water with ``CurrMod = 1`` needs a numeric ``CurrSSDir``
     - "CurrSSDir must not be set to 'DEFAULT' when WaveMod is set to 0."
     - Yes
     - ``wave_mod=0`` (``DEFAULT`` becomes the deck's ``WaveDir``, which keeps
       the current heading)

The still-water rows are not linearization rules in themselves. They are rejected at
initialization once still water is forced, so VANE treats them together with
``WaveMod``.

HydroDyn
--------

.. list-table::
   :header-rows: 1
   :widths: 22 30 16 32

   * - Restriction
     - OpenFAST message (abridged)
     - VANE check
     - Auto-fix
   * - ``PotMod`` 0 or 1 (no FIT)
     - "Potential-flow model cannot be set to FIT for linearization."
     - Yes
     - None
   * - ``RdtnMod`` 0 or 2 (no convolution)
     - "Cannot set wave radiation model to convolution for linearization."
     - Yes
     - ``hydro_radiation_mode``
   * - ``ExctnMod`` 0 or 2 (no DFT)
     - "Cannot set wave excitation model to DFT for linearization."
     - Yes
     - ``hydro_excitation_mode``
   * - State-space ``.ss``/``.ssexctn`` files present next to ``PotFile``
     - Fails while reading the coefficient files
     - Yes
     - ``hydro_radiation_fit`` (a missing ``.ss`` is fitted from the WAMIT ``.1`` data
       in each case copy)
   * - ``MnDrift``, ``NewmanApp``, ``DiffQTF``, ``SumQTF`` all 0
     - "Mean-drift 2nd-order forces cannot be used for linearization." and similar
     - Yes (only with ``PotMod = 1``: HydroDyn switches them off otherwise)
     - ``disable_second_order_hydro``
   * - Additional body DOFs (``NAddDOF > 0``, v5) cannot use the state-space models,
       so such a body linearizes only with ``RdtnMod = ExctnMod = 0``
     - "Nonzero NAddDOF currently cannot be used with state-space wave radiation model
       (RdtnMod=2)."
     - Yes
     - None
   * - ``RdtnDT`` must equal ``DT`` with WAMIT flow and a radiation or state-space
       excitation model
     - "RdtnDT must be equal to the glue-code DT if PotMod = 1 and using RdtnMod > 0
       or ExctnMod > 1."
     - Yes
     - ``hydro_radiation_mode``/``hydro_excitation_mode`` (``RdtnDT = DEFAULT``)
   * - State-space excitation needs no waves from file, no spreading, and no
       second-order kinematics in SeaState
     - "Given SeaState conditions cannot be used with state-space wave excitations."
     - Covered by the SeaState rows
     - Covered by the SeaState rows

Mooring and substructure
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 16 32

   * - Restriction
     - OpenFAST message (abridged)
     - VANE check
     - Auto-fix
   * - MoorDyn lines must not use VIV (line-type ``Cl > 0``) or a viscoelastic model
       (bar-separated ``EA``)
     - "Linearization cannot be used with the VIV or Viscoelastic model in MoorDyn"
     - Yes (only line types that a line uses)
     - None
   * - MAP++, SubDyn, ExtPtfm (v5)
     - No linearization restriction
     - Not needed
     - None

Remaining limits
----------------

* Restrictions that depend on the solution, such as the BEMT operating-point checks
  and CalcSteady convergence, can only be seen during the run. The sweep's retry
  ladder handles non-convergence (see :doc:`../user_guide/linearization_sweep`).
* The ExtPtfm restriction depends on the OpenFAST version, which the deck does not
  record, so it is documented here but not enforced.
* No auto-fix exists where every fix would change the physical model: OLAF, nacelle
  drag, lidar, upflow, FIT, ``NAddDOF``, the MoorDyn line models, and the
  unsupported modules. These are reported with the keyword to change.
