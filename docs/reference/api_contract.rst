.. SPDX-License-Identifier: Apache-2.0

API contract
============

This page states the guarantees VANE's public entry points make, independent of their
implementation. They hold for the documented public API; private helpers may change
without notice.

General guarantees
------------------

**Determinism.**
   Given the same inputs, every analysis function returns identical results. There are
   no random restarts, hidden global state, or wall-clock dependence in the numerical
   results (only the provenance timestamp records the time of the run). Stochastic
   utilities are seeded explicitly: the benchmark noise sweep (``robustness_curve``)
   defaults to ``seed=0``.

**Validation at the boundary.**
   Inputs are validated where they enter the library and rejected with a specific,
   typed exception and a clear message, rather than surfacing as an opaque lower-level
   error. Numerical results are never silently produced from malformed input.

**Typed and documented.**
   Every public function has complete type annotations and a NumPy-style docstring
   stating its parameters, returns, and the conditions under which it raises.

Entry-point guarantees
----------------------

``read_lin_file``
   Returns a ``LinFile`` only for a well-formed modern ``.lin`` file. Raises
   ``LinFileFormatError`` for an empty, truncated, legacy, or structurally invalid
   file; for a non-finite operating-point scalar or operating-point table value; for a
   missing ``A`` matrix or a state count that disagrees with the header; or for an
   ``A``/``B``/``C``/``D`` matrix whose shape disagrees with the channel-table
   dimensions. Overflow markers in matrix data become ``nan`` (with a logged warning).
   A missing file raises the usual ``FileNotFoundError``. ``find_nonfinite_lin_value`` scans a file (header,
   operating-point tables and matrices) and returns the first NaN or infinite value
   with the channel it belongs to, or ``None``.

``mbc3_transform``
   Requires one coherent operating point: a non-empty set of azimuth files with
   consistent channel layout, distinct azimuths (including the wrap-around), and a
   rotor and wind speed that do not drift. The default relative limit is 1% for each;
   an explicitly looser, finite, nonnegative tolerance is required otherwise.
   Violations, a file without an ``A`` matrix, and a rotating channel of a blade
   numbered above three raise ``ValueError``. Files without blade triplets are averaged
   without the transform. The result is the azimuth-averaged non-rotating model; with ``retain_per_azimuth=True`` it also keeps aligned per-azimuth
   ``A``/``B``/``C``/``D`` matrices and input/output operating-point vectors. This is
   an explicit memory-cost opt-in for Floquet analysis and azimuth-periodic
   state/input/output models.

``select_states``
   ``select_states(model, *, keep, drop, keep_categories, drop_categories)`` accepts a
   ``LinFile`` or an ``MBCResult`` and returns a new object of the same type with a
   subset of its states. States are kept or removed by case-insensitive regular
   expressions matched anywhere in the state description, or by ``DofCategory``;
   removal is applied after ``keep``. The displacement and velocity of a second-order
   DOF are always selected together. ``A`` keeps the selected rows and columns, ``B``
   the selected rows and ``C`` the selected columns; ``D``, the inputs and the outputs
   are unchanged. For an ``MBCResult`` the per-azimuth matrices, ``ndof2``, ``ndof1``
   and the MBC coordinate labels follow the selection; for a ``LinFile`` the module
   Jacobians are not carried over. Raises ``ValueError`` when no selector is given, a
   pattern or category matches no state, a pattern is not a valid regular expression,
   a category is unknown, or every state would be removed, and ``TypeError`` for
   another model type or a non-string pattern.

``read_wamit_diagonal_difference_qtf`` / ``newman_slow_drift_load``
   Read one real, single-heading WAMIT diagonal difference-frequency QTF and evaluate
   Newman's mean- and slow-drift approximation on an equally sampled periodic wave
   record. The period/heading diagonal-selection tolerance must be finite and
   nonnegative; infinity cannot turn off the diagonal mask. The real-FFT construction
   doubles only positive-frequency bins with a
   distinct negative-frequency partner; the unique DC and even-record Nyquist bins
   retain their physical amplitudes. Incident-wave cutoffs must be valid ordered
   numbers; only the high cutoff may be non-finite, and then only as positive infinity.
   Direct QTF construction requires a finite heading and stores immutable copies of
   the frequency and load arrays. Period-to-frequency conversion, HydroDyn
   dimensionalization, FFT frequency construction and the reconstructed load must all
   remain finite; extreme finite inputs raise ``ValueError`` rather than returning
   infinities or ``NaN``.
   Directional QTF interpolation and cropped-record edge correction are outside this
   contract.

``compute_modes`` / ``modes_from_mbc``
   Raise ``ValueError`` for a non-finite, non-square or mis-partitioned state matrix;
   ``modes_from_mbc`` also raises it when the MBC result has no averaged ``A``. Return
   conjugate-collapsed modes (positive-imaginary representatives) with frequencies,
   damping, phase-normalized shapes, condition numbers, degeneracy flags, and
   unstable/over-damped counts. Real eigenvalues and numerically split rigid-body pairs
   (below ``rigid_body_frequency_hz``) are not returned as modes.

``estimate_averaging_error``
   Takes the delivered azimuths (radians), the fixed-frame state matrices of shape
   ``(n_azimuth, n, n)`` (for example ``MBCResult.per_azimuth_a`` from
   ``mbc3_transform(..., retain_per_azimuth=True)``) and the rotor speed in rad/s, and
   returns an ``AveragingErrorEstimate`` with one ``BranchErrorEstimate`` per assessed
   branch: its sampling, quadrature and periodicity terms, its eigenvalue ``condition``
   number and ``well_conditioned`` flag (``max_condition``, default ``1e3``), and
   ``within_tolerance`` (default tolerances 0.1 % on frequency and 1 % on damping).
   ``AveragingErrorEstimate.within_tolerance`` holds when every branch is within
   tolerance and the azimuth assessment (``azimuth_assessment``, available when the
   rotor-azimuth states are given) is not ``"inconsistent-and-risky"``. A predicted
   grid whose first alias order :math:`\operatorname{lcm}(N, B)` lies above the orders
   the delivered samples resolve gives ``sampling_resolved=False`` and is not accepted.
   Inconsistent inputs or a non-positive tolerance raise ``ValueError``. The estimate
   covers the averaging error of the supplied linearization, not the accuracy of the
   linearization itself (see :doc:`../user_guide/azimuth_averaging`).

``identify_modes``
   Rejects a ``frequency_weight``, ``mac_threshold``, or ``ambiguity_margin`` outside
   ``[0, 1]`` (NaN included) and inconsistent mode-shape layouts. Returns
   deterministic tracks, each with a confidence and an ambiguity flag. Track labels use
   the same participation vote as ``label_solution``.

``label_solution``
   Labels each mode by its scale-invariant left--right participation summed per DOF
   category (displacement and velocity states of a DOF together). A mode is named after
   its largest classified category only when classified DOFs carry at least
   ``min_known_participation`` (default ``0.5``) of the participation; otherwise it is
   ``"Unidentified"``. Confidence is the named category's share of the participation.
   Raises ``ValueError`` for missing DOF descriptions or a threshold outside ``[0, 1]``
   or non-finite, and ``TypeError`` for a non-numeric threshold. A solution without
   usable left eigenvectors falls back to the raw-magnitude plurality vote.

``excitation_frequencies`` / ``find_resonances``
   Reject non-positive, fractional, non-finite, or duplicate harmonics; an empty
   harmonic set is a valid no-op. ``find_resonances`` requires a rotor-speed (rpm)
   Campbell diagram.

``ModalPipeline``
   Validates its tuning weights and harmonics at construction, and rejects an empty
   operating-point set, an unknown parameter name, a ``parameter_rtol`` that is not a
   finite number in ``[0, 1)``, or non-finite/duplicate operating-parameter values at
   ``run``. Values closer than ``parameter_rtol`` (default ``1e-4``) times the
   largest-magnitude value on the axis count as duplicates (a plateau, such as the
   constant above-rated rotor speed); for a rejected rotor-speed axis the error names
   the wind-speed axis (``parameter_name="wind_speed"``, CLI ``--parameter wind``) and
   whether it is distinct for those points, but the axis is never switched automatically. Operating
   points are sorted by the operating parameter, so input order does not affect the
   result.

``PipelineResult.state_space`` / ``state_space_from_mbc`` / ``modal_state_space``
   Export a ``StateSpace`` whose matrix shapes, name lengths, finite ``A`` and sample
   time are checked at construction (``ValueError`` otherwise).
   ``PipelineResult.state_space`` raises ``IndexError`` for an out-of-range operating
   point. A modal realization (``modal_state_space``,
   ``modal_state_space_from_solution``) re-extracts to the same frequencies and damping
   it encodes (export fidelity).

``modes_table`` / ``campbell_table`` / ``write_table``
   Reject length mismatches between a solution and its optional label/confidence/
   spread inputs, a Campbell parameter name colliding with a reserved column, and an
   unsupported file suffix. A ``.mat`` export writes string columns as cell arrays with
   missing values (``None``/``NaN``) as ``""``, so an all-``None`` ``multiblade``
   column (a sweep without blade DOFs) exports; a value MATLAB cannot store, or two
   columns mapping to the same MATLAB variable name, raises ``ValueError``. An
   ``.xlsx`` export without the optional ``openpyxl`` package also raises
   ``ValueError``.

``discover_operating_points``
   Recursively groups a directory's ``<case>.<n>.lin`` files (the case name is the
   path relative to the directory) into named
   ``DiscoveredOperatingPoint`` objects (one per ``<case>`` root, ordered by case);
   each behaves as a sequence of its ``LinFile`` objects. OpenFAST per-module
   (``LinOutMod``) files ``<case>.<n>.<Token>.lin`` are always excluded, whatever the
   alphabetic module token (``ED``, ``SEA``, ``SED``, ``ExtLd``, ``BD1``, ...).
   Raises ``FileNotFoundError`` when the directory contains no full-system ``.lin``
   files. Files are parsed with ``read_lin_file``, so a malformed file raises
   ``LinFileFormatError``. Accepts a ``str`` or ``Path``.

``run_study``
   Validates ``parameter_name`` and ``parameter_rtol`` and rejects a plateaued sweep
   axis (the same check as ``ModalPipeline.run``) before any MBC work, then runs the
   pipeline and returns a
   ``StudyResult`` carrying the ``PipelineResult`` and a ``Provenance`` record.
   Operating points are sorted deterministically by the operating parameter, and the
   recorded provenance is in that same sorted order (so it lines up with the Campbell
   result).

``Provenance`` / ``StudyResult.write_bundle``
   The provenance records the library version, a timestamp (the UTC time of the run by
   default; a caller-injected ``timestamp`` is recorded as provided), the tuning
   thresholds, the sweep-axis plateau tolerance ``parameter_rtol``, and
   pipeline-normalized harmonics, an ``environment`` block (Python, platform, and
   scientific-dependency versions), and, per operating point, the case name, full
   azimuth list, averaged rotor/wind speed, operating-parameter value, and that
   point's own SHA-256-hashed source files. ``write_bundle(output_dir, *, report=True)``
   writes ``provenance.json`` (the manifest), ``campbell.csv`` and, unless
   ``report=False``, a self-contained ``report.html`` audit report to a directory it
   creates if needed, accepting a ``str`` or ``Path``.

System identification (``vane.sysid``)
--------------------------------------

The ``vane.sysid`` layer turns identified modes into a reduced, sensor-aware, estimator-
ready model. The guarantees and failure modes for the principal entry points are below;
each raises ``ValueError`` on the malformed arguments named in its entry unless the
entry names another exception type.

**State-space export.**

``modal_state_space`` / ``StateSpace``
   ``modal_state_space`` exports a dimension-checked LTI ``StateSpace`` whose modal
   realization re-extracts to the frequencies and damping it encodes (export fidelity).
   The ``StateSpace`` dataclass rejects a non-square ``A`` and any ``B``/``C``/``D``
   whose dimensions disagree with it. (``ModalPipeline.state_space`` is documented
   under the entry-point guarantees above.)

``static_gain``
   Computes ``D - C A^-1 B`` for a continuous system, treating an absent ``D`` as zero.
   A discrete system, missing B/C, or singular A is rejected because this continuous
   equilibrium gain is then undefined under the contract.

**Reduction.**

``modal_reduction``
   Keeps the modes that drive a chosen output (by residue), optionally within a frequency
   band. ``rank_outputs`` forms an equal-weight joint objective after normalizing each
   output's residue vector, so mixed units do not determine the ranking. Returns a
   ``ModalReduction`` whose reduced system reproduces the retained modes' eigenvalues —
   except that with the default ``stabilize=True`` a retained pole with a positive real
   part is reflected into the stable half-plane (the count is reported in
   ``ModalReduction.n_stabilized``); pass ``stabilize=False`` to preserve such unstable
   poles exactly. Rejects a non-positive mode count, an unknown output, or conflicting
   single/joint ranking selections; a count larger than the number of available modes is
   capped, not rejected.

``periodic_state_space_from_mbc``
   Requires an ``MBCResult`` built with ``retain_per_azimuth=True`` and a
   ``ModalReduction`` of that result's averaged system. Every per-azimuth member is
   projected through the reduction's **same** left/right state basis, so interpolation
   never mixes independently phased modal coordinates. Periodic ``B``, ``C`` and ``D``
   variations are applied as deviations about the accepted reduced system; therefore
   a feedthrough correction introduced by ``match_dc_gain=True`` is retained, including
   when no per-azimuth ``D`` family is available. The deviations are mapped onto the
   reduction's channels by input name and by output name plus MBC coordinate, so a
   reduction whose inputs or outputs are a reordering of the MBC result's is remapped;
   a layout that is not such a permutation (different names or coordinates, ambiguous
   duplicates, or a different channel count) is rejected with ``ValueError``. When
   either side is unnamed the channels are matched by position after a count check.
   The returned ``PeriodicStateSpace`` validates a finite, strictly increasing azimuth
   grid of at least two samples in ``[0, 360)`` with one member per azimuth, and
   members with identical matrix dimensions, state names, input names, output names,
   output MBC-coordinate labels and sample time. It interpolates cyclically across the 360/0-degree seam
   and can discretize the complete grid. This is a model-construction guarantee; load
   observability and estimator accuracy must still be validated independently.

``balanced_truncation``
   Returns a ``BalancedReduction`` with an **a-priori error bound** (twice the tail
   Hankel-singular-value sum). Requires an asymptotically stable system; rejects a
   non-positive or oversized order, and an order that retains a Hankel singular value
   at rounding level (the system is not minimal at that order). Numerically indefinite
   Gramians of a lightly damped minimal system are factored through a clipped symmetric
   eigendecomposition rather than rejected. ``B`` and ``C`` are required and a
   negative ``tol`` is rejected.

``build_rom_family``
   Reduces and mode-tracks a family of operating-point systems into a consistent mode
   order. Requires **continuous-time** systems (``modal_reduction`` rejects a discrete
   system) that share the same output layout (count, names, and MBC coordinates), input
   dimension (from ``B``, or ``D`` for a feedthrough-only model) and names, and ``dt``, and
   that ``operating_points`` be a finite, strictly monotonic 1-D schedule. A point yielding
   fewer than the requested modes is rejected. These guards prevent silently interpolating
   incompatible models. With ``match_dc_gain=True``, the feedthrough correction is
   evaluated after the tracked subset is truncated, so each stored operating-point ROM
   exactly preserves the corresponding full model's equilibrium gain rather than the
   gain of an all-candidate intermediate reduction. With the default ``stabilize=True``
   a retained unstable pole is reflected into the stable half-plane in the stored
   realization, but ``RomFamily.damping_ratios`` always reports the **physical**
   (pre-reflection) damping — negative for an unstable mode — and the boolean
   ``RomFamily.stabilized`` mask marks every reflected ``(point, mode)`` entry; a
   warning is logged whenever any mode was reflected. ``stabilize=False`` stores the
   unstable poles unchanged. The builder also rejects an empty family, a length mismatch
   between ``systems`` and ``operating_points``, a member without ``C``,
   ``n_required_transfer_modes`` greater than ``n_modes``, and a ``frequency_weight``
   outside ``[0, 1]`` (default ``0.5``). Every stored modal block is phase-aligned to the previous
   point and scale-normalized (``|b_k| = |c_k|``), so members that differ only by a
   state-space similarity store identical realizations and blend without gain error.

``save_rom_family`` / ``load_rom_family``
   Revalidate the complete reusable family at both archive boundaries, including a
   finite, strictly increasing or strictly decreasing operating-point schedule.
   Nonmonotonic or repeated schedules are rejected before saving and after loading,
   rather than being passed to interpolation. Mode-tracking MAC values must remain in
   their physical interval from zero to one, preventing malformed archive confidence
   from suppressing low-confidence scheduling warnings. The ``stabilized`` reflection
   mask round-trips and must be a boolean array of the modal shape; an archive that has
   no ``stabilized`` entry loads with an all-``False`` mask. ``save_rom_family`` raises
   ``TypeError`` for an object that is not a ``RomFamily`` or has fields of the wrong
   type; ``load_rom_family`` reports every invalid archive as ``ValueError``.

``save_lpv_family`` / ``load_lpv_family`` (also exported from ``vane.export``)
   Archive an ``LPVFamily`` as a pickle-free ``.npz``: the scheduling points, the
   dropped-coupling norms and each member's matrices (explicit indexed keys, absent
   ``B``/``C``/``D`` recorded as such) are arrays; names, the scheduling variable and
   unit, the auto-added states and each member's sample time are JSON metadata stamped
   with the format tag ``vane-lpv-family``, the schema version
   ``LPV_FAMILY_SCHEMA_VERSION`` (1) and the VANE version. The round trip is exact
   (values, data types and names). Both functions validate the family: an exact
   ``LPVFamily`` of exact ``StateSpace`` members, at least one member, finite
   floating-point arrays, strictly ascending scheduling points, norms with one value
   per member or none, and one shared layout (matrix shapes, names, sample time).
   ``save_lpv_family`` raises ``TypeError`` for a wrong type and ``ValueError`` for an
   invalid family; ``load_lpv_family`` opens the file with ``allow_pickle=False`` and
   reports every invalid archive as ``ValueError``: a missing or wrong format tag, a
   missing, non-integer or different schema version (naming both versions and the
   writing VANE version), missing or unexpected entries, and invalid metadata.

``build_physical_lpv_family`` (also exported from ``vane.export``)
   Reads a directory of ``.lin`` files and assembles, per operating point, a state-space
   model whose states/inputs/outputs are the **named physical DOFs** the caller selects
   (e.g. ``"ED PtfmPitch"``, ``"GenTq"``) — the LPV plant a gain-scheduled controller
   schedules on. The matrices are **azimuth-averaged in the physical frame** (no MBC, so
   per-blade names are preserved — use ``build_rom_family`` for the multi-blade-coordinate
   representation). Names are matched against the linearization descriptions (units
   stripped): an exact match first, then a unique substring; a substring that matches
   only a DOF and its ``"First time derivative of <DOF>"`` twin selects the position
   DOF. A missing or ambiguous name, two names that resolve to the same channel, and a
   name that resolves to different channels across operating points are rejected; a
   missing or empty ``lin_dir`` raises ``FileNotFoundError``. The returned ``LPVFamily``
   is sorted ascending by the scheduling variable with the same channel layout at every
   point; a non-finite or repeated scheduling value is rejected. ``scheduling_var`` is
   ``"WindSpeed"`` (default) or ``"RotorSpeed"``. The
   schedule unit follows the raw linearization: ``WindSpeed`` is in **m/s** and
   ``RotorSpeed`` is in **rad/s** (the unit OpenFAST writes), *not* rpm — in contrast to
   the modal/Campbell path, which reports rotor speed in rpm. A controller scheduled on
   ``RotorSpeed`` must therefore index on rad/s. This is the single entry point a
   downstream controller needs — it never reads raw ``.lin`` files itself.

   Selecting a state *subset* is a truncation, and two safeguards keep it honest. With
   ``pair_derivatives=True`` (the default) every selected second-order DOF whose
   ``"First time derivative of <DOF>"`` twin (or, for a selected rate, whose position)
   exists but was not requested gets that twin added next to it — position immediately
   followed by velocity — so a request such as ``["...Platform pitch..."]`` yields the
   position and velocity pair rather than
   collapsing to ``A = [[0]]``. The added names (unit-stripped descriptions) are logged at
   INFO and recorded in ``LPVFamily.auto_added_states``; ``state_names`` is the final,
   paired list. Then, at every operating point, the relative coupling the selection
   still discards is measured as :math:`\lVert A[s,\bar s]\rVert_F /
   \lVert A[s,:]\rVert_F` over the selected states :math:`s` and, when outputs are
   selected, :math:`\lVert C[o,\bar s]\rVert_F / \lVert C[o,:]\rVert_F` over the
   selected outputs :math:`o`. Both are recorded per point
   (``dropped_coupling_a`` / ``dropped_coupling_c``; their element-wise maximum is
   ``dropped_coupling``), and if the largest over the family exceeds
   ``max_dropped_coupling`` (default ``0.05``) a ``ValueError`` names the worst point and
   the dropped states contributing most. ``max_dropped_coupling=None`` disables the check
   (the norms are still recorded); a NaN, infinite, negative, boolean or non-numeric
   tolerance is rejected. The norms are taken in the linearization's own units, so the
   measure is a screening check rather than a unit-invariant error bound. Selecting every
   state gives zero dropped coupling and leaves the result unchanged.

**Observability and sensor placement.**

``observability_gramian`` / ``finite_horizon_observability_gramian``
   The infinite- or finite-horizon observability Gramian. With a ``measurement_noise``
   covariance the source is the noise-weighted (Fisher) information ``CᵀR⁻¹C``; ``R`` must
   be finite, symmetric, and positive-definite under the shared, scale-invariant
   covariance rule (see *Covariance validity* below — a singular, numerically singular,
   indefinite, or non-symmetric ``R`` is rejected). The infinite-horizon form requires a
   stable system; the finite-horizon form is defined for marginal/unstable systems too.

``observability_scores`` / ``load_reconstructability``
   ``observability_scores`` reports per-output and per-mode observability.
   ``load_reconstructability`` reports what fraction of a target load lies in the
   sensors' observable subspace, and the a-priori reconstruction variance. The result
   (reconstructable or not) and ``noise_amplification`` are basis-independent; the value
   of a partial ``reconstructable_fraction`` (and ``observability``) is measured in the
   Euclidean metric of the state coordinates and changes under a similarity transform, so
   rank sensor sets by ``noise_amplification``. The finite-horizon continuous Gramian is
   split into sub-intervals (no overflow from fast stable modes) and a non-finite Gramian
   raises ``ValueError``. The sensor set
   **never includes the load output**: the default excludes it, and an explicit set
   containing it is rejected (a load cannot be reconstructed by measuring itself). Any
   blind component (a ``reconstructable_fraction`` below 1) gives an infinite
   ``noise_amplification``, so a smaller-is-better ranking never prefers a partly blind
   load.

``recommend_sensors`` / ``constrained_sensor_selection``
   Greedy D-optimal sensor placement, optionally **noise-weighted** (a full ``R``; a noisy
   channel ranks below a calibrated one) and over a **finite horizon** (admitting
   marginal/augmented models). ``constrained_sensor_selection`` additionally honours
   required/forbidden sensors, per-sensor costs and a budget, a count cap, and per-group
   caps, rejecting a request whose required sensors alone already violate a constraint.

**Load mapping.**

``build_qoi_map``
   Builds a quantity-of-interest (load) map, returning a ``QoiMap`` whose
   ``QoiSensitivity`` entries carry each load's state and input sensitivities, and — **only
   when a modal solution is supplied** — its per-mode modal residues (otherwise the
   residues are ``None``). The residue ranking is **modal-residue screening** of which modes
   drive a load, not a validated fatigue/design-load assessment.

Spectral propagation functions
   ``frequency_response``, ``welch_cross_spectral_density`` and
   ``propagate_cross_spectral_density`` form the frequency-domain propagation path.
   ``frequency_response`` evaluates selected named channels as
   :math:`H=C(sI-A)^{-1}B+D` on a physical-frequency grid, using :math:`s=j2\pi f`
   for a continuous model and :math:`s=\exp(j2\pi f\Delta t)` for a discrete model.
   Named selectors must be unambiguous and a discrete request cannot exceed Nyquist.
   ``welch_cross_spectral_density`` returns the complete one-sided Hermitian input CSD,
   including cross terms, with the convention :math:`S_{ij}=E[X_iX_j^*]`.
   Segment and overlap lengths must be exact integers. Frequency-response and CSD
   objects copy and freeze their arrays; they require finite, dimensionally consistent
   values and unique nonempty channel names. A CSD must be Hermitian positive
   semidefinite at every frequency, and continuous-time frequency conversion must
   remain finite.
   ``propagate_cross_spectral_density`` requires an exact frequency/name match and
   evaluates :math:`S_y=H S_u H^{\mathsf H}`; ``integrated_spectral_variance`` then
   integrates its diagonal over a declared band, interpolating the autospectra at band
   limits that fall between bins and clipping the band to the spectrum's frequency
   range (a band that does not overlap it is rejected). This is a linear, model-side
   input-error-to-QoI assessment, not nonlinear load or fatigue validation.

**Noise models, augmentation, and the estimator assessment.**

*Covariance validity.* Every entry point that consumes a covariance —
``build_kalman_model`` (``Q``, ``P0``, ``R``), ``steady_state_estimator``
(``process_noise``, ``measurement_noise``), ``discretize_process_noise``,
``augment_with_disturbance``, and the noise-weighted observability and sensor-placement
tools — applies one shared rule, so the same matrix is accepted or
rejected identically everywhere. The rule is **invariant to the unit of every channel**
(to any positive per-channel rescaling ``S M S``, and so to a global rescaling):
definiteness is judged on the correlation matrix ``K = D^-1/2 M D^-1/2`` with
``D = diag(M)``. ``R = diag(1e-12)`` (a strain gauge with one-microstrain noise) and a
mixed-unit ``R = diag(1e8, 1e-6)`` (a tension variance in N² next to an angle variance
in rad²) are as valid as ``R = diag(1)``, and an indefinite matrix is rejected however
small its entries.

* *Symmetric*: every entry finite and
  ``|M_ij - M_ji| <= 1e-9 · max(sqrt(|M_ii| |M_jj|), |M_ij|, |M_ji|)``; every term
  rescales with the pair's own units, so no channel's units affect another pair.
* *Positive semidefinite* (``Q``, ``Qd``, ``P0``, process-noise densities, disturbance
  intensities): no variance
  below ``-1e-9 · max|M_ii|`` (smaller negatives are treated as round-off of a zero
  variance); every off-diagonal entry of a non-positive-variance channel satisfies
  ``|M_ij| <= sqrt(1e-9 · |M_ii| |M_jj|)`` (so an exactly zero variance requires an
  exactly zero row, whatever the other channels' units); and ``λ_min(K) >= -1e-9`` on
  the positive-variance channels. The zero matrix is valid. Only the allowance for a
  tiny negative variance refers to the largest variance, so that one case is invariant
  under global rescaling only.
* *Positive definite* (``R``): every variance strictly positive, the Cholesky
  factorization of ``K`` succeeds, **and** ``λ_min(K) > 1e-12``. Since ``K`` has unit
  diagonal this bounds the unit-free condition number near ``1e12``: only a genuinely
  near-dependent correlation structure (for example two perfectly correlated channels)
  is rejected as numerically singular, never a disparity of units.

``measurement_noise_from_specs`` / ``discretize_process_noise`` / ``modal_process_noise``
   Build the measurement- and process-noise covariances from instrument specifications and
   modal assumptions. ``measurement_noise_from_specs`` returns
   ``R = diag(noise_std**2)`` and rejects a negative or non-finite standard deviation; a
   zero standard deviation gives a singular ``R``, which the estimator builders then
   reject.

``initialize_covariances`` / ``covariances_from_confidence``
   Build the estimator covariances ``Q``, ``R``, ``P0``. ``initialize_covariances`` makes
   **uniform** diagonal covariances from scalar noise/variance values.
   ``covariances_from_confidence`` instead inflates **each mode's** initial variance by its
   identification confidence — a less-trusted mode starts with a larger variance — so the
   identification uncertainty propagates into the filter. Choose the confidence-driven one
   to keep that propagation; the uniform one discards it.

``augment_with_disturbance`` / ``harmonic_disturbance``
   ``harmonic_disturbance`` builds a narrowband disturbance model;
   ``augment_with_disturbance`` appends it to a **continuous-time** model and returns the
   augmented system, the recovery map that reads the disturbance back out, and the
   augmented process-noise spectral density, for unknown-input estimation. Every
   prescribed oscillator
   frequency must be finite and positive, and its conversion to angular frequency must
   remain representable; invalid frequencies are rejected before a state matrix is
   allocated.

``steady_state_estimator`` / ``assess_load``
   ``steady_state_estimator`` solves the discrete-algebraic-Riccati steady-state Kalman
   filter and returns the gain and the (Joseph-form, symmetric-PSD) error covariance;
   it raises a ``ValueError`` when no stabilizing Riccati solution exists, naming
   both causes: ``(A, C)`` not detectable from the sensors, or ``(A, Q^1/2)`` not
   stabilizable (a marginal mode with no process noise).
   ``assess_load`` reports each load's steady-state **filtered** (a-posteriori,
   :math:`\sqrt{\mathbf L\mathbf P^+\mathbf L^{\mathsf T}}`) error standard deviation and,
   **only when a tolerance is supplied and met**, sets ``within_tolerance`` (a load
   checked without a tolerance is reported with ``within_tolerance = False``). The bound is exact **for the supplied model** and is
   conditional on its assumptions (accurate linear model, correct ``Qd``/``R``,
   Gaussian/stationary noise, calibration, detectability); it is not a field-validated
   accuracy guarantee.

``build_kalman_model`` / ``reduced_estimator``
   ``build_kalman_model`` assembles a Kalman-ready ``KalmanModel`` (matrices,
   covariances, and names). ``reduced_estimator`` builds a discrete ``KalmanModel`` on a
   reduced modal system (from ``modal_reduction``) for the chosen sensor rows, inflating
   each mode's initial covariance by its identification confidence (through
   ``covariances_from_confidence``); its ``process_noise`` is a hand-set discrete ``Q``,
   not a discretized density.

**Running the estimator.**

``KalmanFilter`` / ``run_filter`` / ``SteadyStateKalmanFilter``
   Run the assembled model. ``KalmanFilter`` performs the recursive predict/update on a
   **discrete** model (it rejects a continuous one), forming the gain by a Cholesky solve
   and the covariance by the Joseph form (symmetric-PSD even for ill-conditioned
   innovations). Each update accepts an ``available`` channel mask (a dropped sensor
   updates only the reporting channels) and an optional chi-square NIS gate that
   rejects an outlier rather than corrupting the estimate: a fixed ``gate`` threshold,
   or ``gate_probability`` ``p`` for the per-step quantile ``chi2.ppf(p, n_used)`` that
   scales with the reporting channels (mutually exclusive). The NIS, an acceptance flag,
   and the innovation's channel count (``n_active``, also for a gated-out step) are
   recorded. ``run_filter`` drives a whole measurement sequence into a ``FilterResult``
   (filtered and predicted means/covariances, per-step NIS, acceptance).
   ``SteadyStateKalmanFilter`` runs the same predict/correct step with the constant
   steady-state gain (from ``steady_state_estimator``), without a channel mask, NIS
   check or covariance update, for an embedded twin.

   **Control timing (zero-order hold).** Row ``k`` of ``controls`` is the input logged
   with ``measurements[k]``. Matching ``StateSpace.discretized``, the filter predicts
   ``x_k = A x_{k-1} + B u_{k-1}`` and forms the innovation against ``C x_k + D u_k``:
   the previous sample's control drives the state, the current one only the
   feedthrough. ``initial_state``/``P0`` is the prior one step *before*
   ``measurements[0]``; the first prediction uses ``initial_control`` (default
   ``controls[0]``, the input held at its first logged value; pass zeros for a plant at
   rest); ``initial_control`` without ``controls`` is rejected. ``KalmanFilter.step``, ``SteadyStateKalmanFilter.step``/``run``,
   and ``rts_smoother`` (through the recorded predictions)
   follow this convention; ``KalmanFilter.predict(u)`` is the raw transition with the
   held input ``u``.

``filter_consistency`` / ``rts_smoother`` / ``reconstruct_load``
   Audit and refine a run. ``filter_consistency`` checks the NIS sequence against its
   chi-square expectation (a mis-tuned ``Q``/``R`` is flagged, not silent); gated-out
   steps are included (their NIS is computed before the gate), so gating does not bias a
   correctly tuned filter towards "under-confident".
   ``rts_smoother`` runs the Rauch-Tung-Striebel backward pass for an offline,
   reduced-variance estimate. ``reconstruct_load`` reads a linear load functional off the
   (filtered or smoothed) state and returns its value with the propagated standard
   deviation — the virtual sensor and its error bar.

``schedule_model`` / ``run_scheduled_filter``
   Run the estimator over an operating-point sweep (LPV scheduling) on a ``RomFamily``.
   ``schedule_model`` linearly interpolates the family to one operating point (clamping and
   flagging an out-of-range point as ``extrapolated``); ``run_scheduled_filter`` re-schedules
   the plant each step as the operating point drifts, carrying the state and covariance in
   the family's mode-tracked, phase-aligned basis. Each step records the lowest tracked-mode
   **tracking MAC** (the interpolation's confidence) and the extrapolation flag, and a
   low-confidence or out-of-range schedule **logs a warning** rather than silently blending
   unrelated modes. Control timing and gating follow ``run_filter`` (zero-order hold with
   ``initial_control``; fixed ``gate`` or per-step ``gate_probability``). Requires a
   **continuous-time** family and a positive ``dt``; rejects a discrete family, non-finite
   operating point/threshold, or mismatched sequence lengths.

``save_result`` / ``load_result``
   Archive and reload an estimator result (``FilterResult`` or ``ScheduledFilterResult``)
   as a single ``.npz``. The arrays are stored as plain NumPy arrays and
   every other field (names, flags, the nested consistency report) as a JSON metadata blob,
   stamped with the result type and the producing **VANE version**. ``load_result`` reads
   with ``allow_pickle=False`` — a tampered file can carry no executable payload — and
   reconstructs the result exactly; a file that is not a VANE estimator-result archive, or
   names an unknown type, is rejected with a ``ValueError``. ``save_result`` rejects any
   other type, including a subclass, with a ``TypeError``. An estimator result is pure
   data, so loading one needs no trust boundary.

Driving OpenFAST (the linearization sweep)
------------------------------------------

Automatic configuration (``run_operating_sweep(auto_configure=True)``, the default)
   ``run_operating_sweep(fst_path, output_dir, wind_speeds, performance_csv=csv)`` is a
   complete call. Everything else the linearization needs is read from the user's own
   deck and curve (``auto_configure_sweep``); VANE computes no rotor performance. Each
   value is logged at INFO and recorded as an ``InferredValue`` (``name``, ``value``,
   ``source`` naming the file and keyword or the rule, ``detail``) in an
   ``AutoConfiguration``, which also carries the warnings raised while inferring. It
   is returned as ``OperatingSweepResult.auto_configuration``, written to
   ``output_dir/auto_configuration.json`` and copied into the driver's
   ``RunProvenance.auto_configuration``. A value passed explicitly is never replaced:
   an explicit ``template_root``, ``executable`` or ``settings`` is used as given, and
   only the turbine constants the settings leave unset (``rated_wind_speed`` and
   ``rated_generator_torque`` when ``None``, and ``trim_gain_pitch`` and ``trim_tol``
   left at their class defaults ``2e-4`` and ``1e-3``) are inferred. With
   ``auto_configure=False`` nothing is inferred: ``settings=None`` means
   ``LinearizationSettings.offshore()``, ``executable=None`` means ``"openfast"`` on
   ``PATH``, and ``template_root=None`` means the deck's own directory.

   * ``template_root`` (when not given and the deck references files outside its own
     directory): the deepest directory holding the deck and every file reached from it
     through a relative reference (``deck_files.infer_template_root``). Absolute
     references, and the references inside an absolutely referenced file, stay in place
     and do not widen it. An ``output_dir`` inside the inferred root raises
     ``ValueError`` before anything is copied.
   * The OpenFAST executable (``executables.select_openfast_executable``): the deck's
     input-format generation is detected from keyword signatures
     (``detect_deck_generation``: v5 primary-file ``ModCoupling``/``ConvTol``/... or
     ElastoDyn ``PitchDOF``; v4.2 ElastoDyn ``HubIner_Teeter``; v4.1 HydroDyn
     ``NPropSetsCyl``; v4.0 HydroDyn ``NPropSets``; v3 no ``CompSeaSt``) as a release
     range, and each candidate's release is read once from ``openfast -v`` (cached per
     file). Candidates, in order: ``executable``, the ``VANE_OPENFAST_EXECUTABLES``
     environment variable (``os.pathsep``-separated), the file named by
     ``VANE_OPENFAST_CONFIG`` (default ``~/.vane/openfast_executables.txt``, one path per
     line) and ``openfast``/``openfast_x64``/``OpenFAST`` on ``PATH``. The first whose
     release the deck accepts is used, else the first of unknown version (warned). None
     compatible raises ``OpenFastError`` before any case is copied, naming the release
     range, the keyword evidence and every executable considered. An explicit
     executable of known, incompatible release raises rather than being swapped.
   * ``rated_wind_speed``: the first curve wind at which the rotor speed is within
     0.5 % of the curve maximum (held for at least two samples) and, when the CSV has a
     ``cp`` column, ``cp * V**3`` stays within 0.5 % of its final value over the
     constant-speed region from that wind to the end of the curve (at least two
     samples; a power still rising at the last sample has no plateau and is an
     error). With a controller input file the curve's rated speed and power are
     compared with ``VS_RefSpd``/``GBRatio`` and ``VS_RtPwr`` (2 % and 5 %); a
     disagreement is a warning. A curve without a plateau leaves it unset, and a
     rotating sweep that needs it then raises ``ValueError`` before launch.
   * ``rated_generator_torque``: ``VS_RtTq`` of the controller input file (ServoDyn
     ``DLL_InFile``, ``value ! Keyword`` layout) when ServoDyn runs a control
     channel through the library; else ServoDyn ``VS_RtTq`` with ``VSContrl = 1``; else
     ``cp * rho/2 * pi * R**2 * V**3 / Omega * GBoxEff / GBRatio`` at the curve's rated
     plateau (``rho`` the deck's ``AirDens``, ``R`` ElastoDyn ``TipRad``). The ``cp``
     column is read by ``curve_power_coefficient`` with the units of
     ``rotor_performance_from_csv`` below; a wrong or ambiguous unit raises
     ``ValueError`` before launch, and a column that is absent, incomplete, negative or
     not aligned with the curve is not used.
   * ``trim_gain_pitch`` and ``trim_tol`` (when left at their class defaults 2e-4
     and 1e-3; an explicitly passed value, even an equal one, is kept):
     ``scaled_trim_gain_pitch`` and ``scaled_trim_tol`` of ElastoDyn ``TipRad``.
   * Module settings (``settings=None`` only): ``LinearizationSettings.offshore()``
     adapted to the deck. With WAMIT flow (``PotMod = 1``) a radiation model whose
     ``<PotFile>.ss`` is missing keeps ``RdtnMod = 2`` when
     ``preview_radiation_state_space`` fits the ``<PotFile>.1`` data within the
     acceptance limit: ``hydro_radiation_fit = True`` and an ``InferredValue``
     ``hydro_radiation_fit`` (source the ``.1`` file, detail the state count and
     error) are recorded. A radiation model that cannot be fitted (no or malformed
     ``.1`` data, more than one body, no water density, a fit beyond the limit), an
     excitation model whose ``.ssexctn`` is missing, or any model with
     ``NAddDOF > 0`` becomes 0 with a warning that says why; the deck's own 0, or a
     file without the line (an older HydroDyn has no ``ExctnMod``), is left
     unchanged; without WAMIT flow both are left unchanged. A modern AeroDyn deck's
     linearizable ``UA_Mod`` (4-6) is kept. When BEMT (``Wake_Mod = 1``) with
     ``TipLoss`` would keep it on the tip node (``UAEndRad`` absent or ``>= 1``),
     ``ua_end_radius = 0.99`` is written or, if the file has no ``UAEndRad`` line,
     quasi-steady aerodynamics (0) is used with a warning. Other ``UA_Mod`` values, and
     a legacy-format AeroDyn file, get the preset's 0.
   * ``sim_time`` (settle sweeps whose platform probe ran, when not set):
     ``FLOATING_TRIM_PERIODS`` (20) periods of the slowest platform mode, within
     ``MIN_CALC_STEADY_TMAX`` and the retry policy's ``max_sim_time``.

   Before any case is copied the template is checked with ``validate_deck`` under the
   effective settings (with ``validate=True``), and, for a curve sweep without
   ``settle=True``, every rotating curve point's trim configuration is checked against
   it (``FastInputError``), so a problem common to every case (no rated
   torque for an above-rated point, a missing ``UAEndRad`` line) raises once, naming
   the keyword and its remedy.

``run_operating_sweep``
   Drives OpenFAST over a list of wind speeds and returns an ``OperatingSweepResult``
   (the derived ``operating_points`` and the per-point ``SweepCase`` records). It refuses
   to guess the operating point: a rotor-performance source must be given — an operating
   curve (``curve_wind``/``curve_rotor_speed``/``curve_pitch``) or a ``performance_csv``
   (at most one of the two), and/or ``settle=True`` — and supplying none, or both a curve
   and a CSV, raises a ``ValueError`` **before any case is copied or any simulation is
   run**. A curve
   or CSV combined with ``settle=True`` is used **only as the settle runs' initial
   conditions** (ElastoDyn ``RotSpeed`` and ``BlPitch(1..3)`` of each settle copy, at the
   curve value for that wind); the settled point, not the curve, is linearized, and every
   requested wind must lie in the curve's range. After each seeded settle,
   ``compare_settled_to_curve`` compares the settled rotor speed (relative) and pitch
   (absolute) with the curve under ``settle_curve_tolerances`` (a ``TrimTolerances``,
   default 1 % and 0.5 deg); the ``SettleCurveCheck`` is attached to the point as
   ``settle_curve_check`` and recorded in ``CaseProvenance`` as
   ``settle_curve_rotor_speed_rpm``/``settle_curve_pitch_deg``/``settle_curve_status``.
   A ``"flag"`` is logged as a warning and never raises. Without a curve a settle starts
   from 0 rpm and 0 deg pitch. The platform offset is never derived from the curve;
   ``settle_initial_platform`` (one ``PlatformPosition`` or a wind-keyed mapping whose
   keys must be requested winds) optionally sets it, else the deck's position is kept.
   ``settle_initial_platform`` without ``settle=True``, or ``settle_curve_tolerances``
   without a seeded settle, raises ``ValueError``. The settle copies keep the template
   deck's ElastoDyn DOF flags (``structural_dofs`` applies to the linearization cases
   only); the template's values of the flags the linearization sets are returned as
   ``OperatingSweepResult.settle_structural_dofs`` and a difference is logged. One
   settle point never aborts the sweep. Every wind is settled (with
   ``require_settled=False`` passed to ``solve_steady_state``), and each outcome is then
   classified. A settle whose criterion was not met or could not be assessed is
   ``"not_settled"``. Under ``require_settled=True`` (the default) it is not linearized,
   except that a prescribed-rotor settle (no loadable controller library) of a rotating
   ``CalcSteady`` point is linearized, since the trim finds the equilibrium itself;
   under ``False`` it is linearized at its last mean. A settle that raised
   ``SteadyStateError`` (OpenFAST failed, unreadable output) is ``"failed"`` and is not
   linearized. Any other exception is ``"error"``: it is logged with its traceback, and
   once every other wind has been settled and linearized it is re-raised as
   ``SweepSettleError`` (a ``SteadyStateError``) chained to the first such exception,
   with the complete ``OperatingSweepResult`` attached as ``.result``. Every such point
   is listed, in wind order, in ``OperatingSweepResult.settle_failures`` as a
   ``SettleFailure`` with ``wind_speed``, ``reason``, ``message``, ``linearized``, the
   last-mean ``operating_point`` and the ``convergence`` evidence (final RMS, every
   attempt's horizon and window, the criterion, and any two-stage confirmation).
   ``to_dict()`` gives a strict-JSON form. A summary warning is logged.
   ``operating_points`` and ``cases`` hold only the linearized points.
   ``strict_settle=True`` raises ``SweepSettleError`` (with ``.result``) at the end
   when any point is in ``settle_failures``. A settle sub-file that resolves outside
   ``template_root`` is a configuration error shared by every wind, so it raises
   before any settle case is copied. It validates the request at
   the boundary (a missing template deck, an empty/non-finite/negative/duplicate wind-speed
   list, a partial or mismatched curve, a ``template_root`` that does not contain the deck,
   and a positive wind point with no usable InflowWind file each raise one clear error up
   front). The worker count defaults to the CPU count minus ``reserve_cpus`` (default
   1), at least one (``default_max_workers``). Unless ``sim_time`` is set, a
   ``CalcSteady`` trim runs with ``TMax`` of at least ``MIN_CALC_STEADY_TMAX``
   (``vane.run.lin_config``, 600 s), since template decks often carry
   a placeholder ``TMax`` shorter than the trim; OpenFAST stops once the trim converges.
   Deck keywords are matched case-insensitively, as in OpenFAST. A non-converging
   ``CalcSteady`` trim is retried on the ``LinearizationRetryPolicy`` ladder, which
   classifies each failed run (``classify_openfast_output`` for trim not converged,
   motoring, rotor overspeed, divergence, initialization error and unknown; the driver
   adds timeout, incomplete linearization, and a non-periodic sweep whose rotor speed
   varies across the azimuth files by more than MBC3 accepts) and picks the next
   settings from that class. A trim that ran out of time gets a longer ``TMax``, a
   looser ``TrimTol`` (not beyond 1e-2 unless it already was) and a lower pitch gain. A
   non-periodic sweep gets ``TrimTol`` tightened back to its base value and a longer
   ``TMax``. An unstable trim (motoring, overspeed, divergence) gets lower gains;
   overspeed and divergence also reduce the trim damping, and any of the three removes
   it after a failure within ``early_failure_time`` (1 s). After an unstable failure,
   the last of the ``max_attempts`` (4) attempts trims an at- or above-rated point by
   generator torque at the curve pitch. An initialization error is never retried. A run that otherwise converged is still
   rejected when any full-system ``.lin`` file holds a NaN or infinity
   (``non_finite_linearization``, naming the first offending variable); that is a
   modelling degeneracy, so it is not retried either. Every attempt's full OpenFAST
   output is written to ``<deck stem>.attempt<k>.log`` beside the case's primary deck, and its failure
   mode, settings and log path are recorded in ``SweepCase.attempt_history`` and the
   case provenance (``CaseProvenance.attempt_history``/``failure_mode``/``sim_time``).
   A point that never converges is recorded ``converged=False`` with its
   ``attempts``/``n_lin`` rather than silently accepted. One linearization case never
   discards the others: an *unexpected* exception while a case is generated, run or
   collected (a file-system error, an unreadable sub-file) ends that point's retry
   ladder and is recorded as a failed ``SweepCase`` with ``converged=False``,
   ``failure_mode == "case_error"`` (``OpenFastFailureMode.CASE_ERROR``, never
   retried), return code ``-1`` and ``n_lin == 0``. The final ``LinearizationAttempt``
   carries ``"case_error: <type>: <message>"`` as its ``reason`` (also in the
   provenance), its attempt log holds the traceback, and the earlier attempts' records
   are kept. ``LinearizationDriver.run_sweep(strict=True)`` and
   ``run_operating_sweep(strict_cases=True)`` instead raise ``SweepCaseError`` once
   every other case has finished, chained to the first exception, with every case
   attached as ``.cases`` (and, from ``run_operating_sweep``, the complete
   ``OperatingSweepResult`` as ``.result``). ``run_case`` called directly still
   propagates the exception. Boundary errors are still raised: the argument checks
   before any case is copied, and a ``DeckValidationError`` from the pre-flight guard
   (an input error found before that case's OpenFAST run). On Windows the case-file
   operations (the template copy and hard links, the deck edits' atomic replace, the
   case-directory removal before a retry, the settle copy edits, stale-output removal
   and the JSON provenance write) retry an access-denied, sharing-violation or
   lock-violation error (``winerror`` 5, 32 or 33), with up to five tries in all (four
   retries) and a backoff doubling from 0.05 s (0.75 s of waiting in all), before
   raising; antivirus scans, the
   search indexer and a recently exited OpenFAST briefly hold such handles. Any other
   error, and every POSIX error, is raised at once (``vane.utils.fs_retry``).
   Setting ``LinearizationDriver(acceptance_policy=LinearizationAcceptancePolicy())``
   (a driver field, not a ``run_operating_sweep`` argument) adds an independent
   physical check after that numerical criterion: VANE estimates the azimuth mean of
   the ElastoDyn platform translational/angular rate states and rejects a secular
   residual above the configured limits (defaults 1e-3 m/s and 1e-4 rad/s). Rotating
   points require 12 unique azimuths by default, at least four under any explicit
   override, no duplicate azimuth, and a layout whose harmonic-cancelling mean weights
   (the minimum-norm weights that cancel the 1P--3P harmonics at the actual, possibly
   non-uniform, azimuths) exist and inflate the variance of an equal-weight mean by at
   most a factor of two. A physical-residual rejection stops without entering the normal
   retry ladder, because that ladder loosens ``TrimTol`` and would move in the wrong
   direction. The complete residual audit is retained in the case provenance.
   ``LinearizationDriver.run_sweep`` names each case directory after its operating
   point with trailing dots dropped and rejects, before any case is copied or run, two
   points whose names coincide case-insensitively (the default Windows and macOS
   filesystems would give them one directory, and the second case would delete the
   first one's results) as well as a reserved Windows device name (``CON``, ``NUL``,
   ``COM1`` ...).
   Every case (and every settle run) is also checked before it is configured: the
   primary deck and every sub-file the run will edit must resolve inside the case copy
   (``require_inputs_within``), because a sub-file referenced outside it
   (``../ElastoDyn.dat``) would be edited where it lives, shared by every case and the
   template. A linearization always edits ElastoDyn and InflowWind, and SeaState,
   HydroDyn, AeroDyn or ServoDyn only when the settings change them; a settle run edits
   ElastoDyn, InflowWind and SeaState, ServoDyn when a controller library is written
   into the settle copy (given or found automatically), and HydroDyn when a settle
   platform damping is added to its ``AddBLin``. Files that are only read may stay shared. Set
   ``template_root`` to a directory that contains every edited input.
   ``configure_linearization`` applies the same check when called with ``confine_to=``.
   What a case receives from ``template_root`` is selected by ``template_copy``
   (``LinearizationDriver`` field and ``run_operating_sweep`` argument; settle cases
   follow it too). The default, ``"referenced"``, copies only the files the deck
   references (``deck_file_closure``): starting from the primary ``.fst``, every
   referenced text input is read and every value token that looks like a path is resolved
   relative to the referencing file's directory and to the deck's directory, and kept when
   it names an existing file under ``template_root``; a root-name keyword (``PotFile``,
   ``WvKinFile``, any ``...Root``) brings every ``<root>.<ext>`` input beside it, and a
   ``.dll``/``.so``/``.dylib`` reference brings that library's platform and bitness
   variants (``_x64``, ``_win64``, ...) but no other sidecar unless referenced. Files keep
   their paths relative to ``template_root``. Unknown keywords, ``"unused"``,
   ``"default"``, numbers and missing files are ignored rather than fatal. A relative
   reference that resolves outside ``template_root`` or an unreadable input raises
   ``DeckReferenceError`` inside the walk, and ``copy_template_files`` then logs a warning
   and copies the whole tree, exactly as ``template_copy="tree"`` does; an absolute
   reference outside the root is left in place (it still resolves from the case) and
   listed in ``DeckFileClosure.external``. Either way the edited-input confinement above
   still applies, inputs are hard-linked when ``LinearizationDriver.link_inputs`` is set
   (the default; a byte copy where linking is not supported), and OpenFAST output files
   are byte-copied, never linked.

``operating_curve_from_csv`` / ``operating_curve_from_performance``
   Parse a rotor-performance schedule into the ``(curve_wind, curve_rotor_speed,
   curve_pitch)`` arrays the sweep consumes, sorted by ascending wind. The CSV requires a
   header row naming three required columns — ``wind_speed`` (m/s), ``rotor_speed_rpm`` (rpm),
   ``blade_pitch_deg`` (deg) — matched tolerantly (case/whitespace/separator
   insensitive); other columns are ignored, units included. A unit in ``[...]`` or
   ``(...)`` on a required column is parsed case/whitespace-insensitively: no unit or the
   canonical unit (among others ``m/s``/``mps``, ``rpm``/``rev/min``,
   ``deg``/``degree(s)``) is read
   as is; ``km/h`` (÷3.6), ``rad/s`` (×60/2π), and ``rad`` (×180/π) are converted and
   logged at INFO; any other unit — or a converting unit on a header whose name states
   the canonical unit, e.g. ``rotor_speed_rpm [rad/s]`` — raises ``ValueError`` naming the
   column, the unit, and the accepted units. Both paths reject a non-numeric/NaN/infinite
   value, a negative wind or rotor speed (a negative collective blade pitch is allowed)
   and duplicate wind speeds, and require **at least two** operating points (the
   schedule is interpolated); ``operating_curve_from_performance`` called without a
   ``performance_table`` raises ``RuntimeError``. Units are read from the header only; a
   units row raises ``ValueError`` as a non-numeric value, with a hint. VANE
   computes no rotor performance itself and has no aerodynamic-performance dependency.

``rotor_performance_from_csv`` / ``curve_power_coefficient``
   Read the optional power- and thrust-coefficient columns (``cp``,
   ``power_coefficient``, ``RtAeroCp``, ``RtFldCp``; ``ct``, ``thrust_coefficient``,
   ``RtAeroCt``, ``RtFldCt``) and return them as fractions. More than one column for a
   coefficient raises ``ValueError``. The header bracket declares the unit: none,
   ``-``, ``1`` or ``fraction`` is a fraction; ``%``, ``percent`` or ``pct`` a
   percentage, divided by 100 and logged at INFO; any other unit raises ``ValueError``
   listing the accepted forms. A converted value above the physical bound (the Betz
   limit 16/27 for ``cp``, 2 for ``ct``) raises ``ValueError`` (a percentage without
   its unit), as does a column declared in percent whose non-zero values are all at
   most 1 in magnitude. Blank or non-numeric cells are ``NaN`` in
   ``rotor_performance_from_csv``; ``curve_power_coefficient`` returns ``None`` when a
   ``cp`` value is missing or negative, the rows do not match the curve winds, or the
   file has no ``cp`` column.

``LinearizationSettings`` (``structural_dofs``, ``offshore``)
   Self-validates at construction (rejecting non-physical counts/tolerances and module
   switches that do not select a linearizable model). ``structural_dofs`` maps ElastoDyn
   DOF flags; its default enables the blade and tower flexibility DOFs (``FlapDOF1``,
   ``FlapDOF2``, ``EdgeDOF``, ``TwFADOF1``, ``TwFADOF2``, ``TwSSDOF1``, ``TwSSDOF2``)
   and disables ``YawDOF`` and ``DrTrDOF`` (dropping the spurious free nacelle-yaw
   rigid-body mode and, for ``CalcSteady`` trim convergence, the drivetrain-torsion
   mode; pass ``{"DrTrDOF": True}`` to keep it), and a supplied mapping is merged on top
   of those defaults. ``auto_configure_sweep`` logs a warning naming each flexibility
   DOF that the deck has off and the settings turn on, and records it in
   ``AutoConfiguration.warnings``. A flag listed but absent from the ElastoDyn deck is
   an error (``FastInputError``); a default flag the deck lacks is skipped.
   ``full_jacobians`` requires ``lin_inputs=2`` and ``lin_outputs=2`` (``ValueError``
   at construction). When ``full_jacobians`` or ``module_linearizations`` is enabled,
   the primary deck must contain ``LinOutJac`` or ``LinOutMod``, respectively;
   otherwise ``configure_linearization`` raises ``FastInputError`` before any file is
   written.
   ``unsteady_aero_mode``/``dynamic_wake_mode`` also configure a legacy-format AeroDyn
   deck (``AFAeroMod``/``UAMod``/``WakeMod`` keywords): quasi-steady aero sets
   ``AFAeroMod = 1``, and a frozen (``-1``) or no (``0``) dynamic wake turns a
   dynamic-BEMT ``WakeMod = 2`` into plain BEMT with ``FrozenWake`` on or off.
   ``validate_deck`` checks those legacy keywords and reports a legacy deck that would
   still linearize with unsteady airfoils, a dynamic or free-vortex wake, or a model the
   legacy format cannot express (``unsteady_aero_mode`` 4--6, ``dynamic_wake_mode=3``).
   ``ua_end_radius`` writes AeroDyn's ``UAEndRad`` (a number in ``(0, 1]``); an enabled
   modern-format AeroDyn file without that line raises ``FastInputError`` before any
   file is written, and a legacy-format file receives the value only when it already
   has the line.
   ``hydro_radiation_fit`` (a bool, only with ``hydro_radiation_mode = 2``) makes the
   ``LinearizationDriver`` call ``ensure_radiation_state_space(deck,
   confine_to=case_dir)`` on every configured case copy, before it is validated and
   launched, so a missing ``<PotFile>.ss`` is fitted into the copy and never next to
   the template's data. A fit whose maximum relative error exceeds 0.2 writes nothing,
   and the driver raises ``RadiationFitError`` (a ``ValueError``), as it does for more
   than one WAMIT body, a missing or unfittable ``.1`` file, or a ``PotFile`` outside
   the case copy; ``run_sweep`` records it as that case's ``failure_mode="case_error"``
   and ``run_case`` raises it. The generated file's
   ``RadiationFitRecord`` is ``SweepCase.radiation_fit`` and, as ``to_dict()``,
   ``CaseProvenance.radiation_fit``. ``validate_deck`` accepts a missing ``.ss`` under
   this setting when ``<PotFile>.1`` exists.
   With ``calc_steady=False``, ``linearization_times`` may specify a complete set of
   absolute sampling times for an independently settled rotating point. The sequence
   must be finite, non-negative, strictly increasing, equal in length to
   ``n_lin_times``, and no later than ``sim_time`` (or the deck's own ``TMax`` when
   ``sim_time`` is unset). The driver accepts the run only when every requested
   full-system ``.lin`` file exists. VANE does not itself check that the explicit times
   are at equilibrium; set ``LinearizationDriver(acceptance_policy=...)`` to apply the
   physical residual audit.

``remap_hydrodyn_state_space_wave_input``
   Applies the known HydroDyn state-space wave-input label correction only when its
   complete matrix, operating-point and channel signature is present in both the
   averaged model and every retained azimuth-resolved B/D member. ``zero_atol`` and
   ``nonzero_atol`` must both be finite and non-negative; ``NaN`` or infinity cannot
   weaken the signature checks. A missing signature raises ``ValueError``; on success
   only the two input descriptions and their MBC-coordinate labels are swapped, and no
   coefficient changes.

``apply_hydrodyn_wave_dof_signs``
   Applies an independently established HydroDyn wave-DOF sign convention only when
   the averaged and every retained azimuth-resolved wave column excite the same
   contiguous, HydroDyn-only state block whose size matches the caller's
   ``states_per_dof`` (the ``.ssexctn`` header counts, see
   ``read_state_space_excitation_metadata``); each ``dof_signs`` entry must be +1 or -1,
   and any mismatch raises ``ValueError``.

``solve_steady_state``
   Runs the fully-coupled controller at a constant wind, discards the transient, and
   returns the settled ``OperatingPoint`` (rotor speed, pitch, and the floating platform
   equilibrium). It judges settling by the platform-velocity RMS over the steady window
   (the working copy's ElastoDyn ``OutList`` gains any missing ``PtfmTVxi``..``PtfmRVzi``
   velocity, ``PtfmSurge``..``PtfmYaw`` position, ``RotSpeed`` and ``BldPitch1``
   channels, appended as one quoted line before the first ``OutList`` block's ``END``);
   with ``require_settled=True`` an unsettled or unverifiable point is an error rather than
   a silently-accepted ``TMax`` guess. Every run, including each adaptive retry at a
   longer ``TMax``, is judged over a trailing window of fixed length
   ``settle_time - discard_time`` ending at that run's ``TMax`` (``TStart`` is set to
   the window start), and the returned rotor speed, pitch and platform position are the
   means over that same window (see the period alignment below), so a longer run discards the start-up transient rather
   than diluting it. The output step ``DT_Out`` is the whole multiple of ``DT`` nearest
   one second (one second when ``DT`` is missing, non-finite or not positive), as
   OpenFAST requires. A NaN or
   infinite sample in the window, or an overflowed (``*****``) output field, means the
   run diverged and raises ``SteadyStateError``. The window bounds are recorded as ``window_start``/``window_end`` on
   ``SettleConvergence`` and on each ``SettleAttempt``. ``SettleConvergence.attempts``
   preserves the ordered horizon and RMS result from every adaptive attempt; a rejected
   required settle exposes that same evidence as ``SteadyStateError.convergence``. An
   optional ``platform_position`` warm-start writes only the supplied ElastoDyn initial
   DOFs; it does not alter the controller or physical-module settings. The settle run
   and the linearization share one inflow convention: both write the InflowWind
   power-law shear exponent ``PLExp`` from ``wind_shear_exponent`` (default ``0.0``,
   steady uniform inflow; ``None`` keeps the deck's value), and ``run_operating_sweep``
   passes ``LinearizationSettings.wind_shear_exponent`` to its settle runs, so a settled
   point is never linearized in a different (sheared) wind. The value is recorded with
   the other settings in the run provenance. The settle run also shares the
   linearization's still water: SeaState ``WaveMod = 0`` plus the same neutralization of
   the options OpenFAST rejects in still water (``WaveStMod`` 2/3 reset to 0, ``MCFD``
   reset to 0, a ``DEFAULT`` ``CurrSSDir`` replaced by ``WaveDir``, ``WaveDirMod`` and
   ``ConstWaveMod`` reset to 0).
   Unlike the linearization, the settle run keeps the deck's Bladed-style controller
   library active. When ServoDyn routes any control through it (``PCMode``,
   ``VSContrl``, ``HSSBrMode``, ``YCMode``, ``AfCmode`` or ``CCmode`` = 5),
   ``check_controller_library`` verifies before OpenFAST launches (and before any deck
   is edited) that ``DLL_FileName``, resolved relative to the ServoDyn file, exists and
   loads on this host (checked by loading it in a child process), and otherwise raises
   ``SteadyStateError`` ("controller library not found: ...; pass a controller library
   path (controller_library=...) built for this platform", or "... cannot be loaded on
   this host ..."). This is the usual case for a published deck naming a Linux
   ``libdiscon.so`` on Windows. ``controller_library=`` (on ``solve_steady_state`` and
   ``run_operating_sweep``) supplies a library built for the host; it must exist
   (``FileNotFoundError`` otherwise) and load on this host (``SteadyStateError``
   otherwise). When ServoDyn routes control through the library it is written, as an
   absolute path, into ``DLL_FileName`` of the settle copy only, never the template;
   otherwise it is ignored with a warning. The settle confinement set is
   ``SETTLE_EDITED_SUBFILE_KEYS`` (``EDFile``, ``InflowFile``, ``SeaStFile``,
   ``ServoFile``); ``ServoFile`` is dropped from it without an override, so with an
   override the ServoDyn file must resolve inside the settle case. ``run_operating_sweep`` resolves the library once,
   before any case is copied, through ``plan_floating_settle`` (below), and rejects
   ``controller_library`` without ``settle=True``. ``prescribed_rotor=True`` settles
   without a controller: the settle copy gets ``CompServo = 0`` and ElastoDyn
   ``GenDOF = DrTrDOF = False``, the library is neither checked nor written, the rotor
   turns at ``rotor_speed_rpm`` with the blades at ``blade_pitch_deg``, the returned
   rotor speed and pitch are exactly those values, and ``SettleConvergence.prescribed_rotor``
   is ``True``; combining it with ``controller_library`` raises ``ValueError``.
   ``criterion`` selects how a run is judged settled: ``"auto"`` (default), ``"velocity"``
   (every platform-velocity RMS within ``velocity_tol``/``angular_velocity_tol``) or
   ``"mean_drift"`` (the means of the platform position, ``RotSpeed`` and
   ``BldPitch1`` over the first and second halves of the analysis window differ by no
   more than ``equilibrium_tolerances``, an ``EquilibriumTolerances``: 0.1 m, 0.05 deg,
   0.5 % of ``max(rotor speed, 1 rpm)``, 0.1 deg by default). ``"mean_drift"`` accepts a
   stationary oscillation about a steady mean and returns that mean; a run it accepts
   while still above the velocity tolerances is logged as a warning naming the
   oscillation. Every ``SettleAttempt`` and the ``SettleConvergence`` record the
   ``criterion``, the half-window ``mean_drift`` and the ``oscillation_period`` (dominant
   period of the most active platform-velocity channel, ``None`` when there is too
   little data or no motion), whichever criterion is used, and the oscillation's
   ``residual_damping_ratio`` over the analysis window. ``"auto"`` judges each run by the
   velocity criterion and switches that run to the mean-drift criterion only when the
   velocity criterion fails, the run was assessed, at least one platform position
   channel was output, and the oscillation is sustained:
   ``|residual_damping_ratio| < equilibrium_tolerances.residual_damping_ratio`` (a
   decaying transient keeps the velocity criterion, a growing oscillation never
   switches a single run, and fewer than two whole cycles never switch). The recorded ``criterion``
   (convergence and per attempt) is always ``"velocity"`` or ``"mean_drift"``, the one
   that judged the run; ``requested_criterion`` holds the argument and
   ``criterion_switched`` whether ``"auto"`` switched.
   ``platform_damping`` (a ``SettlePlatformDamping``: non-negative, finite N/(m/s) and
   N m/(rad/s) per platform DOF, at least one positive; ``from_critical_fraction``
   sizes it as ``4 pi zeta m / T``) makes the settle two-stage. Before launch the deck
   must run HydroDyn (``CompHydro`` not 0, an existing ``HydroFile``) whose ``AddBLin``
   block is ``6 NBody`` numeric rows (``6 NBody`` columns for ``NBodyMod = 1``, else 6)
   starting at the ``AddBLin`` label and followed by ``AddBQuad``; otherwise
   ``ValueError``. Stage 1 adds the damping to the diagonal of body 1's block (other
   tokens and labels verbatim, via an atomic replace that never writes through a hard
   link) and settles with the configured schedule, outputting ``TTDspFA``/``TTDspSS``
   as well. The deck's original HydroDyn bytes are then restored, also when stage 1
   fails. Stage 2 is one undamped run of ``confirm_time`` seconds judged
   over ``[confirm_time / 4, confirm_time]``, started from the stage-1 means of
   ``RotSpeed``, ``BldPitch1``, the platform DOFs and the ``TTDspFA``/``TTDspSS`` (and,
   when the deck outputs them, ``OoPDefl1``/``IPDefl1``) deflections. ``confirm_time``
   ``None`` (the default) is ``ceil(8 T / 0.75)`` s -- eight periods ``T`` of the
   slowest platform mode in the analysis window -- with ``T`` the ``platform_period``
   argument (positive and finite, else ``ValueError``) or else the damped stage's
   ``oscillation_period``, floored at 1200 s (also the length without any period) and
   capped at ``max_settle_time`` when given, else 3600 s; an explicit value is used as
   given. Every run's returned means (every settle run's, not only stage 2's) and
   both halves of the mean-drift comparison are averaged over the trailing whole
   number of that run's ``oscillation_period`` (aligned to the window end; the plain
   window mean when no period is detected or less than one fits). Its
   ``residual_damping_ratio`` is estimated from the platform-velocity channel with the
   largest de-meaned RMS relative to its tolerance: the slope of the log of the RMS in
   successive whole periods, counted back from the end, divided by ``-2 pi``. It is
   ``None`` when no period was detected or fewer than two whole cycles are available. ``stable`` holds when stage 2 met the criterion by itself, or, when stage 2
   was judged by the velocity criterion, when ``residual_damping_ratio`` is at least
   ``equilibrium_tolerances.residual_damping_ratio`` (default 0.005). When stage 1
   converged and stage 2 is decaying in that sense but a mean moved beyond
   ``equilibrium_tolerances``, stage 2 is rerun once at twice its length (capped at
   ``max_settle_time`` when given; no rerun when the cap leaves it no longer) and the
   rerun is judged instead (``extended``). With ``criterion="auto"``, when stage 1
   converged, a stage 2 that is not ``stable`` but whose means all stay within ``equilibrium_tolerances`` of
   stage 1, whose ``residual_damping_ratio`` is known (sustained *or growing*: there
   is no lower bound), and whose measured platform-velocity RMS values (at least one
   measured) are all within
   ``equilibrium_tolerances.oscillation_velocity_rms`` (default 0.2 m/s and deg/s) is
   accepted about the confirmed equilibrium: ``sustained_oscillation`` is set on the
   confirmation and the convergence, and a warning is logged. Linear damping is zero
   at zero platform velocity, so the stage-1 equilibrium is the undamped model's,
   stable or not. When the accepted residual is below
   ``-equilibrium_tolerances.growing_damping_ratio`` (default 0.02, a classification
   threshold only), the read-only ``SettleConfirmation.growing_oscillation`` and
   ``SettleConvergence.growing_oscillation`` are true and a distinct warning says the
   equilibrium is unstable under the deck's controller and is linearized without it
   (the trim replaces the controller, so that closed-loop instability is not part of
   the linear model). ``SettleConvergence.to_dict()`` and
   ``SettleConfirmation.to_dict()`` add ``growing_oscillation`` to the ``asdict``
   fields; ``SettleFailure.to_dict()`` and the ``CaseProvenance`` record carry it.
   Under the ``"velocity"`` and ``"mean_drift"`` criteria this acceptance does not
   apply: they never accept a non-decaying stage 2 that does not meet them.
   ``converged`` is true only when stage 1 met the criterion and either stage 2 is
   ``stable`` with no stage-2 mean moved from stage 1 beyond ``equilibrium_tolerances``
   (a stage 2 that still oscillates, that is, one accepted by the mean-drift criterion
   rather than by velocity or decay, must also keep every platform-velocity RMS within
   ``equilibrium_tolerances.oscillation_velocity_rms``), or the ``"auto"`` acceptance
   above applies (``sustained_oscillation``). A converged point
   returns the stage-1 means; an unconverged one returns the (last) stage-2 means. The
   evidence is a ``SettleConfirmation`` on ``SettleConvergence.confirmation``: the
   damping, the stage-1 result (``damped_converged``), RMS, ``TMax`` and means, the stage-2 means, the
   ``shift`` and the flagged quantities, ``confirm_converged``,
   ``residual_damping_ratio``, ``stable``, ``confirmed``, the judged ``confirm_time``
   and its ``confirm_time_source`` (``"user"``, ``"platform_period"``,
   ``"damped_stage"`` or ``"fallback"``), ``extended``, ``sustained_oscillation`` and
   the applied ``tolerances``.
   ``attempts`` lists every run in order, each tagged ``stage="damped"`` or
   ``"confirm"``. A failed confirmation is logged or, with ``require_settled``,
   raised, with its reason: the damped stage did not settle, the equilibrium moved,
   or the undamped run is still moving and not decaying (described as growing when
   its residual is below ``-growing_damping_ratio``), together with what blocked
   its acceptance: a criterion other than ``"auto"``, the moved equilibrium, the
   unsettled damped stage, fewer than two analysed cycles, or a velocity RMS above
   ``oscillation_velocity_rms``. Growth alone is never the reason.
   ``PlatformProbe.slowest_period`` is the largest positive finite probe period
   (``None`` without one). ``run_operating_sweep`` exposes ``settle_criterion``
   (default ``"auto"``), ``settle_platform_damping`` (default ``None``: for a floating
   deck, sized from the standstill platform probe unless ``settle_auto_damping=False``),
   ``settle_confirm_time`` (default
   ``None``, automatic) and ``settle_equilibrium_tolerances``, and passes a probe's
   ``slowest_period`` as ``platform_period``. All four are rejected
   without ``settle=True`` (a ``settle_criterion`` other than ``"auto"`` counts). With
   damping, the template's HydroDyn is checked before any case is copied, and
   ``HydroFile`` joins the settle case's confinement set. Linearization cases are
   copied from the template and never see the damping. ``CaseProvenance`` records
   ``settle_criterion`` and ``settle_confirmation``, with an unmeasured RMS written as
   ``null``.

``run_openfast``
   A supplied timeout must be positive and finite (a boolean raises ``TypeError``, any
   other invalid value ``ValueError``); it is checked before any retained output is
   removed. Before launch, stale full-system and per-module (``LinOutMod``)
   linearization files and the primary deck's text, binary and summary outputs are
   removed; every path in the
   returned result therefore belongs to the current process invocation. The executable
   runs in its own process group (Windows) or session (POSIX); on a timeout the whole
   process tree is killed before ``OpenFastTimeoutError`` is raised, so a wrapper
   executable (a ``.bat`` or shell script launching OpenFAST) cannot leave an orphaned
   solver running or block the call on its still-open output pipes. The console output
   is decoded with the locale encoding, an undecodable byte being replaced (U+FFFD)
   rather than failing the run.

   The working directory and executable are passed without the Windows extended-length
   prefix (``\\?\``). OpenFAST does start from a prefixed working directory, but it
   still does not collapse ``..`` and the prefix's four characters count against its
   path limit, so it only fails sooner; a path that needs the prefix raises
   ``OpenFastError`` instead. The deck is also checked with ``check_deck_path_lengths``
   before launch; ``check_path_lengths=None`` (the default) runs the check on Windows
   only, and ``True``/``False`` forces it on or off. OpenFAST joins each referencing
   file's directory, as it spelled it, to the reference without collapsing ``..``
   (``<deck dir>\.\..\5MW_Baseline\<file>``), and the Windows build cannot open a
   path longer than ``OPENFAST_MAX_PATH_LENGTH`` (258) characters: it then aborts at
   initialization with ``The input file ... was not found`` although the file exists.
   The check raises ``DeckPathLengthError`` (a ``DeckValidationError``) naming the
   longest such path and its length; the remedy is a shorter ``output_dir``.
   ``classify_openfast_output`` matches its signatures on the output with whitespace
   and line breaks collapsed, because OpenFAST wraps console messages at about 100
   columns.

JSON run records
   ``RunProvenance.write`` and the sweep's automatic-configuration record serialize
   strict finite JSON (no ``NaN`` or infinity constants) before touching the
   destination. They write a freshly created same-directory temporary file, remove it
   after a failure, and replace the destination atomically.

Floating settle set up from the deck (``vane.run.floating_auto``)
   ``find_controller_library(fst_path, *, search_paths=(), environ=None)`` reads the
   deck, writes nothing, and returns a ``ControllerLibraryResolution`` (``status``,
   ``library``, ``deck_library``, ``source``, ``searched``; ``available`` is
   ``status != "missing"``): ``"not_used"`` when ServoDyn is off or routes no control
   through a library; ``"deck"`` when ``DLL_FileName`` (resolved relative to the
   ServoDyn file) exists, does not carry another platform's suffix and loads in a child
   Python process (``library`` is then ``None``: nothing is rewritten); ``"found"``
   with the first match of the deck library's key (name with the suffix, one bitness tag
   ``_x64``/``_x86``/``_win64``/``_win32``/``_64``/``_32`` and a ``lib`` prefix removed,
   case folded) among files with the host suffix (``.dll``/``.so``/``.dylib``) that
   also load in a child process, searching, in order and each location once, the deck
   library's directory (when ``DLL_FileName`` resolves), the ServoDyn directory, the primary deck's
   directory, ``search_paths``, the ``VANE_CONTROLLER_LIBRARIES`` entries
   (``os.pathsep``-separated) and the lines of ``controller_libraries.txt`` in
   ``VANE_CONFIG_DIR`` (else ``%APPDATA%/vane`` on Windows, ``$XDG_CONFIG_HOME/vane`` or
   ``~/.config/vane``); an entry may be a file (used only when its key matches) or a
   directory (a plain or 64-bit build preferred to a 32-bit one); else ``"missing"``.
   ``is_floating_deck`` is true when an ElastoDyn platform DOF flag is true and either
   ``CompMooring`` is non-zero (a moored SubDyn substructure is a flexible floater) or
   ``CompHydro`` is non-zero with ``CompSub`` 0 or absent.
   ``probe_platform(fst_path, work_dir, ...)`` copies the template twice into
   ``work_dir/base`` and ``work_dir/damped`` (the template is never edited; a deck
   without an active HydroDyn file with an ``AddBLin`` matrix raises ``ValueError``
   first), configures a
   parked, zero-wind single linearization at ``t = 0`` with
   ``LinearizationSettings.offshore(hydro_radiation_mode=0, hydro_excitation_mode=0)``,
   sets ``CompAero``, ``CompServo`` and ``CompInflow`` to 0 where present and
   ``TMax = max(0.1, 2 DT)`` (``DT`` taken as 0.01 when absent), adds
   ``probe_damping`` (default 1e6 N s/m, 1e9 N m s/rad) to the second copy's ``AddBLin``,
   runs both, and passes the two state matrices to ``platform_probe_from_jacobians``.
   That function takes ``M^-1[:, v_j] = -(A_damped - A_base)[:, v_j] / b_j`` for each
   platform velocity state ``v_j``, computes left and right eigenvectors of ``A_base``,
   and for each DOF picks the oscillatory mode with the largest
   ``Re((w^H M^-1[:, v_j]) u[v_j] / (w^H u))``, summed over the mode's cluster of
   repeated eigenvalues (within ``1e-6 |lambda|``) as
   ``-Re trace((W^H U)^-1 W^H dA U)`` with ``dA = -M^-1[:, v_j] e_{v_j}^T`` (the
   per-mode value of a repeated root depends on the arbitrary eigenvector basis, the
   cluster sum does not). For that sensitivity ``s``, ``effective_mass = 1 / (2 s)``,
   ``natural_period = 2 pi / |lambda|``, ``damping_ratio = -Re lambda / |lambda|`` and
   ``mode_frequency = Im lambda / (2 pi)`` (Hz) are returned per DOF in a
   ``PlatformProbe`` (with ``probe_damping``, ``n_states`` and the ``slowest_period``
   property; a DOF that no oscillatory mode responds to is omitted), whose
   ``settle_damping(fraction)`` is ``SettlePlatformDamping.from_critical_fraction``.
   ``PlatformProbeError`` reports a failed run or one that wrote no ``.lin`` file, a
   ``.lin`` without ``A``, state matrices whose shape does not match the state
   descriptions, no free ElastoDyn platform DOF, or a probe damping that damped no
   oscillatory platform mode. ``plan_floating_settle`` combines both into a ``FloatingSettlePlan``
   (``controller``, ``floating``, ``platform_damping``, ``damping_source`` ``"user"`` /
   ``"probe"`` / ``"none"``, ``probe``, ``damping_fraction``, ``notes``;
   ``prescribed_rotor`` is ``not controller.available``; ``to_dict()`` is JSON-ready): a
   given ``controller_library`` is checked by ``check_controller_library`` (status
   ``"user"``, or ``"not_used"`` when ServoDyn runs no controller library; a missing
   file raises ``FileNotFoundError`` and one that does not load ``SteadyStateError``);
   a given damping is used as is; otherwise a floating deck with ``auto_damping`` is
   probed and sized at ``damping_fraction`` (default 0.5; a non-positive or non-finite
   value raises ``ValueError``) of critical,
   and a probe that raises ``PlatformProbeError``, ``OpenFastError`` (a timed-out or
   unstartable run), ``ValueError``, ``OSError``, ``FastInputError`` or
   ``LinAlgError`` leaves the damping off with a note. ``run_operating_sweep`` calls
   it once for ``settle=True`` (probe directory ``output_dir/probe``; options
   ``settle_auto_damping`` and ``controller_search_paths``), raises
   ``SteadyStateError`` before any settle when the library is missing and no curve or
   CSV was given, passes ``prescribed_rotor`` to every settle (seeded from the curve),
   linearizes an unconverged prescribed-rotor settle of a rotating point when
   ``CalcSteady`` is enabled (it is still listed in ``settle_failures`` with
   ``linearized=True``; a parked point must settle), and returns the plan as ``OperatingSweepResult.floating_plan``.

State-space radiation fits (``vane.run.radiation_fit``)
   ``read_wamit_radiation(path, *, water_density, length_scale=1.0)`` reads a single-body
   WAMIT ``.1`` file: ``PER = 0`` rows are ``A(inf)``, ``PER < 0`` rows (zero frequency)
   are ignored, other rows are at ``omega = 2 pi / PER``; values are scaled by
   ``rho L^k`` (added mass) and ``rho omega L^k`` (damping), ``k`` = 3, 4, 5 for two
   translations, mixed, two rotations. It raises ``RadiationFitError`` (a
   ``ValueError``) for a DOF index outside 1 to 6, no ``PER = 0`` rows, fewer than four
   finite frequencies, a malformed row, or a non-positive or non-finite
   ``water_density`` or ``length_scale``.
   ``fit_radiation_state_space`` fits every diagonal entry of
   ``K = B + j omega (A - A(inf))`` with a non-zero peak and every off-diagonal entry
   whose peak is at least ``coupling_threshold`` (0.05) of the geometric mean of the two
   diagonal peaks, by vector fitting (``iterations`` = 20 pole relocations, unstable
   poles reflected) with the final residues constrained to ``K(0) = 0``, raising the
   order two states at a time up to ``max_states_per_entry`` (20) until the relative
   error ``||K_fit - K|| / ||K||`` is at most ``tolerance`` (0.05), else keeping the best.
   The ``RadiationStateSpace`` groups each entry's real modal realization by input DOF
   (``states_per_dof``) with output ``C = -c`` (the radiation force).
   ``write_radiation_state_space`` writes HydroDyn's ``.ss`` layout (header, six enabled
   DOFs, state count, states per DOF, rows of ``A``, ``B``, ``C``) through an atomic
   replace. ``ensure_radiation_state_space(fst_path, *, tolerance=0.05,
   overwrite=False, confine_to=None)`` returns ``None`` when HydroDyn is off,
   ``PotMod != 1``, or ``<PotFile>.ss`` exists and ``overwrite`` is false; otherwise it
   fits ``<PotFile>.1`` (density from a positive ``WtrDens`` of the primary deck,
   HydroDyn or SeaState; length from ``WAMITULEN``), writes the file only when every
   entry is within 0.2 (``RadiationFitRecord.accepted``), logs a warning either way
   (the provenance of a written file, or the rejection of a fit above 0.2), and returns
   a ``RadiationFitRecord`` (``to_dict()`` is JSON-ready). More than one WAMIT body
   (``NBody != 1``), no usable ``PotFile``, a missing ``.1``, no positive ``WtrDens``,
   or a ``.ss`` target outside ``confine_to`` (checked before the existence check and
   the fit) raise ``RadiationFitError``.
   ``preview_radiation_state_space(fst_path, *, tolerance=0.05)`` performs the same fit
   without writing anything (an existing ``.ss`` is ignored; ``written`` is ``False``).
   Wave-excitation (``.ssexctn``) fits are not generated.

Known invalid uses
------------------

VANE's claims are deliberately narrow. Each entry point does exactly what is stated above
and **no more**; the following are the misreadings to avoid, per scientific claim. The
:doc:`../user_guide/limitations` page covers the modal-extraction assumptions in more
depth; this list focuses on the system-identification and estimator surface.

* **Modal extraction / Campbell / labels.** VANE faithfully recovers the modal content of
  the linearization *it is given*; it does **not** validate whether that linearization is
  physically correct. A comparison against published natural frequencies
  (``compare_to_reference``) is a **plausibility check**, not a pass/fail validation —
  published values come from other tools/model versions and answer a different question
  (see :doc:`../user_guide/validation`).
* **QoI residue map.** ``build_qoi_map`` is **modal-residue screening**, not a validated
  fatigue or design-load ranking; a large residue flags a mode worth examining, not a
  confirmed load contribution.
* **Observability and sensor placement.** Observability is a **structural** property
  (rank / Gramian conditioning), not an accuracy guarantee: a fully observable load can
  still be poorly estimated under model error or noise. Greedy ``recommend_sensors`` is
  **not globally optimal**, and ``load_reconstructability`` reports the observable
  *subspace fraction* and an a-priori variance — not field accuracy.
* **Estimator assessment / load reconstruction bound.** The reconstruction-error standard
  deviation is **model-conditional**: it holds under an accurate linear model, the correct
  ``Qd``/``R``, Gaussian stationary noise, calibrated sensors, and detectability. It is a
  bound *under those assumptions*, not a guarantee on a real, nonlinear, drifting turbine.
* **NIS consistency.** ``filter_consistency`` measures consistency against the
  **assumed** model and noise. A consistent run is **necessary, not sufficient**, for
  field validity, and a run on simulated measurements from the linear model is not a
  substitute for measured-data or nonlinear time-domain validation.
* **LPV scheduling.** ``schedule_model`` / ``run_scheduled_filter`` interpolation is valid
  only where the family's modes are tracked and phase-aligned. Through a low-tracking-MAC
  crossing it **warns but still blends** the bracketing models — it does *not* fall back to
  a nearest member, and the blend may not correspond to a single physical mode. Only
  *outside* the operating-point range does it clamp to the nearest member (an
  extrapolation, flagged as such). Heed the per-step confidence and extrapolation flag
  rather than assuming a safe fallback.
* **Reduced models / export.** A ``ModalReduction`` / ROM reproduces the **retained** modes
  in their frequency band, not the full-order response; ``balanced_truncation`` requires
  an asymptotically stable system (``ValueError`` otherwise), and its a-priori error bound
  holds only for that system.

Exceptions
----------

* ``LinFileFormatError`` (``vane.io``, a ``ValueError`` subclass): malformed or legacy
  ``.lin`` input.
* ``FstFileError`` and ``FastInputError`` (``vane.io``, ``ValueError`` subclasses): a
  malformed OpenFAST input file, or a keyword that cannot be found or edited.
* ``DeckValidationError`` (``vane.run``, a ``ValueError`` subclass, with the subclass
  ``DeckPathLengthError``), ``DeckReferenceError`` and ``RadiationFitError``
  (``ValueError`` subclasses): a deck that cannot be linearized as configured, a
  referenced file that cannot be followed, or radiation data that cannot be read or
  fitted.
* ``OpenFastError`` (``vane.run``, a ``RuntimeError`` subclass, with the subclass
  ``OpenFastTimeoutError``): the executable is missing, fails or times out.
* ``SteadyStateError`` (a ``RuntimeError`` subclass, with the subclass
  ``SweepSettleError``, which carries the complete sweep result), ``SweepCaseError``
  and ``PlatformProbeError`` (``RuntimeError`` subclasses): settle, strict-sweep and
  platform-probe failures.
* ``ValueError``: invalid arguments or incoherent inputs to an analysis function,
  including out-of-range mode, state or channel indices.
* ``TypeError``: an argument of the wrong type, for example a non-numeric
  ``min_known_participation``.
* ``FileNotFoundError``: a missing input file, or a directory without full-system
  ``.lin`` files passed to ``discover_operating_points``.
* ``IndexError``: ``PipelineResult.state_space`` with an out-of-range operating-point
  index.

Stability
---------

The public API documented in :doc:`api_reference` is stable. VANE follows semantic
versioning: a breaking change to the documented public API is made only with a new
major version and is recorded in the :doc:`../user_guide/changelog`. Names that start
with an underscore are private.
