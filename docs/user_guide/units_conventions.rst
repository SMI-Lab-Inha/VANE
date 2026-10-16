.. SPDX-License-Identifier: Apache-2.0

Units and conventions
=====================

VANE follows OpenFAST's SI conventions on input and states the units of what it
reports.

Quantities and units
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Quantity
     - Unit
     - Notes
   * - Rotor speed (input)
     - rad/s
     - As written in the ``.lin`` header (``LinFile.rotor_speed``).
   * - Rotor speed (reported)
     - rev/min
     - ``rotor_speed_rpm`` = rad/s × 30/π (MBC result, modal solution, Campbell axis).
   * - Azimuth
     - rad (input), deg (reported)
     - ``MBCResult.azimuths_deg``; wrapped to ``[0, 360)`` deg for the sweep checks.
   * - Natural / damped frequency
     - Hz
     - ``natural_frequencies_hz``, ``damped_frequencies_hz``.
   * - Angular frequency
     - rad/s
     - Used in theory expressions; ``ω = 2π f``.
   * - Damping ratio
     - dimensionless
     - ``ζ`` in ``[0, 1)`` for an underdamped mode; negative for an unstable one.
   * - Wind speed
     - m/s
     - Mean hub-height; ``0`` when InflowWind is absent.
   * - Excitation harmonic
     - integer
     - The ``nP`` line, ``n × rpm / 60`` Hz; ``n`` a positive integer.

Sign and normalization conventions
----------------------------------

* **Eigenvalues**: of each complex-conjugate pair, the eigenvalue with positive
  imaginary part is kept.
* **Damping**: ``ζ = -Re(λ)/|λ|``; positive for a stable (decaying) mode. The
  solution separately counts unstable (positive-real-part) and over-damped (stable
  real) eigenvalues in ``n_unstable`` and ``n_overdamped``.
* **Mode shapes**: complex, phase-normalized so the dominant component is real and
  positive. Correlations (MAC) are invariant to per-mode complex scaling, so the
  normalization does not affect identification.
* **Whirl direction**: for the dominant cosine/sine pair of a blade triplet, the sign of
  ``Im(q_s · conj(q_c))`` sets the direction: positive is **regressive** (the
  lower-frequency ``ω0 - Ω`` line) and negative is **progressive**, calibrated against
  the Coleman split (see :doc:`../theory/modal_identification`). A pair whose
  normalized circularity is below ``WHIRL_CIRCULARITY_THRESHOLD`` (0.2) in magnitude
  is a standing ``"cyclic"`` pattern with no whirl direction.

Confidence and uncertainty
--------------------------

All confidence values are dimensionless and lie in ``[0, 1]``, with 1 the most
trusted, but they measure different things:

* ``ModeLabel.confidence`` is the share of a mode's participation carried by the
  category that names it.
* ``ModeTrack.confidence`` is the weakest link correlation (MAC) along a track.
* :func:`~vane.modal.uncertainty.unified_mode_confidence` combines conditioning,
  degeneracy, azimuth spread and tracking into one per-mode value.

The azimuth-spread standard deviations (:class:`~vane.modal.uncertainty.AzimuthSpread`)
are in the units of the underlying quantity: Hz for frequency, dimensionless for
damping. See :doc:`mode_identification`.

Tabular export
--------------

:mod:`vane.export` writes the modal and Campbell tables with the column names and
units above (for example ``natural_frequency_hz`` and ``damping_ratio``). The
operating-parameter column of the Campbell table is named after the diagram's parameter
(``rotor_speed_rpm`` or ``wind_speed``).
