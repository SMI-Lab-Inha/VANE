.. SPDX-License-Identifier: Apache-2.0

Coordinate systems, origins and datums
======================================

This page records the frames, orderings, and reference points VANE assumes, so that
results can be interpreted unambiguously.

Reference frames
----------------

Rotating (blade) frame
   The frame OpenFAST writes for blade degrees of freedom, rotating with the rotor.
   Channels carry a per-channel *rotating-frame* flag in the ``.lin`` tables; the MBC3
   transform uses it to find blade triplets.

Non-rotating (fixed) frame
   The frame after the MBC3 transform. The three per-blade coordinates of a triplet
   become a non-rotating ``[collective, cosine, sine]`` set. The averaged model lives
   entirely in this frame; tower, drivetrain, and platform degrees of freedom are
   already non-rotating and pass through unchanged. A rotating state that forms no
   complete triplet (for example of a one- or two-bladed rotor) stays in the rotating
   frame, and a warning is logged.

Azimuth convention
------------------

Rotor azimuth is read from the ``.lin`` header in radians. It is converted to degrees
(``MBCResult.azimuths_deg``) and wrapped to ``[0, 360)`` deg for the sweep checks, so a
file at 0 deg and one at 360 deg count as duplicates. The MBC3 transform averages the
transformed matrices over the sampled azimuths; for an isotropic rotor the averaged
model does not depend on the azimuth origin. The transform takes blade *b* at azimuth
:math:`\psi + 2\pi (b-1)/3`; see :doc:`../theory/linearization_mbc`.

State-space layout
------------------

After the transform, the state vector is reordered into the block layout

.. code-block:: text

   x = [ q2 ; q2dot ; q1 ]

where ``q2`` are the second-order (displacement) degrees of freedom, ``q2dot`` their
time derivatives, and ``q1`` the first-order states. A velocity is recognised by
the ``First time derivative of`` marker, or by a rate unit (``.../s``) on a state
whose position partner is present: MoorDyn ``Px``/``Vx`` node states and ServoDyn
structural-control ``X  m`` / ``dX/dt  m/s`` states (unit written after whitespace,
not a comma) pair this way. A marker velocity whose displacement partner is present
is paired with it whatever derivative order the file records (SubDyn writes order 0
for its rigid-body mode pairs).

Mode shapes are reported over the displacement and first-order states
(``q2`` then ``q1``); the eigensolver drops the velocity rows from the full
eigenvectors. Multi-blade coordinate annotations (``MBCResult.mbc_coordinates``, carried
to ``ModalSolution.dof_mbc_coordinates``) record, for each state, whether it is a
collective, cosine, or sine coordinate, or ``""`` for a non-rotating state.

Degree-of-freedom mapping
-------------------------

Each ElastoDyn channel description embeds an OpenFAST internal DOF token (for example
``DOF_TFA1`` for the first tower fore-aft mode or ``DOF_BF(1,1)`` for blade 1's first
flapwise mode). :func:`~vane.config.dof_map.classify_dof` maps that token onto a
physical :class:`~vane.config.dof_map.DofCategory`, the originating module, and the blade
index, which drive labelling and the collective, regressive and progressive blade-mode
types.

BeamDyn blade frame
~~~~~~~~~~~~~~~~~~~

BeamDyn states carry no DOF token. Each is a finite-element nodal quantity, for example
``BD_1 finite element node 2 (number of elements = 1; element order = 5)
translational displacement in X, m`` (or ``rotational displacement in Z, rad``), with
a velocity twin spelled ``BD_1 First time derivative of finite element node ...``.
``BD_<n>`` names the blade; node 1 is the blade root, a constraint that is never a
state, and the node index increases toward the tip.

The nodal components are expressed in the BeamDyn blade reference frame frozen at
the operating point: the blade-root orientation ElastoDyn passes to BeamDyn, which
is ElastoDyn's *pitched* blade coordinate system.

* ``Z`` runs along the pitch axis, root to tip.
* ``X`` is normal to the rotor plane at zero pitch: the flapwise direction.
* ``Y`` lies in the rotor plane at zero pitch: the edgewise direction.

The frame turns with the blade pitch, so ``X``/``Y`` stay flapwise/edgewise at any
pitch angle; only the local structural twist mixes them (most near the root), which
is the same pitch-axis flap/edge split ElastoDyn uses for its tip deflections.
:func:`~vane.config.dof_map.classify_dof` therefore maps a translation along ``X`` or a
rotation about ``Y`` to ``BLADE_FLAP``, a translation along ``Y`` or a rotation about
``X`` to ``BLADE_EDGE``, a rotation about ``Z`` to ``BLADE_TORSION``, and a
translation along ``Z`` to ``BLADE_AXIAL``, and reports the node and component on
:class:`~vane.config.dof_map.DofInfo`. The rotations are BeamDyn's Wiener-Milenkovic
parameters, which equal the rotation angle to first order.

Because the linearized beam's root boundary condition is the root-motion input
measured from the same frozen frame, every nodal state is the *total* perturbation of
that node: it includes the rigid motion the blade root carries (a rotor-azimuth
perturbation is a rigid rotation of every node), and the velocity states are total
velocities. The nodal states are flagged rotating, so the MBC3 transform groups the
``BD_1``/``BD_2``/``BD_3`` copies of each node and component into triplets.

Origins and datums
------------------

VANE does not redefine geometric origins; it inherits OpenFAST's. Rotor and tower
geometry extracted from the ElastoDyn input (tip and hub radius, tower height and base
height) is reported in the model's own datum. It is used for context, such as the
participation scale factors and the reference geometries of the mode-shape views, and is
not used to re-reference the state-space model.
