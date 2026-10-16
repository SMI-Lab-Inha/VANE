.. SPDX-License-Identifier: Apache-2.0

Mode identification
===================

This page covers the eigenanalysis of the averaged model and what VANE derives from
it: modal parameters, mode shapes and their correlation, the physical labels and the
blade taxonomy, tracking modes across operating points, the review of crossings and
veerings that tracking exposes, and the measurement of how many azimuth samples the
average needs.

.. contents:: On this page
   :local:
   :depth: 1


Eigenanalysis and modal parameters
----------------------------------

The modes are the eigenpairs of the averaged state matrix,
:math:`\bar{\mathbf{A}}\mathbf{x}_i = \lambda_i\mathbf{x}_i`, computed by
:func:`vane.modal.compute_modes` (or :func:`vane.modal.modes_from_mbc` for an
:class:`~vane.io.MBCResult`). Each eigenvalue of an
underdamped mode is complex, :math:`\lambda = -\zeta\omega_n \pm j\,\omega_n\sqrt{1-\zeta^2}`,
from which

.. math::

   \omega_n = |\lambda| \ \text{(rad/s)}, \qquad
   \zeta = -\frac{\operatorname{Re}\lambda}{|\lambda|}, \qquad
   \omega_d = \operatorname{Im}\lambda = \omega_n\sqrt{1-\zeta^2}.

Complex-conjugate pairs are collapsed to the representative with positive imaginary
part, and only these oscillatory modes are returned. The solution separately counts
**unstable** eigenvalues (:math:`\operatorname{Re}\lambda>0`, ``n_unstable``) and
**over-damped** ones (real, stable and non-zero, ``n_overdamped``).

**Numerical conditioning.** A first-order perturbation :math:`\delta\mathbf{A}` shifts a
simple eigenvalue by [Wilkinson1965]_

.. math::
   :label: cond

   \delta\lambda_i = \frac{\mathbf{y}_i^{\mathsf H}\,\delta\mathbf{A}\,\mathbf{x}_i}
   {\mathbf{y}_i^{\mathsf H}\mathbf{x}_i},

with :math:`\mathbf{x}_i,\mathbf{y}_i` the right and left eigenvectors. The **eigenvalue
condition number**

.. math::

   \kappa_i = \frac{\lVert\mathbf{y}_i\rVert\,\lVert\mathbf{x}_i\rVert}
   {|\mathbf{y}_i^{\mathsf H}\mathbf{x}_i|} = \frac{1}{|\mathbf{y}_i^{\mathsf H}\mathbf{x}_i|}
   \quad(\text{unit-norm eigenvectors})

bounds the sensitivity (``ModalSolution.condition_numbers``); :math:`\kappa_i \ge 1`
by Cauchy–Schwarz, with :math:`\kappa_i = 1` for a normal matrix and large values
flagging a sensitive, strongly non-normal mode. It is evaluated on
:math:`\bar{\mathbf A}` as given, so it depends on the state units. An eigenvalue
within a relative distance of :math:`10^{-6}` of another is flagged **degenerate**
(``is_degenerate``): its mode shape is defined only up to a rotation within the
invariant subspace. The
conditioning and degeneracy factors reappear in the unified confidence on the
:doc:`campbell_uncertainty` page.

**Split rigid-body pairs.** A degree of freedom without restoring stiffness — the free
rotor azimuth of a variable-speed turbine — contributes a zero eigenvalue (the azimuth
integrator) next to a small real one (the aerodynamic damping of the rotor speed).
Where that damping is small, the finite-difference noise in the azimuth column of
:math:`\bar{\mathbf{A}}` merges the two into a complex pair of about 0.0005 Hz instead
of two real eigenvalues (on the reference-turbine sweeps: NREL 5-MW and IEA 3.4-MW at
6 m/s). By :eq:`cond`, a relative error :math:`\delta` in the matrix moves an eigenvalue
by up to :math:`\kappa_i\,\delta\,\rho(\bar{\mathbf{A}})`, so a pair with

.. math::

   |\operatorname{Im}\lambda_i| \le \delta\,\kappa_i\,\rho(\bar{\mathbf{A}})
   \qquad\text{and}\qquad \frac{|\lambda_i|}{2\pi} < f_\text{rb}

is not resolved as an oscillation and is counted as rigid-body
(``n_split_rigid_body_pairs``, included in ``n_rigid_body_modes``; its unresolved real
part does not count as unstable). In this test :math:`\kappa_i` is evaluated on the
diagonally balanced matrix, so it does not depend on the state units; :math:`\delta = 10^{-4}`
(``split_pair_rtol``) reflects the finite-difference Jacobians, and the floor
:math:`f_\text{rb} = 0.003` Hz (``rigid_body_frequency_hz`` of :func:`~vane.modal.compute_modes`, a 333 s period) keeps the
test away from every restoring-force mode — moored surge and sway lie near
0.005–0.01 Hz. The floor is needed because a model with stiff first-order states
(MoorDyn line nodes) has a large spectral radius, under which even its well-resolved
platform modes could fall inside the bound. On the reference sweeps the split pairs sit
10–200 times inside the bound; a well-conditioned slow pair is kept.


Mode shapes, correlation, and participation
-------------------------------------------

Mode shapes (the rows of the eigenvectors for the displacements and the first-order
states) are complex and phase-normalized so the dominant component is real and
positive; this convention does
not affect the correlation measures, which are phase-invariant. The **Modal Assurance
Criterion** [Allemang2003]_ between mode shapes :math:`\boldsymbol\phi_i` and
:math:`\boldsymbol\phi_j` is

.. math::
   :label: mac

   \mathrm{MAC}_{ij} =
   \frac{|\boldsymbol\phi_i^{\mathsf H}\boldsymbol\phi_j|^2}
   {(\boldsymbol\phi_i^{\mathsf H}\boldsymbol\phi_i)
    (\boldsymbol\phi_j^{\mathsf H}\boldsymbol\phi_j)} \in [0,1],

equal to one for collinear shapes and zero for orthogonal ones, and invariant to
per-mode complex scaling (:func:`vane.modal.compute_mac`). For state-space modes the
pole-weighted **MACXP** variant [Vacher2010]_ (:func:`vane.modal.compute_macxp`) also
weights the shapes by the eigenvalues,

.. math::
   :label: macxp

   \mathrm{MACXP}_{ij} =
   \frac{\left(\dfrac{|\boldsymbol\phi_i^{\mathsf H}\boldsymbol\phi_j|}
   {|\bar\lambda_i+\lambda_j|}
   + \dfrac{|\boldsymbol\phi_i^{\mathsf T}\boldsymbol\phi_j|}{|\lambda_i+\lambda_j|}
   \right)^{2}}
   {\left(\dfrac{\boldsymbol\phi_i^{\mathsf H}\boldsymbol\phi_i}
   {2|\operatorname{Re}\lambda_i|}
   + \dfrac{|\boldsymbol\phi_i^{\mathsf T}\boldsymbol\phi_i|}{2|\lambda_i|}\right)
   \left(\dfrac{\boldsymbol\phi_j^{\mathsf H}\boldsymbol\phi_j}
   {2|\operatorname{Re}\lambda_j|}
   + \dfrac{|\boldsymbol\phi_j^{\mathsf T}\boldsymbol\phi_j|}{2|\lambda_j|}\right)},

which equals one for a mode compared with itself and separates modes with similar
shapes but different poles. It is defined only for damped poles, so every entry in the
row or column of an undamped pole is returned as NaN. The per-DOF **participation** of
:func:`vane.modal.compute_participation` is the magnitude of each component of the
right eigenvector, normalized so the largest is one, with a negative sign for a
component more than 90° out of phase with the dominant one.

The physical label is decided instead by the scale-invariant left--right
(biorthogonal) participation of :func:`vane.modal.biorthogonal_participation`. For mode
:math:`k` with right and left eigenvectors :math:`\mathbf x_k` and :math:`\mathbf y_k`
of the full state matrix,

.. math::
   :label: label-participation

   p_{ik} = \frac{|\overline{y}_{ik}\,x_{ik}|}{\sum_j |\overline{y}_{jk}\,x_{jk}|},
   \qquad \sum_i p_{ik} = 1,

which is invariant to any diagonal rescaling of the states. :func:`vane.modal.label_solution`
sums :math:`p_{ik}` onto each DOF (its displacement and velocity states together) and
then per DOF category; first-order module states (aerodynamic, hydrodynamic radiation,
mooring, controller) are unclassified. The mode is named after the largest
*classified* category, provided classified DOFs carry at least a threshold share of the
participation (``min_known_participation``, default 0.5 — the structure must be the
majority participant); otherwise it is *Unidentified*. The confidence is the named
category's share of :math:`p_{\cdot k}`. A raw-magnitude vote is not used because a
coupled model's hundreds of module states (441 mooring-node states on a floating
turbine, for example) carry large amplitudes in their own units yet little
participation, and would otherwise out-vote the platform and tower DOFs of the modes
they merely follow. Where left eigenvectors are unavailable, or a mode's left--right
product is numerically unresolvable, the labeller falls back to the raw-magnitude
plurality vote. For a (near-)repeated eigenvalue the individual eigenvectors are an
arbitrary basis of the shared invariant subspace, so their factors are not unique. Modes
flagged ``is_degenerate`` are therefore labelled from the subspace instead: the diagonal
of the spectral projector :math:`P = X_c (Y_c^H X_c)^{-1} Y_c^H` onto the cluster, which
does not depend on the basis, is normalized into the state participation shared by every
mode of the cluster. A symmetric pair (e.g. a monopile's tower fore-aft / side-side pair)
thus receives one label whose confidence, about one half, exposes the even split.
Clusters are the connected components of the pairwise coincidence test (so a tolerance
chain forms one cluster), and a degenerate mode carries no multi-blade (whirl) suffix,
since its individual eigenvector does not define one.

**Support planes.** On a fixed-bottom support whose substructure flexibility is carried
by the ElastoDyn platform DOFs (a monopile with a Guyan-reduced substructure), the
global tower bending modes move the transition piece: the OC3 monopile's 2.42 Hz
fore-aft mode, for example, has participation 0.30 on platform surge and 0.06 on pitch
against 0.21 and 0.09 on the two tower fore-aft DOFs (and 0.29 on the second blade
flap), so a per-category vote calls it a platform mode. The labeller therefore pools,
per plane, the platform translation and tilt with the tower bending DOFs (fore-aft:
surge, pitch, TFA1, TFA2; side-side: sway, roll, TSS1, TSS2). A plane whose tower DOFs
carry at least a third of its pool (``TOWER_SUPPORT_SHARE``) and whose pool exceeds
every other category names a tower mode, after its dominant tower DOF, with the pool's
share as the confidence. The tower-share condition is what separates the two support
types without inspecting the model: a floating platform's rigid-body modes have
(near-)zero tower participation (0.00--0.01 on the OC3 spar and OC4 semi-submersible),
and the monopile's local substructure modes above 8 Hz carry less than a quarter, so
both keep their platform labels; the global bending modes of the OC3 and IEA 10-MW
monopiles carry 0.40--0.56.

Within each plane the tower modes are then **numbered by frequency**: the dominant tower
DOF's number (1 or 2) is a lower bound, and each mode is numbered above the previous
one, the modes of a degenerate cluster sharing a number. A fixed-bottom plane has more
global bending modes than tower DOFs, so the monopile's third fore-aft mode (5.26 Hz on
the OC3 monopile, still dominated by TFA2) becomes "3rd Tower Fore-Aft" rather than a
second "2nd". The reported ``category`` remains the dominant DOF category.

**BeamDyn blades.** A BeamDyn blade has no modal DOFs; its states are nodal
translations and section rotations in the blade frame frozen at the operating point
(see :doc:`../user_guide/coordinate_systems`). They vote in the same participation
scheme as flapwise (translation along :math:`X`, rotation about :math:`Y`), edgewise
(translation along :math:`Y`, rotation about :math:`X`), torsional (rotation about
:math:`Z`), or axial (translation along :math:`Z`) blade motion. The vote gives the
family; the mode *order* comes from the spanwise shape. Let
:math:`\theta_j` be the real section-rotation profile of the mode over the state nodes
:math:`j = 2,\dots,N`, in root-to-tip order — the rotation about :math:`Y` for flap,
about :math:`X` for edge, about :math:`Z` for torsion — taken from the multi-blade
coordinate (or blade) carrying most of its energy and turned to the phase that
maximizes its real part. The increment :math:`\Delta\theta_j = \theta_{j+1} -
\theta_j` is the rotation accumulated over one span segment, so its sign is that of
the segment's mean curvature (or twist rate), and a cantilevered blade's
:math:`n`-th mode changes that sign :math:`n-1` times. The shape order is

.. math::
   :label: beamdyn-shape-order

   n_{\text{shape}} = 1 + \#\{\text{sign changes of } \Delta\theta_j\ :\
   |\Delta\theta_j| > 0.05\,\max_i|\Delta\theta_i|\}.

The increments, unlike the nodal translations, are blind to a rigid rotation of the
blade root, which every linearized nodal state carries. Within one family and one
multi-blade type (collective, regressive, progressive) the modes are then numbered in
frequency order, each at least its :math:`n_{\text{shape}}` and above the previous
one; standing cyclic or degenerate modes, and models without MBC, share each number
among the two cyclic coordinates (three blades without MBC). The frequency order
therefore numbers a regressive/collective/progressive triplet consistently even where
the Coleman split interleaves families, and the shape bound keeps a mode's number when
a lower mode of its family is missed (for example, an aerodynamically swamped first
flap). Because the numbering counts within a multi-blade type, a member carrying the
wrong type would be counted in another type's sequence and shift every number above
it; the modes of one category and shape order are therefore first split into families
of three at their widest frequency gaps, and a family that breaks the
one-mode-per-type rule has its types assigned jointly (as described below for the
per-point rule) before the numbering. A mode whose section rotation varies along the
span by less than 10 % of its largest value (``BEAMDYN_RIGID_RANGE``) is rigid blade
motion, not bending: it is not numbered, and a rigid collective edgewise mode — all blades turning together in the
rotor plane — is the rotor-azimuth mode, reported as *Rotor/Generator*.

On the NREL 5-MW BeamDyn model of the OpenFAST regression tests this gives 1st flap
at 0.54/0.77/0.94 Hz
(regressive/collective/progressive at 12.1 rpm), 1st edge 0.92/1.32 Hz, 2nd flap
1.83/2.12/2.22 Hz, 2nd edge 3.60/3.99 Hz and 1st torsion 5.46/5.66/5.86 Hz. On an IEA
22-MW rotor-only sweep at 6 m/s the numbered lines reproduce the published OpenFAST
Campbell labels within 2 % in frequency for the first and second edge, second and third
flap; the published legend swaps two pairs that the shape resolves differently (the
2.61 Hz collective edge, whose shape is the 2nd edge, and the 3.5--3.7 Hz triplet,
whose participation is 79--88 % torsion, against a 4th flap at 3.8--4.0 Hz with three
curvature sign changes).

The states mix units — blade and tower deflections are in metres while the remaining
DOFs are rotations (radians) or generalized coordinates — so comparing raw eigenvector
components over-weights whichever carries the larger numeric units. Participation is
unscaled by default, but it can optionally scale each translational deflection by a
characteristic length before normalizing: blade flap/edge by the flexible blade length
and tower fore-aft/side-side by the flexible tower length, turning each into a
dimensionless tip-deflection ratio, while rotations and rigid-body platform DOFs are left
unscaled. A BeamDyn blade's nodal translations are scaled by the blade length as well
(its nodal rotations, in radians, are not): on the NREL 5-MW BeamDyn regression-test
model the
largest nodal translation of the first flap and edge modes is 24--49 times the largest
rotation unscaled, and 0.4--0.8 times it scaled by the 61.5 m blade, about what a
cantilever's tip deflection ratio is to its tip slope.
:func:`vane.modal.participation_scale_factors` derives these per-DOF factors from the
state descriptions and the turbine geometry to hand to the participation computation.

The scaling matters only on the raw-magnitude paths: :func:`vane.modal.compute_participation`
and the plots built on it, and the labeller's fallback for a mode without usable left
eigenvectors. The labels themselves use the biorthogonal participation
:eq:`label-participation`, which a diagonal rescaling of the states leaves unchanged
(the right eigenvector scales by :math:`S` and the left by :math:`S^{-1}`, so each
product :math:`\overline{y}_{ik}x_{ik}` is unchanged), so they
need no scale factors; the MAC used for tracking compares one model with itself across
operating points, in the same units at every point.


Blade-mode taxonomy
-------------------

After the MBC transform each blade triplet is exactly
:math:`[\,q_0,\ q_c,\ q_s\,]`, so a mode is **collective** when its participation
concentrates in the collective coordinates and **cyclic** otherwise. The split uses the
biorthogonal participation of :eq:`label-participation`; where it is unavailable, it uses
each coordinate's share of the blade energy

.. math::

   \sum_{b=1}^{N} |q_b|^2 = N\,|q_0|^2 + \tfrac{N}{2}\big(|q_c|^2 + |q_s|^2\big)
   \qquad (N \ge 3),

so a cyclic coordinate counts half. Summing raw magnitudes instead would count the
cosine/sine pair twice and turn a mainly collective mode that has mixed with a nearby
cyclic one into a second cyclic mode. A mode is collective when the collective share is
at least the cyclic share. A cyclic mode is classified by its whirl direction from the
phase-invariant indicator of its dominant cosine/sine pair

.. math::
   :label: whirl

   w = \operatorname{Im}\!\big(q_s\,\overline{q_c}\big),

which is positive for a **regressive** (backward-whirling) mode and negative for a
**progressive** (forward-whirling) one. The sign follows from
:math:`q^{(b)} = q_0 + q_c\cos\psi_b + q_s\sin\psi_b` and is consistent with the
Coleman split below: the lower-frequency :math:`\omega_0-\Omega` line is the regressive
one. A whirl direction is assigned only when the pair is close to circular, measured by
the circularity

.. math::

   c = \frac{2\operatorname{Im}\big(q_s\,\overline{q_c}\big)}{|q_c|^2 + |q_s|^2}
   \in [-1, 1],

which is :math:`\pm1` for a pure backward or forward whirl and zero for a standing
pattern. Below :math:`|c| =`
:data:`~vane.modal.labeler.WHIRL_CIRCULARITY_THRESHOLD` (0.2) the mode is reported as a
standing ``"cyclic"`` pattern, so a mixed pair is not given a whirl direction. The type is reported only for a mode labelled as a
blade mode or *Unidentified*: a mode whose participation assigns it to the tower,
drivetrain, or platform carries no multi-blade type. A blade mode always receives its
type (the vote has already found it blade-dominated, even where the tower or drivetrain
hold most of the remaining participation, as for the collective edge mode coupled to
the drivetrain); an *Unidentified* mode receives one only when the rotor coordinates
carry at least half of its participation. For an isotropic rotor the three blade modes appear
at the **Coleman frequencies**

.. math::
   :label: coleman

   \omega_{\text{collective}} = \omega_0, \qquad
   \omega_{\text{regressive}} = \omega_0 - \Omega, \qquad
   \omega_{\text{progressive}} = \omega_0 + \Omega,

with :math:`\omega_0` the rotating-frame blade frequency. The multi-blade type is
derived from the structure of the eigenvector and the whirl sign, not from channel
names, and on a rotor-speed sweep it reproduces the ordering of :eq:`coleman`.

Ambiguity and one mode per multi-blade type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A label is a vote, and a vote between two near-equal shares is not an identification.
Every :class:`~vane.modal.labeler.ModeLabel` therefore carries its runner-up
(``alternative``, ``alternative_category``, ``alternative_confidence``) and an
``is_ambiguous`` flag, raised when

* the runner-up category (unclassified module states included, a support plane with
  its pooled weight) carries at least
  :data:`~vane.modal.labeler.LABEL_AMBIGUITY_RATIO` (0.75) of the winner's weight; or
* the collective and cyclic shares of a blade mode are within that ratio of each
  other, or its dominant cosine/sine pair is a standing pattern
  (circularity below :data:`~vane.modal.labeler.WHIRL_CIRCULARITY_THRESHOLD`).

On the reference-turbine sweeps the clean modes have runner-up ratios below 0.4; the
modes whose label would otherwise change between neighbouring points sit between 0.8
and 1.0:
the IEA 10-MW monopile's second fore-aft mode below rated (51 % progressive second
flap against 45 % fore-aft support plane), the NREL 5-MW collective first edge and
second flap at 25 m/s (48 % against 44 %, exchanged by the 23° pitch), the
near-critical collective first flap of the IEA 15- and 22-MW rotors at cut-out (40 %
flap against 39 % generator azimuth: the collective flap root couples with the rotor
speed into a single complex pair seen at that point only), and the IEA 22-MW rotor's
first flap with unsteady aerodynamics (54 % flap against 45 % aerodynamic states).
Such modes are reported, with both names, instead of being named after a few percent
of participation.

A rotating-frame blade mode of a three-bladed rotor has exactly one collective and one
regressive/progressive pair in the fixed frame, the regressive below the progressive.
Near-coincident members of a family mix, and the per-mode vote can then break this: on
the NREL 5-MW OC3 monopile at 4--6 m/s the second-flap collective and progressive
members 0.9 % apart were both "collective" (collective shares 0.48 and 0.51); on the
IEA 10-MW and 22-MW monopiles a tower or platform-yaw hybrid claimed the same whirl type
as the pure family member; on the IEA 3.4-MW and 22-MW rotors the progressive member was
a standing pattern at low rotor speed. Wherever one point's family breaks the rule,
:func:`~vane.modal.label_solution` assigns the three types jointly: each lightly damped
member scores its family share times its fit to the type (the collective share for
collective; the cyclic share times :math:`(1 \pm c)/2`, with :math:`c` the
circularity, for regressive/progressive), the best total is taken subject to the
regressive lying below the progressive and a member being given only a type it fits
by at least a quarter (so a pure whirl never takes a free collective slot), a member
whose type changes is flagged with its previous label as the alternative, and a member
left over takes its non-blade alternative when that is a close competitor of its vote
(the IEA 10-MW hybrid becomes the ambiguous "2nd Tower Fore-Aft", the IEA 22-MW one
"Platform Yaw") or keeps its family name without a type. Heavily damped modes
(below) are left out: the fixed-frame images of an overdamped blade mode whirl forward
at the rotor speed and are not members of a whirl pair. The tower modes of each plane
are numbered after this assignment, so a hybrid renamed after its tower alternative
keeps the tower numbering stable.

**BeamDyn drivetrain.** With BeamDyn blades the rotor's rotation sits in the nodal
states, so the drivetrain-torsion mode does not vote for the ElastoDyn drivetrain DOF.
In a model with a flexible drivetrain (``DOF_DrTr``) it is named
``"Drivetrain Torsion"`` when (generator free) it votes for the generator azimuth but
the blades' collective edgewise rotation is elastic, or (generator locked) it votes
collective edgewise and the blade-root section rotates in phase with the tip by at
least :data:`~vane.modal.labeler.DRIVETRAIN_ROOT_SHARE` (0.25) of the peak rotation:
the whole rotor swings against the shaft spring, whereas in edgewise bending the hub
stays put or counter-rotates. On the NREL 5-MW BeamDyn model of the OpenFAST regression
tests (``DrTrDOF`` on, ``GenDOF`` off), the 0.663 Hz mode (root rotation 0.50 of the
tip, in phase) is labelled as the drivetrain torsion, which the reference turbine
definition places at 0.62 Hz [Jonkman2009]_, and the 2.71 Hz mode (hub
counter-rotating) as the first collective edge mode.

**Flap/edge split and structural twist.** The BeamDyn flap/edge split is the
pitch-axis split of the blade reference frame (:mod:`vane.config.dof_map`); the local
structural twist rotates a principal bending direction by :math:`\theta` and moves
about :math:`\sin^2\theta` of its participation to the other family. For the root
twists of the reference blades (up to about 15°) this is below 7 % and changes no
label on the reference data; it lowers the confidence and, near a flap/edge
coincidence, may raise the ambiguity flag. A twist-resolved split would need the blade
structural input, which the linearization file does not carry.


Cross-operating-point tracking
------------------------------

Modes computed independently at successive operating points (rotor or wind speeds) are
linked into the lines of a Campbell diagram by :func:`vane.modal.identify_modes`.
Define the link **affinity** between mode :math:`i` at point :math:`k` and mode
:math:`j` at point :math:`k{+}1` as the MAC weighted by a frequency-continuity penalty,

.. math::
   :label: affinity

   a^{(k)}_{ij} =
   \begin{cases}
   \mathrm{MAC}_{ij}\,\max\!\big(0,\ 1 - w_f\,|f_i-f_j|/\Delta f\big)\,\ell_{ij},
   & \mathrm{MAC}_{ij}\ge \tau,\\[2pt]
   -\infty, & \text{otherwise},
   \end{cases}

with frequency weight :math:`w_f` (``frequency_weight``, default 0.5), MAC threshold
:math:`\tau` (``mac_threshold``, default 0.5), :math:`\Delta f` the span of the natural
frequencies at the two points, and :math:`\ell_{ij}` the label factor described below
(one unless the two labels differ). A link that passes the threshold also receives a
small constant bonus, so a track continues even where the frequency penalty reaches
zero. Both tracking
strategies share this affinity and differ only in how they resolve it into tracks. The
default, ``strategy="single_path"``, extracts each track as the **globally optimal
continuity path** over the not-yet-assigned modes,

.. math::

   \max_{\text{path } p}\ \sum_{k} a^{(k)}_{p_k,\,p_{k+1}},

by a deterministic dynamic program over the layered mode graph, then removes that path
and repeats until every mode is assigned. ``strategy="simultaneous"`` instead resolves
each transition :math:`k \to k{+}1` with a single **maximum-weight assignment**,

.. math::

   \max_{M}\ \sum_{(i,j)\in M} a^{(k)}_{ij},

over one-to-one matchings :math:`M` (both sides free to stay unmatched), solved by the
Hungarian algorithm. No early high-affinity link can then claim a mode a different
pairing would have used to reach a higher total — the weakness of a per-line
extraction at dense crossings. Each track reports a **confidence** (its weakest-link
MAC) and an **ambiguity** flag where a mode had two continuations whose MAC differs by
less than ``ambiguity_margin`` (default 0.2), which marks a crossing or veering.
Neither strategy performs a single assignment over the whole grid at once; see
:doc:`../user_guide/limitations`.

Each track is named by a **confidence-weighted vote** of its points' labels (each
including the multi-blade type): every distinct label scores the summed confidence of
the points that carry it, and the best-scoring label names the line. Only unambiguous
point labels vote when the track has any. A point where the mode is mixed — a veering
with a neighbouring line, or a degenerate point that carries no whirl suffix —
therefore does not rename a line that is consistent elsewhere. Each track keeps its
``point_labels`` and the fraction ``label_agreement`` of them equal to its label, and
its ``is_ambiguous`` flag is also raised when an unambiguous point label contradicts
the track label (a mode whose character rotates along the line, as the NREL 5-MW
collective edge/second flap pair under pitch), when every point label is ambiguous, or
when the track is a single point of a multi-point sweep (nothing confirms it).

**Labels in the affinity.** Two modes with different unambiguous labels — another
family, number or multi-blade type — are unlikely to be one mode, so their affinity
is multiplied by :data:`~vane.modal.identifier.LABEL_MISMATCH_PENALTY` (0.5). Ambiguous
labels, as at a veering, are not penalized.

**Near-coincident modes.** Two modes of one point whose eigenvalues lie within
:data:`~vane.modal.identifier.NEAR_COINCIDENT_RTOL` (1 %) of each other are mixed by
any coupling of comparable size, so the individual eigenvectors are an arbitrary basis
of their pair's subspace and their MAC with a neighbouring point measures the mixing
rather than the identity. On the IEA 22-MW monopile at 10 m/s the second-flap
collective and regressive members are 0.07 % apart, each a 57/38 mix; their MAC with
the pure collective mode at 12 m/s is at most 0.43, so on the MAC alone the
collective line would break into two tracks at 10 m/s. For a link between two modes
carrying the same unambiguous
label, the **subspace MAC** :math:`\|Q^H \boldsymbol\phi\|^2/\|\boldsymbol\phi\|^2` of
one end against the other end's near-coincident cluster (orthonormal basis :math:`Q`)
is used when it is larger; the one-per-type labels then decide which cluster member
continues which line.

**MAC scaling.** The mode shapes mix metres, radians and the units of the module
states, so the MAC depends on the state units. A MAC with each row normalized by its
RMS value was tried on the reference sweeps and not adopted: on the ElastoDyn-only
models (20 DOFs) it did not
separate the second-flap members better than the raw MAC, and on the VolturnUS-S model
(559 rows, 539 of them aerodynamic, hydrodynamic and mooring states) it lowered the
correlation of continuing modes near the first tower modes between 6 and 8 m/s from
0.98--0.99 to 0.46--0.65, because it gives every unclassified state the weight of a
structural DOF. The raw MAC is kept; its unit dependence is bounded by the
participation-based labels, which are scale-invariant, and by the near-coincident
subspace test above.

Modes below ``rigid_body_frequency_hz`` (default 0.001 Hz, a 1000 s period) are left out
of every track. They have no restoring stiffness: the free rotor azimuth gives a zero
and an aerodynamic-damping eigenvalue, which the eigensolver normally returns as real
and drops, but where the two nearly coincide they come back as a complex pair of about
0.0005 Hz and would otherwise form a line at isolated operating points. The eigensolver
already classifies such numerically split pairs as rigid-body (see *Split rigid-body
pairs* above); the tracking floor remains as a backstop for a slow pair the solver
judged resolved. Restrained modes are far above the floor (the softest moored surge and
sway modes lie near 0.005--0.01 Hz).

Heavily damped (near-critical) modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A mode close to critical damping barely oscillates: its natural frequency
:math:`|\lambda|/2\pi` mostly measures a decay rate, and its shape correlates weakly
with its neighbours across operating points. Tracked together with the lightly damped
modes, it can take over or split a Campbell line. VANE therefore classifies a mode as
**heavily damped** when its damping ratio is at least ``heavily_damped_threshold``
(default 0.99, :data:`vane.modal.identifier.DEFAULT_HEAVILY_DAMPED_THRESHOLD`) and
tracks these modes in a **separate tier**, linked among themselves only, into
``IdentificationResult.heavily_damped_tracks``. Every heavily damped mode is listed per
operating point in ``heavily_damped_modes``, so nothing is dropped silently. These are
complex (oscillatory in the fixed frame) modes; the eigensolver's ``n_overdamped``
counts the real eigenvalues, which are never returned as modes.

The damping ratio is read in the frame where the mode is defined. A cyclic blade mode
with rotating-frame eigenvalue :math:`-\sigma + j\omega_r` appears after the MBC
transform at :math:`\omega_r \pm \Omega` (:eq:`coleman`) with the same decay rate, so
its fixed-frame ratio :math:`\sigma/|\lambda|` differs between the two whirls and rises
towards 1 on the regressive branch as :math:`\omega_r` nears :math:`\Omega`, although
the blade mode is no more damped. :func:`vane.modal.rotor_frame_damping_ratios` undoes
the shift, :math:`\omega_r = |\omega_d - \Omega|` for a progressive and
:math:`\omega_r = \omega_d + \Omega` for a regressive mode (both, keeping the smaller
ratio, for a standing cyclic mode), giving

.. math::
   :label: rotor-frame-damping

   \zeta_r = \frac{\sigma}{\sqrt{\sigma^2 + \omega_r^2}},

the same for the regressive, collective and progressive members of a blade mode. The
rotor speed comes from the linearization (the pipeline passes it, and without explicit
speeds each solution's own ``rotor_speed_rpm`` is used, as by the labels); where it is
unknown a whirl mode is never classified. Other modes keep their fixed-frame ratio.

Past critical damping the rotating-frame blade mode has two real roots
:math:`-\sigma_1, -\sigma_2`. The collective coordinate keeps them real (the
eigensolver drops them), while each cyclic pair becomes the fixed-frame pair
:math:`-\sigma_k \pm j\Omega`: an "oscillation" exactly at the rotor frequency, with
natural frequency :math:`\sqrt{\sigma_k^2+\Omega^2}`, :math:`\zeta_r = 1`, and a
fixed-frame ratio that can be as low as 0.85. The heavily damped tier is linked by the
complex-eigenvalue distance :math:`|\lambda_i - \lambda_j|/2\pi` in place of the
natural-frequency difference in :eq:`affinity`, because both images share the damped
frequency :math:`\Omega` and differ only in decay rate. (The same distance is available
for the main tracks as ``continuity_metric="eigenvalue"``; it equals the frequency
difference for lightly damped modes. On the reference sweeps below it changes nothing
except one IEA 3.4-MW second-flap crossing, which it resolves against the labels, so
the natural-frequency default is kept.) No distance can carry a whirl line through the
transition itself: it is an exceptional point, where one underdamped pair turns into
two real roots, not the continuation of one mode.

**Choice of the default.** In tests on CalcSteady sweeps (36 azimuths,
quasi-steady aerodynamics, frozen wake) of the NREL 5-MW land and OC3 monopile, IEA
3.4-MW, IEA 10-MW monopile and IEA 15-MW monopile turbines, from 4 to 25 m/s, the 124
first-flap modes have fixed-frame damping ratios of 0.36--0.999 (median 0.69; 25 at or
above 0.9). Every first-flap line of the first four turbines tracks cleanly, including
regressive points at 0.996 (IEA 3.4-MW, 25 m/s) and 0.998 (IEA 10-MW, 25 m/s), whose
rotating-frame ratio :eq:`rotor-frame-damping` is 0.95 and 0.91. The IEA 15-MW blade
passes critical between 8 and 10 m/s: at 4--8 m/s its triplet sits at
:math:`\zeta_r = 0.85`--0.92 and tracks; from 10 m/s the collective is real and the
two remaining modes are the :math:`-\sigma_k \pm j\Omega` images (0.36 to 0.23 Hz and
1.36 to 2.01 Hz, :math:`\zeta_r = 1.000`). Tracked jointly by natural frequency, the
progressive line jumped from 0.66 Hz at 8 m/s to 0.36 Hz at 10 m/s and a second
"progressive" line appeared at 1.36--2.01 Hz. No fixed-frame threshold separates the
two populations (clean lines reach 0.998, spurious ones go down to 0.85), whereas the
rotating-frame ratio leaves a gap from 0.95 to 1.000; 0.99, within 1 % of critical,
sits in it. A lower threshold such as 0.9 would cut the IEA 3.4-MW and 10-MW
first-flap lines above rated. Published IEA 22-MW OpenFAST Campbell diagrams (unsteady
aerodynamics) still draw first flap at 0.25--0.78, far below the default. With the tier
the IEA 15-MW first-flap whirl lines end at 8 m/s, and the two images form two
heavily damped tracks from 10 to 25 m/s. A collective first-flap point at 25 m/s
(0.148 Hz, :math:`\zeta = 0.74`) is genuinely oscillatory and stays a one-point line;
it is the collective flap root coupled with the rotor speed (40 % against 39 %), so
both its label and its single-point track are reported ambiguous.

``heavily_damped_threshold=None`` disables the tier and tracks every mode together.
The Campbell diagram carries the tier separately
(:attr:`vane.campbell.CampbellDiagram.heavily_damped_tracks`, omitted with
``build_campbell(..., include_heavily_damped=False)``); it never enters resonance
detection, and :func:`vane.viz.plot_campbell` draws it as dotted grey lines with open
markers (``show_heavily_damped=False`` hides it).


Crossing and veering review
---------------------------

Tracking commits to one continuation at each transition. The **branch review**
(:mod:`vane.modal.branch`, enabled by ``identify_modes(..., branch_review=True)``)
reports where that choice could have gone the other way, so each such region can be
inspected. The tracks themselves are not changed. A pair of tracks :math:`(a,b)` is
flagged over an operating-point window when two conditions coincide:

* **Frequency proximity**: :math:`|f_a - f_b| \le \delta_f` over the window. By
  default the tolerance is a quarter of the median spacing between adjacent natural
  frequencies (the median over operating points), so it scales with the spectrum. The
  window is padded by one point on each side to include the approach and departure.
* **Shape competition**: across a transition in the window, either the
  cross-correlation :math:`\max(\mathrm{MAC}_{a\to b},\ \mathrm{MAC}_{b\to a})` reaches
  the competition threshold (``mac_competition``, default 0.25), or a track's own and
  crossed continuations are a near-tie,
  :math:`|\mathrm{MAC}_{a\to a} - \mathrm{MAC}_{a\to b}| < \varepsilon`
  (``ambiguity_margin``, default 0.2), with the larger of the two at or above the
  threshold.

A flagged window is classified from the frequency ordering at its ends: a **crossing**
if the order swaps, :math:`(f_a-f_b)\big|_{\text{start}}\,(f_a-f_b)\big|_{\text{end}} < 0`,
and a **veering** if the two lines approach and separate without swapping. Overlapping
pair-windows that share a track are merged by a union-find pass into a single group, so a
dense region with several interacting lines is reported as one region rather than many;
a group with more than two tracks is marked **dense**.

For an eligible group (two to four tracks present throughout at least three operating
points), ``resolve=True`` also computes a **minimum-curvature** alternative: over all
per-operating-point orderings of the group's lines it finds the assignment minimizing the
accumulated squared second derivative,

.. math::

   \min_{\text{orderings}}\ \sum_{k}\Big\| \tfrac{\mathrm d^2 f}{\mathrm d p^2}\Big|_{k}\Big\|^2,
   \qquad
   \frac{\mathrm d^2 f}{\mathrm d p^2}\bigg|_{k} =
   2\!\left[\frac{f_{k-1}}{h_-(h_-+h_+)} - \frac{f_k}{h_- h_+}
   + \frac{f_{k+1}}{h_+(h_-+h_+)}\right],

with :math:`h_-, h_+` the (possibly non-uniform) parameter spacings, by a dynamic program
over orderings with backtracking. These smooth branches are reported for review only
and never replace the identified tracks. They show how the lines would run if they
passed straight through the region, for comparison with the tracked result.


Azimuth-sampling convergence
----------------------------

The averaged model of :eq:`avg` is only as good as its azimuth sampling. VANE
measures the convergence from the retained per-azimuth matrices instead of relying on
a fixed rule. :func:`vane.modal.azimuth_density_sweep` recomputes the averaged natural
frequencies from evenly spaced sub-samples of :math:`d` azimuths (by default
:math:`d = 1, 3, 6, 12, 36`, where available) and compares each with the average over
all samples as the reference,

.. math::

   e(d) = \max_i \big| f_i^{(d)} - f_i^{\text{ref}} \big|,

where the frequencies are those of the underdamped modes of the sub-sampled mean matrix,
sorted in ascending order (the same convention as the eigenanalysis above). The number
of underdamped modes must stay the same at every density, so the comparison is
unambiguous. Given a tolerance, :func:`vane.modal.recommend_azimuth_count` returns the
smallest :math:`d` for which :math:`e(d)` **and the error of every denser sub-sample**
are within it. This condition ignores a coarse sample that happens to agree when a
denser one does not. The result is a number of azimuths for the given turbine and
tolerance, based on the model's own frequencies. The per-branch error estimates and the
:math:`\operatorname{lcm}(N, B)` aliasing rule on the :doc:`floquet` page predict the
same error without re-averaging.
