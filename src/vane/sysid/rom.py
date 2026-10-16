# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Reduced-order modal models from a linearized state space.

A full OpenFAST linearization carries hundreds of states, many of them fast,
weakly observable, or numerically marginal aero/actuator modes that a downstream
estimator neither needs nor can stabilise. This module reduces the system to the few
**structural modes that actually drive a quantity of interest**, in a real,
block-diagonal modal realization that preserves the inputs and outputs.

Each retained underdamped mode :math:`\lambda = \sigma + j\omega` (with right
eigenvector :math:`\mathbf v` and left eigenvector :math:`\mathbf w`,
:math:`\mathbf w^{\mathsf H}\mathbf v = 1`) becomes a real :math:`2\times2` block

.. math::

   \mathbf A_k = \begin{bmatrix}\sigma & -\omega\\ \omega & \sigma\end{bmatrix},\quad
   \mathbf B_k = \begin{bmatrix}\operatorname{Re}(\mathbf w^{\mathsf H}\mathbf B)\\
   \operatorname{Im}(\mathbf w^{\mathsf H}\mathbf B)\end{bmatrix},\quad
   \mathbf C_k = \big[\,2\operatorname{Re}(\mathbf C\mathbf v)\ \;
   -2\operatorname{Im}(\mathbf C\mathbf v)\,\big],

with the feed-through :math:`\mathbf D` unchanged. The block's eigenvalues are exactly
:math:`\sigma\pm j\omega`, so the retained modal frequencies and damping are preserved
to machine precision. Modes are ranked by their **residue** in a chosen output (the
load-observability ranking): the residue of mode :math:`k` in output :math:`i` is
:math:`|\mathbf C_i\mathbf v_k|`, i.e. how strongly that mode appears in that load.

This is the reduction engine the digital-twin / reduced-estimator path depends on. It is
the modal counterpart to balanced truncation; unlike balanced truncation it tolerates
the unstable aero poles a turbine linearization contains (they are simply not selected,
and any retained pole can optionally be reflected into the stable half-plane).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import eig as _eig

from vane.sysid.state_space import StateSpace, static_gain

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

__all__ = ["ModalReduction", "modal_reduction", "output_mode_residues"]

_TWO_PI = 2.0 * np.pi
_IMAG_TOLERANCE = 1e-8
# Eigenvalues this close are treated as one degenerate cluster (a genuinely repeated
# mode). Their modal projections are de-coupled together via the cluster dual rather
# than normalized independently, which would not be bi-orthogonal within the repeated
# subspace. The tolerance is tight so distinct-but-near (veering) modes are not
# clustered — those keep the well-conditioned per-vector normalization.
_CLUSTER_RTOL = 1e-8
_CLUSTER_ATOL = 1e-12
# A repeated eigenvalue is *defective* (non-diagonalizable, a Jordan chain) when its
# right eigenvectors do not span its eigenspace: the columns SciPy returns for the
# cluster are then (numerically) parallel and ``cond(V_c)`` blows up. A diagonalizable
# repeated cluster keeps ``cond(V_c)`` near O(1), so above this floor the cluster is not
# realizable as independent modal blocks and the reduction is rejected. The conditioning
# of the unit-norm eigenvector matrix is basis- and scale-independent, unlike a single
# diagonal overlap ``|w_i^H v_i|`` (which depends on LAPACK's arbitrary eigenbasis).
_DEFECTIVE_COND = 1e8
# Do not normalize numerical residue leakage into a full-strength QoI vote. A selected
# row is active only when its candidate-mode residue norm is material relative to the
# row norm of C; sqrt(eps) separates numerical from resolved first-order projection.
_REL_RESIDUE_TOL = np.sqrt(np.finfo(np.float64).eps)


@dataclass
class ModalReduction:
    """A modally reduced state space and the modes it retained.

    Parameters
    ----------
    system : StateSpace
        The reduced, real, block-diagonal continuous state space (two states per
        retained mode), with the original inputs and outputs.
    frequencies_hz : npt.NDArray[np.float64]
        Natural frequency of each retained mode (Hz), in the reduced state order.
    damping_ratios : npt.NDArray[np.float64]
        Damping ratio of each retained mode, consistent with the realized reduced ``A``
        block: a mode reflected into the stable half-plane (see ``n_stabilized``)
        reports its post-stabilisation (positive) damping, not the original value.
    residues : npt.NDArray[np.float64]
        The ranking residue of each retained mode (its magnitude in the ranked
        output, or its observability norm when no output was specified).
    energy_retained : float
        Fraction of the candidate modes' total squared residue captured by the
        retained modes, in ``[0, 1]`` — a cheap measure of how much of the ranked
        output's modal content the reduction keeps.
    n_full_states : int
        Number of states in the original system.
    n_stabilized : int
        How many retained modes had an unstable eigenvalue reflected into the stable
        half-plane (``0`` unless the system had unstable poles among those kept).
    state_projection : npt.NDArray[np.float64] or None
        Real map from a full physical state into the retained modal coordinates.
    state_reconstruction : npt.NDArray[np.float64] or None
        Real map from retained modal coordinates back to the selected full-state
        subspace.  Together these maps permit a consistently based periodic/LPV
        projection without independently reducing and re-phasing every azimuth.
    stabilized : npt.NDArray[np.bool_] or None
        Per retained mode, ``True`` where an unstable eigenvalue was reflected into the
        stable half-plane (``n_stabilized`` is its count). The physical (pre-reflection)
        damping of a reflected mode is ``-damping_ratios[k]``; its natural frequency is
        unchanged. ``None`` for a hand-built reduction that does not record it.
    """

    system: StateSpace
    frequencies_hz: npt.NDArray[np.float64]
    damping_ratios: npt.NDArray[np.float64]
    residues: npt.NDArray[np.float64]
    energy_retained: float
    n_full_states: int
    n_stabilized: int
    state_projection: npt.NDArray[np.float64] | None = None
    state_reconstruction: npt.NDArray[np.float64] | None = None
    stabilized: npt.NDArray[np.bool_] | None = None


def _resolve_output(system: StateSpace, output: int | str) -> int:
    """Resolve an output index or name to a row index of ``C``."""
    if isinstance(output, (int, np.integer)):
        n = 0 if system.c is None else system.c.shape[0]
        index = int(output)
        if not 0 <= index < n:
            msg = f"output index {output} out of range [0, {n})"
            raise ValueError(msg)
        return index
    if not isinstance(output, str):
        msg = (
            f"output must be an integer index or string name, got "
            f"{type(output).__name__}"
        )
        raise ValueError(msg)  # noqa: TRY004
    exact = [i for i, name in enumerate(system.output_names) if name == output]
    if exact:
        return exact[0]
    contains = [i for i, name in enumerate(system.output_names) if output in name]
    if not contains:
        msg = f"no output matches {output!r}; have {system.output_names}"
        raise ValueError(msg)
    if len(contains) > 1:
        hits = ", ".join(repr(system.output_names[i]) for i in contains)
        msg = (
            f"output {output!r} matches multiple channels ({hits}); use a more "
            "specific name so the ranked output is unambiguous"
        )
        raise ValueError(msg)
    return contains[0]


def _resolve_input(system: StateSpace, input_channel: int | str) -> int:
    """Resolve an input index or name to a column index of ``B``."""
    if isinstance(input_channel, (int, np.integer)):
        index = int(input_channel)
        if not 0 <= index < system.n_inputs:
            msg = f"input index {input_channel} out of range [0, {system.n_inputs})"
            raise ValueError(msg)
        return index
    if not isinstance(input_channel, str):
        msg = (
            f"input must be an integer index or string name, got "
            f"{type(input_channel).__name__}"
        )
        raise ValueError(msg)  # noqa: TRY004
    exact = [i for i, name in enumerate(system.input_names) if name == input_channel]
    if exact:
        return exact[0]
    contains = [i for i, name in enumerate(system.input_names) if input_channel in name]
    if not contains:
        msg = f"no input matches {input_channel!r}; have {system.input_names}"
        raise ValueError(msg)
    if len(contains) > 1:
        hits = ", ".join(repr(system.input_names[i]) for i in contains)
        msg = (
            f"input {input_channel!r} matches multiple channels ({hits}); use a more "
            "specific name so the transfer pair is unambiguous"
        )
        raise ValueError(msg)
    return contains[0]


def output_mode_residues(
    system: StateSpace, output: int | str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Rank the underdamped modes by their residue in one output (load).

    The residue of mode :math:`k` in the output is :math:`|\\mathbf C_i\\mathbf v_k|`
    — how strongly the mode appears in that load. This is the load-observability
    ranking that drives :func:`modal_reduction` and is the basis for choosing which
    modes a load estimator must retain.

    Parameters
    ----------
    system : StateSpace
        A continuous system with a ``C`` matrix.
    output : int or str
        The output row index, or an output name (exact match, else substring).

    Returns
    -------
    tuple of npt.NDArray[np.float64]
        ``(frequencies_hz, damping_ratios, residues)`` for the underdamped modes,
        sorted by descending residue.

    Raises
    ------
    ValueError
        If the system is discrete, has no ``C`` matrix, or the output cannot be
        resolved.
    """
    if system.dt is not None:
        msg = "output_mode_residues requires a continuous-time system (dt is None)"
        raise ValueError(msg)
    if system.c is None:
        msg = "system has no C matrix; cannot rank modes by output residue"
        raise ValueError(msg)
    row = _resolve_output(system, output)
    eigvals, eigvecs = np.linalg.eig(system.a)
    keep = eigvals.imag > _IMAG_TOLERANCE
    lam = eigvals[keep]
    vecs = eigvecs[:, keep]
    residues = np.abs(system.c[row] @ vecs)
    order = np.argsort(residues)[::-1]
    lam = lam[order]
    magnitude = np.abs(lam)
    freqs = magnitude / _TWO_PI  # natural frequency |lambda|
    damping = np.where(magnitude > 0.0, -lam.real / magnitude, 0.0)
    return freqs, damping, np.asarray(residues[order], dtype=np.float64)


def modal_reduction(
    system: StateSpace,
    *,
    n_modes: int | None = None,
    frequency_band_hz: tuple[float, float] | None = None,
    rank_output: int | str | None = None,
    rank_outputs: Sequence[int | str] | None = None,
    required_transfer_pairs: Sequence[tuple[int | str, int | str]] | None = None,
    n_required_transfer_modes: int = 0,
    stabilize: bool = True,
    match_dc_gain: bool = False,
) -> ModalReduction:
    """Reduce a state space to a real modal realization of its dominant modes.

    Keeps the underdamped modes (optionally within a frequency band) with the largest
    residue, ranked in a chosen output (``rank_output``, the load-observability ranking)
    by an equal-weight, unit-normalized joint residue over several outputs
    (``rank_outputs``), or by the overall observability norm, as a real block-diagonal
    model with the inputs and outputs preserved. The retained modal frequencies and
    damping are exact; the full model's many marginal/unstable aero modes are simply
    dropped.

    Parameters
    ----------
    system : StateSpace
        A **continuous-time** system (``dt`` is ``None``) with a ``C`` matrix.
    n_modes : int or None, optional
        Number of modes to retain (the highest-residue ones). ``None`` keeps every
        candidate mode (every underdamped mode in the band).
    frequency_band_hz : tuple[float, float] or None, optional
        Restrict candidates to modes whose natural frequency is in ``[low, high]``
        (Hz), e.g. the structural band. ``None`` imposes no band.
    rank_output : int or str or None, optional
        Rank modes by their residue in this output (index or name) — keep the modes
        that drive this load. ``None`` ranks by the observability norm over all
        outputs unless ``rank_outputs`` is given.
    rank_outputs : Sequence[int or str] or None, optional
        Rank modes jointly for these output rows. Each row with numerically resolved
        candidate-mode content has its modal-residue vector normalized to unit Euclidean
        norm before the rows are combined, so quantities with different physical units
        (for example platform motion and line tension) receive equal weight. Mutually
        exclusive with ``rank_output``.
    required_transfer_pairs : sequence of tuple(output, input) or None, optional
        Input-output pairs whose strongest transfer-function modes must be retained
        before the remaining modal budget is allocated by the output ranking. Each
        pair is ranked by the true modal transfer residue
        :math:`|(C_i v_k)(w_k^H B_j)|`, normalized so pairs in different units receive
        equal weight. This is intended for reduced observers that must preserve a
        measured response to an estimated unknown input.
    n_required_transfer_modes : int, optional
        Number of modes reserved for ``required_transfer_pairs``. Must be zero when no
        pairs are supplied, positive when they are supplied, and no greater than
        ``n_modes`` when that budget is finite. Default ``0``.
    stabilize : bool, optional
        Reflect any retained unstable eigenvalue into the stable half-plane
        (``sigma -> -|sigma|``), so the reduced model is a stable estimator basis.
        Default ``True``.
    match_dc_gain : bool, optional
        Correct the reduced feedthrough so its zero-frequency gain exactly matches the
        full model. This preserves equilibrium stiffness when a large direct term is
        cancelled by omitted internal-state response. Both models must have ``B`` and
        nonsingular ``A`` matrices. Default ``False``.

    Returns
    -------
    ModalReduction
        The reduced system and the retained-mode metadata.

    Raises
    ------
    ValueError
        If the system is discrete, has no ``C`` matrix, ``n_modes`` is not positive,
        ``frequency_band_hz`` is invalid, the ranking selections conflict or are empty,
        no modes remain after filtering, or a retained mode belongs to a defective
        (non-diagonalizable) repeated-eigenvalue cluster — a Jordan chain a
        block-diagonal modal realization cannot represent.

    Notes
    -----
    The realization is exact for diagonalizable systems, including genuinely repeated
    (but non-defective) eigenvalues. A *defective* repeated mode (a Jordan chain) cannot
    be written as independent modal blocks and is rejected rather than silently reduced
    to a model with a different transfer function.
    """
    if system.dt is not None:
        msg = "modal_reduction requires a continuous-time system (dt is None)"
        raise ValueError(msg)
    if system.c is None:
        msg = "system has no C matrix; reduction needs outputs to preserve"
        raise ValueError(msg)
    if rank_output is not None and rank_outputs is not None:
        msg = "rank_output and rank_outputs are mutually exclusive"
        raise ValueError(msg)
    if n_modes is not None and n_modes <= 0:
        msg = f"n_modes must be positive, got {n_modes}"
        raise ValueError(msg)
    if n_required_transfer_modes < 0:
        msg = (
            "n_required_transfer_modes must be nonnegative, got "
            f"{n_required_transfer_modes}"
        )
        raise ValueError(msg)
    transfer_pairs = list(required_transfer_pairs or ())
    if transfer_pairs and n_required_transfer_modes == 0:
        msg = "required_transfer_pairs need n_required_transfer_modes > 0"
        raise ValueError(msg)
    if n_required_transfer_modes > 0 and not transfer_pairs:
        msg = "n_required_transfer_modes needs at least one required_transfer_pair"
        raise ValueError(msg)
    if n_modes is not None and n_required_transfer_modes > n_modes:
        msg = (
            f"n_required_transfer_modes={n_required_transfer_modes} exceeds "
            f"n_modes={n_modes}"
        )
        raise ValueError(msg)
    if transfer_pairs and system.b is None:
        msg = "required_transfer_pairs need a system with a B matrix"
        raise ValueError(msg)
    if (
        frequency_band_hz is not None
        and not frequency_band_hz[0] <= frequency_band_hz[1]
    ):
        msg = f"frequency_band_hz must have low <= high, got {frequency_band_hz}"
        raise ValueError(msg)

    c = system.c
    # Left and right eigenvectors together: the left set gives each mode's modal
    # projection without inverting the full right-eigenvector matrix, which raises or
    # amplifies noise badly for the defective / nearly coalesced modes a turbine's
    # veering and crossing modes produce.
    eigvals, left_vecs, eigvecs = _eig(system.a, left=True, right=True)
    candidate = eigvals.imag > _IMAG_TOLERANCE
    if frequency_band_hz is not None:
        freq_all = np.abs(eigvals) / _TWO_PI  # natural frequency
        candidate &= (freq_all >= frequency_band_hz[0]) & (
            freq_all <= frequency_band_hz[1]
        )
    cand_idx = np.flatnonzero(candidate)
    if cand_idx.size == 0:
        msg = "no underdamped modes remain after filtering; nothing to reduce"
        raise ValueError(msg)

    vecs = eigvecs[:, cand_idx]
    if rank_outputs is not None:
        requested = list(rank_outputs)
        if not requested:
            msg = "rank_outputs must contain at least one output"
            raise ValueError(msg)
        rows = [_resolve_output(system, output) for output in requested]
        if len(set(rows)) != len(rows):
            msg = "rank_outputs must resolve to distinct output rows"
            raise ValueError(msg)
        output_residues = np.abs(c[rows] @ vecs)
        scales = np.linalg.norm(output_residues, axis=1)
        row_norms = np.linalg.norm(c[rows], axis=1)
        nonzero = scales > _REL_RESIDUE_TOL * row_norms
        normalized = np.zeros_like(output_residues)
        normalized[nonzero] = output_residues[nonzero] / scales[nonzero, None]
        residues_all = np.linalg.norm(normalized, axis=0)
        if not np.any(residues_all > 0.0):
            msg = "rank_outputs have zero residue in every candidate mode"
            raise ValueError(msg)
    elif rank_output is None:
        residues_all = np.linalg.norm(c @ vecs, axis=0)
    else:
        residues_all = np.abs(c[_resolve_output(system, rank_output)] @ vecs)

    order = np.argsort(residues_all)[::-1]
    keep_count = order.size if n_modes is None else min(n_modes, order.size)
    if n_required_transfer_modes > cand_idx.size:
        msg = (
            f"n_required_transfer_modes={n_required_transfer_modes} exceeds the "
            f"{cand_idx.size} candidate modes"
        )
        raise ValueError(msg)
    if transfer_pairs:
        resolved_pairs = [
            (_resolve_output(system, output), _resolve_input(system, input_channel))
            for output, input_channel in transfer_pairs
        ]
        if len(set(resolved_pairs)) != len(resolved_pairs):
            msg = "required_transfer_pairs must resolve to distinct input-output pairs"
            raise ValueError(msg)
        assert system.b is not None  # validated above
        _reject_defective_modes(eigvals, eigvecs, cand_idx)
        dual = _dual_rows(eigvals, eigvecs, left_vecs, cand_idx)
        transfer_residues = np.asarray(
            [
                np.abs((c[output] @ vecs) * (dual @ system.b[:, input_channel]))
                for output, input_channel in resolved_pairs
            ],
            dtype=np.float64,
        )
        scales = np.linalg.norm(transfer_residues, axis=1)
        if np.any(scales <= np.finfo(np.float64).tiny):
            msg = "a required_transfer_pair has zero residue in every candidate mode"
            raise ValueError(msg)
        normalized_transfer = transfer_residues / scales[:, None]
        transfer_scores = np.linalg.norm(normalized_transfer, axis=0)
        required = np.argsort(transfer_scores)[::-1][:n_required_transfer_modes]
        required_set = set(required.tolist())
        remainder = [index for index in order if int(index) not in required_set]
        kept = np.asarray(
            [*required.tolist(), *remainder[: keep_count - required.size]],
            dtype=np.int_,
        )
    else:
        kept = order[:keep_count]
    sel = cand_idx[kept]

    total_energy = float(np.sum(residues_all**2))
    kept_energy = float(np.sum(residues_all[kept] ** 2))
    energy_retained = kept_energy / total_energy if total_energy > 0.0 else 1.0

    reduction = _build_reduced(
        system, eigvals, eigvecs, left_vecs, sel, stabilize=stabilize
    )
    if match_dc_gain:
        reduced = reduction.system
        if reduced.b is None or reduced.c is None:
            msg = "match_dc_gain requires a system with B and C matrices"
            raise ValueError(msg)
        full_gain = static_gain(system)
        try:
            reduced_equilibrium = np.linalg.solve(reduced.a, reduced.b)
        except np.linalg.LinAlgError as exc:
            msg = "match_dc_gain cannot correct a reduction with singular A"
            raise ValueError(msg) from exc
        reduced.d = np.asarray(
            full_gain + reduced.c @ reduced_equilibrium, dtype=np.float64
        )
    reduction.energy_retained = energy_retained
    reduction.residues = np.asarray(residues_all[kept], dtype=np.float64)
    return reduction


def _cluster_indices(
    eigvals: npt.NDArray[np.complex128], index: int
) -> npt.NDArray[np.int_]:
    """Return the indices of the degenerate cluster (equal eigenvalues) of ``index``."""
    return np.flatnonzero(
        np.isclose(eigvals, eigvals[index], rtol=_CLUSTER_RTOL, atol=_CLUSTER_ATOL)
    )


def _reject_defective_modes(
    eigvals: npt.NDArray[np.complex128],
    eigvecs: npt.NDArray[np.complex128],
    selected: npt.NDArray[np.int_],
) -> None:
    """Raise if a selected mode belongs to a defective (non-diagonalizable) cluster.

    A real ``2x2`` modal block can only reproduce a *diagonalizable* mode. A
    diagonalizable repeated eigenvalue has a full-dimensional eigenspace, so the right
    eigenvectors SciPy returns for the cluster are independent and ``cond(V_c)`` stays
    near O(1). A defective eigenvalue (a Jordan chain) has a deficient eigenspace, so
    those columns are (numerically) parallel and ``cond(V_c)`` blows up. The test uses
    the conditioning of the whole unit-norm cluster matrix (basis- and scale-free)
    rather than a single diagonal overlap, whose value depends on LAPACK's arbitrary
    eigenbasis within the eigenspace. Distinct (singleton) modes are always
    diagonalizable.

    Raises
    ------
    ValueError
        If any selected mode belongs to a defective repeated-eigenvalue cluster.
    """
    seen: set[int] = set()
    for i in selected:
        cluster = _cluster_indices(eigvals, int(i))
        root = int(cluster[0])
        if cluster.size == 1 or root in seen:
            continue
        seen.add(root)
        if np.linalg.cond(eigvecs[:, cluster]) > _DEFECTIVE_COND:
            msg = (
                "modal_reduction cannot reduce a defective (non-diagonalizable) "
                f"repeated mode near eigenvalue {eigvals[i]:.6g}: its eigenvectors do "
                "not span the eigenspace (a Jordan chain), so a block-diagonal modal "
                "realization would drop the generalized-eigenvector dynamics."
            )
            raise ValueError(msg)


def _dual_rows(
    eigvals: npt.NDArray[np.complex128],
    eigvecs: npt.NDArray[np.complex128],
    left_vecs: npt.NDArray[np.complex128],
    selected: npt.NDArray[np.int_],
) -> npt.NDArray[np.complex128]:
    r"""Return each selected mode's bi-orthonormal left-projection row (rows of V^-1).

    Left and right eigenvectors of *distinct* eigenvalues are orthogonal automatically,
    so a mode only needs decoupling from its own degenerate cluster (equal eigenvalues).
    For a singleton cluster the dual is just :math:`\mathbf w_i` scaled to
    :math:`\mathbf w_i^{\mathsf H}\mathbf v_i = 1`; for a repeated-eigenvalue cluster it
    is the small dual :math:`(\mathbf W_c^{\mathsf H}\mathbf V_c)^{-1}\mathbf
    W_c^{\mathsf H}`, which restores bi-orthogonality *within* the cluster without
    inverting the full (possibly ill-conditioned) eigenvector matrix.
    """
    rows = np.empty((selected.size, eigvecs.shape[0]), dtype=np.complex128)
    for k, i in enumerate(selected):
        cluster = _cluster_indices(eigvals, int(i))
        if cluster.size == 1:
            w = left_vecs[:, i].conj()
            rows[k] = w / (w @ eigvecs[:, i])
            continue
        w_c = left_vecs[:, cluster].conj().T  # (m, n)
        v_c = eigvecs[:, cluster]  # (n, m)
        dual_c = np.linalg.solve(
            w_c @ v_c, w_c
        )  # (m, n) = rows of V^-1 for the cluster
        pos = int(np.flatnonzero(cluster == i)[0])
        rows[k] = dual_c[pos]
    return rows


def _build_reduced(
    system: StateSpace,
    eigvals: npt.NDArray[np.complex128],
    eigvecs: npt.NDArray[np.complex128],
    left_vecs: npt.NDArray[np.complex128],
    selected: npt.NDArray[np.int_],
    *,
    stabilize: bool,
) -> ModalReduction:
    """Assemble the real block-diagonal reduced system over the selected modes."""
    c = system.c
    assert c is not None  # guaranteed by the caller
    b = system.b
    # Reject defective (Jordan-chain) repeated modes up front: they cannot be realized
    # as independent modal blocks, regardless of whether an input matrix is present.
    _reject_defective_modes(eigvals, eigvecs, selected)
    dual = _dual_rows(eigvals, eigvecs, left_vecs, selected)

    n_modes = selected.size
    n_states = 2 * n_modes
    a_r = np.zeros((n_states, n_states), dtype=np.float64)
    c_r = np.zeros((c.shape[0], n_states), dtype=np.float64)
    b_r = None if b is None else np.zeros((n_states, b.shape[1]), dtype=np.float64)
    freqs = np.empty(n_modes, dtype=np.float64)
    damping = np.empty(n_modes, dtype=np.float64)
    state_projection = np.empty((n_states, system.n_states), dtype=np.float64)
    state_reconstruction = np.empty((system.n_states, n_states), dtype=np.float64)
    stabilized = np.zeros(n_modes, dtype=np.bool_)

    for k, i in enumerate(selected):
        lam = eigvals[i]
        omega = float(lam.imag)
        sigma = float(lam.real)
        if stabilize and sigma > 0.0:
            sigma = -sigma
            stabilized[k] = True
        block = 2 * k
        a_r[block, block], a_r[block, block + 1] = sigma, -omega
        a_r[block + 1, block], a_r[block + 1, block + 1] = omega, sigma
        cv = c @ eigvecs[:, i]
        c_r[:, block] = 2.0 * cv.real
        c_r[:, block + 1] = -2.0 * cv.imag
        state_projection[block] = dual[k].real
        state_projection[block + 1] = dual[k].imag
        state_reconstruction[:, block] = 2.0 * eigvecs[:, i].real
        state_reconstruction[:, block + 1] = -2.0 * eigvecs[:, i].imag
        if b_r is not None and b is not None:
            wb = dual[k] @ b
            b_r[block] = wb.real
            b_r[block + 1] = wb.imag
        magnitude = abs(lam)
        freqs[k] = magnitude / _TWO_PI  # natural frequency |lambda|
        # Use the post-stabilization sigma so the reported damping matches the realized
        # A-block: a stabilized mode (sigma flipped negative) reports positive damping,
        # not the original unstable value. |lambda| is unchanged by the sign flip.
        damping[k] = -sigma / magnitude if magnitude > 0.0 else 0.0

    state_names = [f"modal_{k}_{part}" for k in range(n_modes) for part in ("re", "im")]
    reduced = StateSpace(
        a=a_r,
        b=b_r,
        c=c_r,
        d=system.d,
        state_names=state_names,
        input_names=list(system.input_names),
        output_names=list(system.output_names),
        dt=None,
        output_mbc_coordinates=list(system.output_mbc_coordinates),
    )
    return ModalReduction(
        system=reduced,
        frequencies_hz=freqs,
        damping_ratios=damping,
        residues=np.empty(n_modes, dtype=np.float64),
        energy_retained=0.0,
        n_full_states=system.a.shape[0],
        n_stabilized=int(np.count_nonzero(stabilized)),
        state_projection=state_projection,
        state_reconstruction=state_reconstruction,
        stabilized=stabilized,
    )
