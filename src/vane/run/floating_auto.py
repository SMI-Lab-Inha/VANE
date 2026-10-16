# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Set up a floating-turbine settle automatically: controller library and damping.

A floating operating point is found by settling the fully coupled model and then
linearizing at the settled mean (see :func:`~vane.run.solve_steady_state`). Two
inputs of that settle used to come from the user; this module derives both from the
deck itself.

**The controller library.** Published decks name the controller library of the
machine that generated them (often a Linux ``libdiscon.so``).
:func:`find_controller_library` looks for a library this host can load: the deck's
own ``DLL_FileName``; the same library with another platform or bitness suffix
(``.dll``, ``_x64.dll``, ``.so``, ...) next to it, next to the ServoDyn file and next
to the primary deck; then in the directories or files listed by the caller, by the
``VANE_CONTROLLER_LIBRARIES`` environment variable (``os.pathsep``-separated) and by
the configuration file ``controller_libraries.txt`` (one entry per line, ``#``
comments) in VANE's configuration directory (``VANE_CONFIG_DIR``, else
``%APPDATA%\vane`` on Windows and ``$XDG_CONFIG_HOME/vane`` or ``~/.config/vane``
elsewhere). Candidates are matched by the library's name with the ``lib`` prefix,
the bitness tag and the suffix ignored, so ``libdiscon.so`` matches ``libdiscon.dll``,
``discon_x64.dll`` or ``DISCON.dll``. When none is found the settle is not
abandoned: :func:`plan_floating_settle` falls back to a **prescribed-rotor settle**
(no controller; the rotor turns at the operating curve's speed with the blades at
its pitch while the platform finds its equilibrium), and the point is then
linearized by CalcSteady from that equilibrium.

**The settle damping.** The two-stage settle accelerator
(:class:`~vane.run.SettlePlatformDamping`) needs, per platform DOF, the effective
mass and the natural period. :func:`probe_platform` measures both from the model:
it runs two single-snapshot standstill linearizations of the platform (still air and
still water, rotor parked, aerodynamics and controller off, radiation memory off so
no state-space hydrodynamic fit is needed), the second with a known damping added to
HydroDyn's ``AddBLin``. The difference of the two state matrices is
``-M^-1 Delta_B`` -- the platform columns of the exact inverse mass matrix, including
the hydrodynamic added mass -- and first-order eigenvalue perturbation of the
undamped state matrix gives, for every mode, how fast a damper on each platform DOF
damps it. For each DOF the mode it damps the most is its platform mode; that mode's
natural period, and the mass ``m = 1 / (2 dsigma/dB)`` for which a single-DOF
oscillator would be damped at the same rate, size the damping as a fraction of
critical (:meth:`PlatformProbe.settle_damping`). The probe costs two OpenFAST runs of
a fraction of a second of simulated time and needs no user data.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.linalg

from vane.io.fst_reader import _fortran_true
from vane.io.fst_writer import FastInputError, FastInputFile
from vane.io.lin_reader import read_lin_file
from vane.run.deck_files import copy_template_files
from vane.run.lin_config import (
    LinearizationSettings,
    _deck_float,
    _keep_output_start_before_end,
    configure_linearization,
)
from vane.run.operating_points import OperatingPoint
from vane.run.runner import OpenFastError, run_openfast
from vane.run.steady_state import (
    SettlePlatformDamping,
    _active_servodyn,
    _add_platform_damping,
    _hydrodyn_with_damping_matrix,
    _loads_on_host,
    _replace_file,
    _uses_controller_library,
    check_controller_library,
)
from vane.utils.fs_retry import rmtree_with_retry
from vane.utils.paths import resolve_path

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from vane.run.deck_files import TemplateCopyMode

__all__ = [
    "CONTROLLER_LIBRARIES_ENV",
    "ControllerLibraryResolution",
    "ControllerLibraryStatus",
    "FloatingSettlePlan",
    "PlatformProbe",
    "PlatformProbeError",
    "controller_library_search_paths",
    "find_controller_library",
    "is_floating_deck",
    "plan_floating_settle",
    "platform_probe_from_jacobians",
    "probe_platform",
]

logger = logging.getLogger(__name__)

#: Environment variable listing controller libraries or directories to search.
CONTROLLER_LIBRARIES_ENV = "VANE_CONTROLLER_LIBRARIES"
#: Environment variable overriding VANE's configuration directory.
CONFIG_DIR_ENV = "VANE_CONFIG_DIR"
#: Name of the controller-library list in VANE's configuration directory.
CONTROLLER_LIBRARIES_FILE = "controller_libraries.txt"

#: How :func:`find_controller_library` resolved the settle's controller library.
ControllerLibraryStatus = Literal["not_used", "deck", "found", "user", "missing"]

_LIBRARY_SUFFIXES = (".dll", ".so", ".dylib")
# Bitness/platform tags controller builds append to the library stem.
_LIBRARY_TAGS = ("_x64", "_x86", "_win64", "_win32", "_64", "_32")


def _host_library_suffix() -> str:
    """Return the shared-library suffix this host loads."""
    if sys.platform == "win32":
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


def _library_key(name: str) -> str:
    """Return a library file name reduced to its identity (``libDISCON_x64.dll``).

    The suffix, a bitness tag and a ``lib`` prefix are dropped and the case folded,
    so the builds of one controller for different platforms share a key.
    """
    stem = name.lower()
    for suffix in _LIBRARY_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for tag in _LIBRARY_TAGS:
        if stem.endswith(tag):
            stem = stem[: -len(tag)]
            break
    if stem.startswith("lib") and len(stem) > 3:
        stem = stem[3:]
    return stem


def _loadable(path: Path) -> bool:
    """Return whether ``path`` is an existing library this host can load."""
    if not path.is_file():
        return False
    suffix = path.suffix.lower()
    if suffix in _LIBRARY_SUFFIXES and suffix != _host_library_suffix():
        return False
    return _loads_on_host(path)


@dataclass(frozen=True)
class ControllerLibraryResolution:
    """Where a settle run's controller library comes from.

    Parameters
    ----------
    status : {"not_used", "deck", "found", "user", "missing"}
        ``"not_used"``: ServoDyn runs no controller library. ``"deck"``: the deck's
        own ``DLL_FileName`` is loadable here. ``"found"``: a host build was found
        (``library``). ``"user"``: the caller supplied it. ``"missing"``: none could
        be found; a controller settle is impossible on this host.
    library : pathlib.Path or None
        The library to write into the settle copy's ``DLL_FileName``; ``None``
        unless ``status`` is ``"found"`` or ``"user"``.
    deck_library : str
        The deck's ``DLL_FileName`` as written (empty when not used).
    source : str
        Where ``library`` was found (e.g. ``"next to the deck's library"``,
        ``"VANE_CONTROLLER_LIBRARIES"``); empty when not searched.
    searched : tuple of pathlib.Path
        The directories and files that were searched, in order.
    """

    status: ControllerLibraryStatus
    library: Path | None = None
    deck_library: str = ""
    source: str = ""
    searched: tuple[Path, ...] = ()

    @property
    def available(self) -> bool:
        """Return whether a settle can run the deck's controller on this host."""
        return self.status != "missing"

    def to_dict(self) -> dict[str, object]:
        """Return the resolution as a JSON-ready dictionary.

        Returns
        -------
        dict[str, object]
            ``status``, ``library``, ``deck_library``, ``source`` and ``searched``
            (paths as strings).
        """
        return {
            "status": self.status,
            "library": None if self.library is None else str(self.library),
            "deck_library": self.deck_library,
            "source": self.source,
            "searched": [str(path) for path in self.searched],
        }


def _config_dir(environ: Mapping[str, str]) -> Path:
    """Return VANE's configuration directory for this user."""
    configured = environ.get(CONFIG_DIR_ENV)
    if configured:
        return Path(configured)
    if sys.platform == "win32" and environ.get("APPDATA"):
        return Path(environ["APPDATA"]) / "vane"
    xdg = environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "vane"
    return Path.home() / ".config" / "vane"


def controller_library_search_paths(
    environ: Mapping[str, str] | None = None,
) -> tuple[tuple[Path, str], ...]:
    """Return the configured controller-library locations and where each is from.

    Parameters
    ----------
    environ : Mapping[str, str] or None, optional
        The environment to read; ``None`` uses :data:`os.environ`.

    Returns
    -------
    tuple of (pathlib.Path, str)
        Each configured file or directory with its origin: the
        ``VANE_CONTROLLER_LIBRARIES`` entries (``os.pathsep``-separated) first, then
        the lines of ``controller_libraries.txt`` in VANE's configuration directory
        (blank lines and ``#`` comments ignored; relative entries are relative to
        that file).
    """
    env = os.environ if environ is None else environ
    entries: list[tuple[Path, str]] = []
    for raw in env.get(CONTROLLER_LIBRARIES_ENV, "").split(os.pathsep):
        item = raw.strip().strip('"')
        if item:
            entries.append((Path(item), CONTROLLER_LIBRARIES_ENV))
    config = _config_dir(env) / CONTROLLER_LIBRARIES_FILE
    if config.is_file():
        for line in config.read_text(encoding="utf-8").splitlines():
            item = line.split("#", 1)[0].strip().strip('"')
            if item:
                path = Path(item)
                if not path.is_absolute():
                    path = config.parent / path
                entries.append((path, str(config)))
    return tuple(entries)


def _matching_library(location: Path, key: str) -> Path | None:
    """Return a loadable library at ``location`` (a file or directory) for ``key``."""
    if location.is_file():
        if _library_key(location.name) == key and _loadable(location):
            return location
        return None
    if not location.is_dir():
        return None
    host = _host_library_suffix()
    matches = sorted(
        entry
        for entry in location.iterdir()
        if entry.suffix.lower() == host
        and _library_key(entry.name) == key
        and _loadable(entry)
    )
    if not matches:
        return None

    # Prefer the plain name, then the 64-bit tags (a 64-bit OpenFAST cannot load a
    # 32-bit library).
    def rank(entry: Path) -> tuple[int, str]:
        stem = entry.stem.lower()
        if any(stem.endswith(tag) for tag in ("_x86", "_win32", "_32")):
            return (2, entry.name)
        if any(stem.endswith(tag) for tag in _LIBRARY_TAGS):
            return (1, entry.name)
        return (0, entry.name)

    return min(matches, key=rank)


def find_controller_library(
    fst_path: str | Path,
    *,
    search_paths: Sequence[str | Path] = (),
    environ: Mapping[str, str] | None = None,
) -> ControllerLibraryResolution:
    """Find a controller library this host can load for a settle run.

    Nothing is written; the result says what the settle should use.

    Parameters
    ----------
    fst_path : str or pathlib.Path
        The primary ``.fst`` deck.
    search_paths : Sequence of str or pathlib.Path, optional
        Extra library files or directories to search, after the deck's own
        locations and before the configured ones.
    environ : Mapping[str, str] or None, optional
        The environment holding ``VANE_CONTROLLER_LIBRARIES`` / ``VANE_CONFIG_DIR``;
        ``None`` uses :data:`os.environ`.

    Returns
    -------
    ControllerLibraryResolution
        ``"not_used"`` when ServoDyn runs no controller library, ``"deck"`` when the
        deck's own library loads here, ``"found"`` with the host build to use, or
        ``"missing"``.

    Notes
    -----
    The search order is: the deck's ``DLL_FileName``; then, for the same library
    name (the ``lib`` prefix, a bitness tag and the suffix ignored, case folded),
    the directory of the deck's library, the ServoDyn file's directory and the
    primary deck's directory; then ``search_paths``; then ``VANE_CONTROLLER_LIBRARIES``
    and the configuration file (see :func:`controller_library_search_paths`). Only
    libraries with this host's suffix (``.dll`` on Windows, ``.so`` on Linux,
    ``.dylib`` on macOS) are accepted; within a directory a plain or 64-bit build is
    preferred to a 32-bit one. A found library must still be compatible with the
    deck's controller input file (``DLL_InFile``); VANE cannot check that.
    """
    fst = Path(fst_path)
    servo = _active_servodyn(FastInputFile.read(fst))
    if servo is None or not _uses_controller_library(servo):
        return ControllerLibraryResolution(status="not_used")
    deck_value = (
        " ".join(servo.tokens("DLL_FileName")).strip().strip("\"'")
        if "DLL_FileName" in servo
        else ""
    )
    deck_library: Path | None
    try:
        deck_library = servo.resolve("DLL_FileName")
    except FastInputError:
        deck_library = None
    if deck_library is not None and _loadable(deck_library):
        return ControllerLibraryResolution(
            status="deck", deck_library=deck_value, searched=(deck_library,)
        )
    key = _library_key(Path(deck_value.replace("\\", "/")).name) if deck_value else ""
    locations: list[tuple[Path, str]] = []
    if deck_library is not None:
        locations.append((deck_library.parent, "next to the deck's library"))
    locations.append((servo.path.parent, "next to the ServoDyn file"))
    locations.append((fst.parent, "next to the primary deck"))
    locations.extend((Path(path), "search_paths") for path in search_paths)
    locations.extend(controller_library_search_paths(environ))
    searched: list[Path] = []
    seen: set[str] = set()
    for location, source in locations:
        marker = os.path.normcase(str(location))
        if marker in seen:
            continue
        seen.add(marker)
        searched.append(location)
        if not key:
            continue
        match = _matching_library(location, key)
        if match is not None:
            library = resolve_path(match)
            logger.info(
                "controller library %r is not loadable on this host; using %s (%s)",
                deck_value,
                library,
                source,
            )
            return ControllerLibraryResolution(
                status="found",
                library=library,
                deck_library=deck_value,
                source=source,
                searched=tuple(searched),
            )
    return ControllerLibraryResolution(
        status="missing", deck_library=deck_value, searched=tuple(searched)
    )


# ElastoDyn platform DOF switches (any one True makes the platform move).
_PLATFORM_DOF_FLAGS = (
    "PtfmSgDOF",
    "PtfmSwDOF",
    "PtfmHvDOF",
    "PtfmRDOF",
    "PtfmPDOF",
    "PtfmYDOF",
)


def is_floating_deck(fst_path: str | Path) -> bool:
    """Return whether a deck models a floating platform.

    A deck is floating when ElastoDyn frees at least one platform DOF and the platform
    is moored, or is carried by HydroDyn without a substructure module. A substructure
    (``CompSub != 0``) uses the platform DOFs too: without mooring it is a fixed-bottom
    monopile or jacket, with mooring a flexible floater modelled in SubDyn.

    Parameters
    ----------
    fst_path : str or pathlib.Path
        The primary ``.fst`` deck.

    Returns
    -------
    bool
        Whether the platform floats.
    """
    try:
        deck = FastInputFile.read(fst_path)
        elastodyn = FastInputFile.read(deck.resolve("EDFile"))
    except (OSError, FastInputError):
        return False
    free = any(
        flag in elastodyn and _fortran_true(elastodyn.get(flag))
        for flag in _PLATFORM_DOF_FLAGS
    )
    substructure = (_deck_float(deck, "CompSub") or 0.0) != 0.0
    mooring = (_deck_float(deck, "CompMooring") or 0.0) != 0.0
    hydro = (_deck_float(deck, "CompHydro") or 0.0) != 0.0
    return free and (mooring or (hydro and not substructure))


class PlatformProbeError(RuntimeError):
    """Raised when the standstill platform probe cannot measure the platform."""


# The six platform DOFs and their ElastoDyn internal DOF tags in .lin descriptions.
_PLATFORM_DOF_TAGS = {
    "surge": "DOF_Sg",
    "sway": "DOF_Sw",
    "heave": "DOF_Hv",
    "roll": "DOF_R",
    "pitch": "DOF_P",
    "yaw": "DOF_Y",
}

#: Probe damping added in the second standstill linearization (N s/m, N m s/rad).
DEFAULT_PROBE_DAMPING = SettlePlatformDamping(
    surge=1.0e6, sway=1.0e6, heave=1.0e6, roll=1.0e9, pitch=1.0e9, yaw=1.0e9
)


@dataclass(frozen=True)
class PlatformProbe:
    """Platform dynamics measured by two standstill linearizations.

    Parameters
    ----------
    effective_mass : dict[str, float]
        Per platform DOF, the mass (kg; kg m^2 for rotations) of a single-DOF
        oscillator that a damper on that DOF would damp at the same rate as it damps
        the DOF's platform mode: ``1 / (2 dsigma/dB)``. It includes the
        hydrodynamic added mass and the coupling to the other DOFs.
    natural_period : dict[str, float]
        Per DOF, the undamped natural period (s) of its platform mode, ``2 pi /
        |lambda|``.
    damping_ratio : dict[str, float]
        Per DOF, the damping ratio the platform mode already has at standstill.
    mode_frequency : dict[str, float]
        Per DOF, the damped frequency (Hz) of its platform mode.
    probe_damping : SettlePlatformDamping
        The damping added in the second linearization.
    n_states : int
        Number of continuous states of the standstill linearization.
    """

    effective_mass: dict[str, float]
    natural_period: dict[str, float]
    damping_ratio: dict[str, float]
    mode_frequency: dict[str, float]
    probe_damping: SettlePlatformDamping
    n_states: int

    @property
    def slowest_period(self) -> float | None:
        """Return the longest finite natural period (s) of the measured modes.

        It sizes the undamped confirmation of a two-stage settle (``platform_period``
        of :func:`~vane.run.solve_steady_state`): the confirmation must span several
        periods of the slowest platform mode. Standstill periods are the right
        scale; under thrust the mooring stiffens and the periods shorten slightly
        (about 125 s at standstill against about 106 s at rated for the IEA 15-MW
        semi-submersible's surge), so sizing from them errs long.

        Returns
        -------
        float or None
            The largest positive, finite period; ``None`` when there is none.
        """
        periods = [
            period
            for period in self.natural_period.values()
            if math.isfinite(period) and period > 0.0
        ]
        return max(periods) if periods else None

    def settle_damping(self, fraction: float = 0.5) -> SettlePlatformDamping:
        """Return the settle damping as a fraction of critical per measured DOF.

        Parameters
        ----------
        fraction : float, optional
            Fraction of critical damping to add to every measured DOF (on top of
            the damping the mode already has). Default ``0.5``.

        Returns
        -------
        SettlePlatformDamping
            ``4 pi fraction m / T`` per DOF (see
            :meth:`SettlePlatformDamping.from_critical_fraction`).
        """
        return SettlePlatformDamping.from_critical_fraction(
            fraction, mass=self.effective_mass, natural_period=self.natural_period
        )

    def to_dict(self) -> dict[str, object]:
        """Return the probe as a JSON-ready dictionary.

        Returns
        -------
        dict[str, object]
            The per-DOF measurements, the probe damping, the state count and the
            slowest period.
        """
        return {
            "effective_mass": dict(self.effective_mass),
            "natural_period": dict(self.natural_period),
            "damping_ratio": dict(self.damping_ratio),
            "mode_frequency": dict(self.mode_frequency),
            "probe_damping": dict(
                zip(_PLATFORM_DOF_TAGS, self.probe_damping.diagonal(), strict=True)
            ),
            "n_states": self.n_states,
            "slowest_period": self.slowest_period,
        }


def _platform_state_indices(
    descriptions: Sequence[str],
) -> dict[str, tuple[int, int]]:
    """Return ``{dof: (displacement, velocity)}`` state indices of the platform DOFs."""
    indices: dict[str, tuple[int, int]] = {}
    for dof, tag in _PLATFORM_DOF_TAGS.items():
        marker = f"= {tag})"
        displacement = velocity = None
        for index, text in enumerate(descriptions):
            if marker not in text or not text.lstrip().startswith("ED"):
                continue
            if "First time derivative" in text:
                velocity = index if velocity is None else velocity
            else:
                displacement = index if displacement is None else displacement
        if displacement is not None and velocity is not None:
            indices[dof] = (displacement, velocity)
    return indices


def _cluster_sensitivity(
    eigenvalues: npt.NDArray[np.complex128],
    left: npt.NDArray[np.complex128],
    right: npt.NDArray[np.complex128],
    minv_column: npt.NDArray[np.float64],
    velocity: int,
    mode: int,
) -> float:
    """Return ``-Re`` of the summed eigenvalue sensitivity of ``mode``'s cluster.

    A repeated eigenvalue (a symmetric platform's surge/sway or roll/pitch) has an
    arbitrary eigenvector basis, and its left and right eigenvectors need not pair
    up, but the sum of the cluster's eigenvalue changes is basis-independent:
    ``trace((W^H U)^-1 W^H dA U)`` with ``dA = -M^-1[:, v] e_v^T``.
    """
    magnitude = abs(eigenvalues[mode])
    cluster = np.flatnonzero(
        np.abs(eigenvalues - eigenvalues[mode]) <= 1e-6 * magnitude
    )
    w = left[:, cluster]
    u = right[:, cluster]
    gram = w.conj().T @ u
    change = -np.outer(w.conj().T @ minv_column, u[velocity, :])
    try:
        total = complex(np.trace(np.linalg.solve(gram, change)))
    except np.linalg.LinAlgError:
        return float("nan")
    return float(-total.real)


def platform_probe_from_jacobians(
    a_base: npt.NDArray[np.float64],
    a_damped: npt.NDArray[np.float64],
    descriptions: Sequence[str],
    probe_damping: SettlePlatformDamping = DEFAULT_PROBE_DAMPING,
) -> PlatformProbe:
    r"""Measure the platform from two state matrices differing by a platform damping.

    Parameters
    ----------
    a_base : npt.NDArray[np.float64]
        Full-system state matrix ``A`` of the standstill linearization.
    a_damped : npt.NDArray[np.float64]
        The same with ``probe_damping`` added to HydroDyn's ``AddBLin``.
    descriptions : Sequence[str]
        The continuous-state descriptions (``.lin`` "Order of continuous states").
    probe_damping : SettlePlatformDamping, optional
        The damping that was added.

    Returns
    -------
    PlatformProbe
        The per-DOF effective mass, natural period and damping ratio of each
        platform DOF that is free in the model and has an oscillatory mode.

    Raises
    ------
    PlatformProbeError
        If the matrices disagree in shape with the descriptions, no ElastoDyn
        platform DOF is free, or the probe damping did not change ``A``.

    Notes
    -----
    Adding a damping ``b_j`` on platform DOF ``j`` changes only the columns of the
    platform velocities: ``Delta A[:, v_j] = -M^{-1}[:, v_j] b_j``. For each mode
    ``k`` with right and left eigenvectors ``u_k`` and ``w_k`` of ``a_base``, the
    first-order sensitivity of its eigenvalue to ``b_j`` is

    .. math::

        \frac{\partial \lambda_k}{\partial b_j} = -\frac{(w_k^H M^{-1}_{:,v_j})
        \, u_{k, v_j}}{w_k^H u_k}

    The platform mode of DOF ``j`` is the oscillatory mode whose decay rate
    ``-Re lambda`` the damper raises the most; a single-DOF oscillator of mass ``m``
    has ``d(-Re lambda)/db = 1/(2m)``, which defines the effective mass.
    """
    a0 = np.asarray(a_base, dtype=np.float64)
    a1 = np.asarray(a_damped, dtype=np.float64)
    n_states = len(descriptions)
    if a0.shape != (n_states, n_states) or a1.shape != a0.shape:
        msg = (
            f"state matrices {a0.shape} and {a1.shape} do not match the "
            f"{n_states} state descriptions"
        )
        raise PlatformProbeError(msg)
    indices = _platform_state_indices(descriptions)
    if not indices:
        msg = "the standstill linearization has no free ElastoDyn platform DOF"
        raise PlatformProbeError(msg)
    added = dict(zip(_PLATFORM_DOF_TAGS, probe_damping.diagonal(), strict=True))
    eigenvalues, left, right = scipy.linalg.eig(a0, left=True, right=True)
    oscillatory = np.flatnonzero(eigenvalues.imag > 1e-9)
    masses: dict[str, float] = {}
    periods: dict[str, float] = {}
    ratios: dict[str, float] = {}
    frequencies: dict[str, float] = {}
    for dof, (_, velocity) in indices.items():
        if added[dof] <= 0.0:
            continue
        minv_column = -(a1[:, velocity] - a0[:, velocity]) / added[dof]
        if not np.any(minv_column):
            continue
        # -Re(d lambda / d b): how much faster each mode decays per unit damping. It
        # is summed over each mode's cluster of repeated eigenvalues: the per-mode
        # value depends on the arbitrary eigenvector basis of a repeated root (whose
        # left and right eigenvectors need not even pair up), the cluster sum does not.
        sensitivity = np.full(eigenvalues.size, -np.inf)
        for index in oscillatory:
            value = _cluster_sensitivity(
                eigenvalues, left, right, minv_column, velocity, int(index)
            )
            if math.isfinite(value):
                sensitivity[index] = value
        mode = int(np.argmax(sensitivity))
        total = float(sensitivity[mode])
        if not math.isfinite(total) or total <= 0.0:
            continue
        eigenvalue = complex(eigenvalues[mode])
        magnitude = abs(eigenvalue)
        masses[dof] = float(1.0 / (2.0 * total))
        periods[dof] = float(2.0 * math.pi / magnitude)
        ratios[dof] = float(-eigenvalue.real / magnitude)
        frequencies[dof] = float(eigenvalue.imag / (2.0 * math.pi))
    if not masses:
        msg = (
            "the probe damping did not change the platform dynamics of the standstill "
            "linearization (is HydroDyn's AddBLin linearized?)"
        )
        raise PlatformProbeError(msg)
    return PlatformProbe(
        effective_mass=masses,
        natural_period=periods,
        damping_ratio=ratios,
        mode_frequency=frequencies,
        probe_damping=probe_damping,
        n_states=n_states,
    )


def _probe_case(
    fst_path: Path,
    template_root: Path,
    case: Path,
    template_copy: TemplateCopyMode,
    damping: SettlePlatformDamping | None,
) -> Path:
    """Copy and configure one standstill probe linearization; return its deck."""
    if case.exists():
        rmtree_with_retry(case)
    copy_template_files(fst_path, template_root, case, mode=template_copy)
    deck_path = case / resolve_path(fst_path).relative_to(resolve_path(template_root))
    settings = LinearizationSettings.offshore(
        # Radiation memory needs a state-space fit some decks lack; the
        # infinite-frequency added mass the probe needs is always present.
        hydro_radiation_mode=0,
        hydro_excitation_mode=0,
    )
    configure_linearization(
        deck_path, OperatingPoint(0.0, 0.0, 0.0), settings, confine_to=case
    )
    deck = FastInputFile.read(deck_path)
    # Structure, hydrostatics, added mass and mooring only: no aerodynamics, no
    # controller (whose library may not load here), linearized at t = 0.
    for keyword in ("CompAero", "CompServo", "CompInflow"):
        if keyword in deck:
            deck.set(keyword, 0)
    step = _deck_float(deck, "DT") or 0.01
    deck.set("TMax", max(0.1, 2.0 * step))
    _keep_output_start_before_end(deck)
    deck.write()
    if damping is not None:
        hydro = _hydrodyn_with_damping_matrix(deck_path)
        text = hydro.read_text(encoding="utf-8", errors="surrogateescape")
        _replace_file(
            hydro,
            _add_platform_damping(text, damping, hydro).encode(
                "utf-8", "surrogateescape"
            ),
        )
    return deck_path


def probe_platform(
    fst_path: str | Path,
    work_dir: str | Path,
    *,
    template_root: str | Path | None = None,
    template_copy: TemplateCopyMode = "referenced",
    executable: str | Path = "openfast",
    timeout: float | None = None,
    probe_damping: SettlePlatformDamping = DEFAULT_PROBE_DAMPING,
) -> PlatformProbe:
    """Measure a floating platform's effective masses and natural periods.

    Two single-snapshot standstill linearizations of a copy of the deck are run in
    ``work_dir`` (see the module notes): still air and water, rotor parked, AeroDyn
    and ServoDyn off, HydroDyn radiation memory and wave excitation off; the second
    with ``probe_damping`` added to HydroDyn's ``AddBLin``. The template deck is
    never edited.

    Parameters
    ----------
    fst_path : str or pathlib.Path
        The template primary ``.fst`` deck.
    work_dir : str or pathlib.Path
        Directory for the two probe cases (``base``/``damped``; existing
        ones are replaced).
    template_root : str or pathlib.Path or None, optional
        Directory the deck's files are copied from (default: the deck's directory).
    template_copy : {"referenced", "tree"}, optional
        How the template is copied (see :func:`~vane.run.copy_template_files`).
    executable : str or pathlib.Path, optional
        OpenFAST executable.
    timeout : float or None, optional
        Per-run timeout (s).
    probe_damping : SettlePlatformDamping, optional
        The known damping added in the second run.

    Returns
    -------
    PlatformProbe
        The measured platform dynamics.

    Raises
    ------
    PlatformProbeError
        If a probe run fails, writes no ``.lin`` file, or cannot measure the
        platform.
    ValueError
        If the deck has no active HydroDyn with an ``AddBLin`` matrix.
    """
    fst = Path(fst_path)
    root = Path(template_root) if template_root is not None else fst.parent
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    _hydrodyn_with_damping_matrix(fst)
    decks = [
        _probe_case(fst, root, work / "base", template_copy, None),
        _probe_case(fst, root, work / "damped", template_copy, probe_damping),
    ]

    def run(deck: Path) -> Path:
        result = run_openfast(deck, executable=executable, timeout=timeout)
        if not result.succeeded or not result.lin_files:
            tail = "\n".join(result.output.strip().splitlines()[-10:])
            msg = (
                f"standstill platform probe failed for {deck} (exit "
                f"{result.returncode}, {result.n_lin} .lin file(s)):\n{tail}"
            )
            raise PlatformProbeError(msg)
        return result.lin_files[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        base_lin, damped_lin = pool.map(run, decks)
    base = read_lin_file(base_lin)
    damped = read_lin_file(damped_lin)
    if base.a is None or damped.a is None:
        msg = "the standstill platform probe .lin files carry no state matrix"
        raise PlatformProbeError(msg)
    probe = platform_probe_from_jacobians(
        base.a, damped.a, base.x.descriptions, probe_damping
    )
    logger.info(
        "platform probe: %s",
        ", ".join(
            f"{dof} T={probe.natural_period[dof]:.1f} s "
            f"m={probe.effective_mass[dof]:.3g}"
            for dof in probe.natural_period
        ),
    )
    return probe


#: Where a floating settle plan's platform damping came from.
DampingSource = Literal["user", "probe", "none"]


@dataclass(frozen=True)
class FloatingSettlePlan:
    """How a sweep's settle runs are set up, decided from the deck and this host.

    Parameters
    ----------
    controller : ControllerLibraryResolution
        How the controller library was resolved.
    floating : bool
        Whether the deck models a floating platform (:func:`is_floating_deck`).
    platform_damping : SettlePlatformDamping or None
        The two-stage settle damping to use (``None``: a single-stage settle).
    damping_source : {"user", "probe", "none"}
        Whether the damping was given, sized from the platform probe, or is off.
    probe : PlatformProbe or None
        The platform probe, when one was run.
    damping_fraction : float
        Fraction of critical damping the probe-sized damping uses.
    notes : tuple of str
        Human-readable notes on every automatic decision (also logged).
    """

    controller: ControllerLibraryResolution
    floating: bool
    platform_damping: SettlePlatformDamping | None = None
    damping_source: DampingSource = "none"
    probe: PlatformProbe | None = None
    damping_fraction: float = 0.5
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def controller_library(self) -> Path | None:
        """Return the library to write into the settle copies (``None``: keep)."""
        return self.controller.library

    @property
    def prescribed_rotor(self) -> bool:
        """Return whether the settle must run without a controller (the fallback)."""
        return not self.controller.available

    def to_dict(self) -> dict[str, object]:
        """Return the plan as a JSON-ready dictionary.

        Returns
        -------
        dict[str, object]
            The controller resolution, floating flag, damping and its source, the
            probe and the notes.
        """
        damping = self.platform_damping
        return {
            "controller": self.controller.to_dict(),
            "floating": self.floating,
            "prescribed_rotor": self.prescribed_rotor,
            "platform_damping": None
            if damping is None
            else dict(zip(_PLATFORM_DOF_TAGS, damping.diagonal(), strict=True)),
            "damping_source": self.damping_source,
            "damping_fraction": self.damping_fraction,
            "probe": None if self.probe is None else self.probe.to_dict(),
            "notes": list(self.notes),
        }


def plan_floating_settle(
    fst_path: str | Path,
    work_dir: str | Path,
    *,
    controller_library: str | Path | None = None,
    platform_damping: SettlePlatformDamping | None = None,
    auto_damping: bool = True,
    damping_fraction: float = 0.5,
    controller_search_paths: Sequence[str | Path] = (),
    template_root: str | Path | None = None,
    template_copy: TemplateCopyMode = "referenced",
    executable: str | Path = "openfast",
    timeout: float | None = None,
) -> FloatingSettlePlan:
    """Decide, without user input, how a deck's settle runs are set up.

    1. **Controller library.** A given ``controller_library`` is checked and used.
       Otherwise :func:`find_controller_library` looks for a host build; when none
       exists the plan's :attr:`~FloatingSettlePlan.prescribed_rotor` is ``True``:
       the settle runs without a controller at the operating curve's rotor speed and
       pitch (see ``prescribed_rotor`` of :func:`~vane.run.solve_steady_state`), and
       the point is then linearized by CalcSteady from the platform equilibrium it
       finds. That needs an operating curve; the caller enforces it.
    2. **Settle damping.** A given ``platform_damping`` is used as is. Otherwise, for
       a floating deck with ``auto_damping``, :func:`probe_platform` measures the
       platform and the damping is ``damping_fraction`` of critical on every
       measured DOF. A probe that fails is logged and the settle runs undamped.

    Parameters
    ----------
    fst_path : str or pathlib.Path
        The template primary ``.fst`` deck.
    work_dir : str or pathlib.Path
        Directory for the platform probe cases.
    controller_library : str or pathlib.Path or None, optional
        A host controller library to use instead of searching.
    platform_damping : SettlePlatformDamping or None, optional
        A settle damping to use instead of probing.
    auto_damping : bool, optional
        Size the damping from the platform probe for a floating deck. Default
        ``True``.
    damping_fraction : float, optional
        Fraction of critical damping for the probe-sized damping (``0 < f``).
        Default ``0.5``.
    controller_search_paths : Sequence of str or pathlib.Path, optional
        Extra library locations (see :func:`find_controller_library`).
    template_root : str or pathlib.Path or None, optional
        The template copy root (see :func:`probe_platform`).
    template_copy : {"referenced", "tree"}, optional
        How the template is copied for the probe.
    executable : str or pathlib.Path, optional
        OpenFAST executable for the probe.
    timeout : float or None, optional
        Per-run timeout (s) of the probe.

    Returns
    -------
    FloatingSettlePlan
        The resolved controller library, damping and the evidence behind them.

    Raises
    ------
    FileNotFoundError
        If ``controller_library`` is given but does not exist.
    ValueError
        If ``damping_fraction`` is not positive and finite.
    """
    if not math.isfinite(damping_fraction) or damping_fraction <= 0.0:
        msg = f"damping_fraction must be positive and finite, got {damping_fraction}"
        raise ValueError(msg)
    fst = Path(fst_path)
    notes: list[str] = []
    if controller_library is not None:
        library = check_controller_library(fst, controller_library)
        controller = ControllerLibraryResolution(
            status="user" if library is not None else "not_used", library=library
        )
    else:
        controller = find_controller_library(fst, search_paths=controller_search_paths)
        if controller.status == "found":
            notes.append(
                f"controller library {controller.deck_library!r} is not loadable on "
                f"this host; the settle uses {controller.library} "
                f"({controller.source})"
            )
        elif controller.status == "missing":
            notes.append(
                f"no loadable controller library for {controller.deck_library!r} "
                f"(searched {len(controller.searched)} location(s); set "
                f"{CONTROLLER_LIBRARIES_ENV} to a directory holding a host build); "
                "no controller settle is possible, so each point settles with the "
                "rotor prescribed at the operating curve's speed and pitch and is "
                "linearized by CalcSteady from that platform equilibrium"
            )
    floating = is_floating_deck(fst)
    damping = platform_damping
    source: DampingSource = "user" if damping is not None else "none"
    probe: PlatformProbe | None = None
    if damping is None and floating and auto_damping:
        try:
            probe = probe_platform(
                fst,
                work_dir,
                template_root=template_root,
                template_copy=template_copy,
                executable=executable,
                timeout=timeout,
            )
        except (
            PlatformProbeError,
            OpenFastError,
            ValueError,
            OSError,
            FastInputError,
            np.linalg.LinAlgError,
        ) as error:
            # A probe run that times out or cannot start (OpenFastError), or an
            # eigensolver that fails, only costs the accelerator, like any probe error.
            notes.append(
                f"platform probe failed, so the settle runs without the damping "
                f"accelerator: {error}"
            )
        else:
            damping = probe.settle_damping(damping_fraction)
            source = "probe"
            notes.append(
                f"settle damping sized at {damping_fraction:g} of critical from the "
                "standstill platform probe: "
                + ", ".join(
                    f"{dof} T={probe.natural_period[dof]:.1f} s "
                    f"m={probe.effective_mass[dof]:.3g}"
                    for dof in probe.natural_period
                )
            )
    for note in notes:
        if "failed" in note or "no loadable" in note:
            logger.warning(note)
        else:
            logger.info(note)
    return FloatingSettlePlan(
        controller=controller,
        floating=floating,
        platform_damping=damping,
        damping_source=source,
        probe=probe,
        damping_fraction=damping_fraction,
        notes=tuple(notes),
    )
