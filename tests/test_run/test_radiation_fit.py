# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.run.radiation_fit -- state-space radiation fits from WAMIT data."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

from vane.run import radiation_fit as radiation_module
from vane.run.radiation_fit import (
    RadiationFitError,
    RadiationStateSpace,
    ensure_radiation_state_space,
    fit_radiation_state_space,
    preview_radiation_state_space,
    read_radiation_state_space,
    read_wamit_radiation,
    write_radiation_state_space,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

_RHO = 1025.0
_LENGTH = 2.0
_OMEGA = np.linspace(0.05, 5.0, 60)
_A_INF = np.diag([1.0e7, 1.0e7, 2.0e7, 5.0e9, 5.0e9, 8.0e9])
_A_ZERO = 1.2 * _A_INF


def _second_order(
    gain: float, natural: float, zeta: float, s: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """``gain * s / (s^2 + 2 zeta natural s + natural^2)``: K(0) = 0, proper."""
    return gain * s / (s**2 + 2 * zeta * natural * s + natural**2)


def _true_kernel(omega: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
    """Return a known kernel: diagonal entries and a surge-pitch coupling."""
    s = 1j * omega
    kernel = np.zeros((omega.size, 6, 6), dtype=np.complex128)
    for dof, (gain, natural, zeta) in enumerate(
        [
            (2.0e6, 1.0, 0.4),
            (2.0e6, 1.0, 0.4),
            (5.0e6, 0.8, 0.3),
            (1.0e9, 1.2, 0.5),
            (1.0e9, 1.2, 0.5),
            (2.0e9, 1.5, 0.6),
        ]
    ):
        kernel[:, dof, dof] = _second_order(gain, natural, zeta, s)
    kernel[:, 0, 4] = kernel[:, 4, 0] = _second_order(-3.0e7, 1.1, 0.45, s)
    # A negligible surge-heave coupling the threshold drops.
    kernel[:, 0, 2] = kernel[:, 2, 0] = _second_order(1.0, 1.0, 0.4, s)
    return kernel


def _scale(row: int, column: int) -> float:
    return _RHO * _LENGTH ** (3 + (row >= 3) + (column >= 3))


def _write_wamit(path: Path) -> None:
    """Write the true kernel as a non-dimensional WAMIT ``.1`` file."""
    kernel = _true_kernel(_OMEGA)
    lines = []
    for period, matrix in ((-1.0, _A_ZERO), (0.0, _A_INF)):
        for i in range(6):
            for j in range(6):
                if matrix[i, j] != 0.0:
                    value = matrix[i, j] / _scale(i, j)
                    lines.append(f"{period:14.6E} {i + 1:5d} {j + 1:5d} {value:.12E}")
    for k, omega in enumerate(_OMEGA):
        period = 2 * math.pi / omega
        for i in range(6):
            for j in range(6):
                value = kernel[k, i, j]
                added = _A_INF[i, j] + value.imag / omega
                damping = value.real
                if added == 0.0 and damping == 0.0:
                    continue
                a_bar = added / _scale(i, j)
                b_bar = damping / (_scale(i, j) * omega)
                lines.append(
                    f"{period:.12E} {i + 1:5d} {j + 1:5d} {a_bar:.12E} {b_bar:.12E}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_deck(
    tmp_path: Path,
    *,
    pot_mod: int = 1,
    n_body: int = 1,
    comp_hydro: int = 1,
    density_in: str = "fst",
) -> Path:
    (tmp_path / "HydroData").mkdir(parents=True, exist_ok=True)
    _write_wamit(tmp_path / "HydroData" / "body.1")
    hydro = [
        f"{pot_mod} PotMod - potential flow",
        "2 RdtnMod - radiation",
        f"{n_body} NBody - bodies",
        '"HydroData/body" PotFile - root',
        f"{_LENGTH} WAMITULEN - length",
    ]
    if density_in == "hydro":
        hydro.append(f"{_RHO} WtrDens - density")
    (tmp_path / "HD.dat").write_text("\n".join(hydro) + "\n", encoding="utf-8")
    (tmp_path / "SS.dat").write_text(
        f"{_RHO if density_in == 'seastate' else 0} WtrDens - density\n",
        encoding="utf-8",
    )
    primary = [
        f"{comp_hydro} CompHydro - hydro",
        '"HD.dat" HydroFile - HydroDyn',
        '"SS.dat" SeaStFile - SeaState',
    ]
    if density_in == "fst":
        primary.append(f"{_RHO} WtrDens - density")
    fst = tmp_path / "Model.fst"
    fst.write_text("\n".join(primary) + "\n", encoding="utf-8")
    return fst


class TestReadWamitRadiation:
    """Reading and dimensionalising a WAMIT ``.1`` file."""

    def test_dimensional_values(self, tmp_path: Path) -> None:
        """Added mass, damping and the infinite-frequency limit are recovered."""
        path = tmp_path / "body.1"
        _write_wamit(path)
        radiation = read_wamit_radiation(path, water_density=_RHO, length_scale=_LENGTH)
        np.testing.assert_allclose(radiation.frequencies, _OMEGA, rtol=1e-9)
        np.testing.assert_allclose(radiation.added_mass_infinite, _A_INF, rtol=1e-6)
        np.testing.assert_allclose(
            radiation.kernel(), _true_kernel(_OMEGA), rtol=1e-6, atol=1e-6
        )

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("0.0 1 1\n", "malformed"),
            ("0.0 1 7 1.0\n", "single body"),
            ("1.0 1 1 1.0 1.0\n", "infinite-frequency"),
            ("0.0 1 1 1.0\n1.0 1 1 1.0 1.0\n", "at least 4"),
        ],
    )
    def test_rejects_bad_files(self, tmp_path: Path, text: str, match: str) -> None:
        """Malformed, multi-body, incomplete or too-short data are rejected."""
        path = tmp_path / "bad.1"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(RadiationFitError, match=match):
            read_wamit_radiation(path, water_density=_RHO)

    @pytest.mark.parametrize("density", [0.0, -1.0, float("nan")])
    def test_rejects_bad_density(self, tmp_path: Path, density: float) -> None:
        """The density must be positive and finite."""
        with pytest.raises(RadiationFitError, match="water_density"):
            read_wamit_radiation(tmp_path / "x.1", water_density=density)


class TestFitRadiationStateSpace:
    """Vector fitting of the radiation kernel."""

    def _model(self, tmp_path: Path) -> RadiationStateSpace:
        path = tmp_path / "body.1"
        _write_wamit(path)
        radiation = read_wamit_radiation(path, water_density=_RHO, length_scale=_LENGTH)
        return fit_radiation_state_space(radiation, tolerance=1e-4)

    def test_recovers_a_known_kernel(self, tmp_path: Path) -> None:
        """A rational kernel is reproduced; the force is ``-K``; K(0) = 0."""
        model = self._model(tmp_path)
        entries = {(pair.row, pair.column) for pair in model.pairs}
        assert entries == {
            (0, 0),
            (4, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (0, 4),
            (4, 4),
            (5, 5),
        }
        assert model.max_relative_error < 1e-3
        omega = np.linspace(0.02, 8.0, 50)
        truth = _true_kernel(omega)
        response = model.force_response(omega)
        for pair in model.pairs:
            i, j = pair.row, pair.column
            np.testing.assert_allclose(
                -response[:, i, j],
                truth[:, i, j],
                rtol=5e-3,
                atol=1e-3 * abs(truth[:, i, j]).max(),
            )
            assert (
                abs(pair.evaluate(np.array([1e-9]))[0])
                < 1e-6 * abs(truth[:, i, j]).max()
            )
        assert np.all(np.linalg.eigvals(model.a).real < 0.0)
        assert sum(model.states_per_dof) == model.n_states
        # Blocks are grouped by input DOF: only that DOF's column of B is used.
        offset = 0
        for dof, count in enumerate(model.states_per_dof):
            block = model.b[offset : offset + count]
            assert np.all(np.delete(block, dof, axis=1) == 0.0)
            offset += count

    def test_write_read_round_trip(self, tmp_path: Path) -> None:
        """A written ``.ss`` file reads back to the same matrices."""
        model = self._model(tmp_path)
        path = write_radiation_state_space(model, tmp_path / "out" / "body.ss")
        text = path.read_text(encoding="utf-8").splitlines()
        assert text[1].startswith("1 1 1 1 1 1")
        assert int(text[2].split()[0]) == model.n_states
        back = read_radiation_state_space(path)
        np.testing.assert_allclose(back.a, model.a, rtol=1e-8)
        np.testing.assert_allclose(back.b, model.b, rtol=1e-8)
        np.testing.assert_allclose(back.c, model.c, rtol=1e-8)
        assert back.states_per_dof == model.states_per_dof
        assert math.isnan(back.max_relative_error)

    @pytest.mark.parametrize("counts", ["1 0 0 0 0", "2 -1 0 0 0 0", "0 0 0 0 0 0"])
    def test_read_rejects_inconsistent_state_counts(
        self, tmp_path: Path, counts: str
    ) -> None:
        """Per-DOF counts must be six non-negative numbers adding up to the states.

        The matrix sizes alone matched, so a file with too few, negative or
        miscounted entries assigned radiation states to the wrong platform DOF.
        """
        model = self._model(tmp_path)
        path = write_radiation_state_space(model, tmp_path / "out" / "body.ss")
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[3] = f"{counts}\t%Radiation States per DoF"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(RadiationFitError, match="per DOF"):
            read_radiation_state_space(path)

    def test_written_file_never_writes_through_a_hard_link(
        self, tmp_path: Path
    ) -> None:
        """A hard-linked ``.ss`` in a case copy is replaced, not overwritten."""
        model = self._model(tmp_path)
        template = tmp_path / "template.ss"
        template.write_text("template\n", encoding="utf-8")
        link = tmp_path / "case.ss"
        os.link(template, link)
        write_radiation_state_space(model, link)
        assert template.read_text(encoding="utf-8") == "template\n"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"tolerance": 0.0}, "tolerance"),
            ({"max_states_per_entry": 1}, "max_states_per_entry"),
        ],
    )
    def test_rejects_bad_arguments(
        self, tmp_path: Path, kwargs: dict[str, float], match: str
    ) -> None:
        """Fit options are validated."""
        path = tmp_path / "body.1"
        _write_wamit(path)
        radiation = read_wamit_radiation(path, water_density=_RHO)
        with pytest.raises(RadiationFitError, match=match):
            fit_radiation_state_space(radiation, **kwargs)  # type: ignore[arg-type]

    def test_zero_data_cannot_be_fitted(self, tmp_path: Path) -> None:
        """All-zero radiation data has nothing to fit."""
        path = tmp_path / "zero.1"
        rows = ["0.0 1 1 1.0"] + [f"{p:.3f} 1 1 1.0 0.0" for p in (1, 2, 3, 4)]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        radiation = read_wamit_radiation(path, water_density=_RHO)
        with pytest.raises(RadiationFitError, match="zero"):
            fit_radiation_state_space(radiation)

    @pytest.mark.parametrize(
        "text",
        ["header\n1 1 1 1 1 1\nx\n", "h\n1 1 1 1 1 1\n1\n1 0 0 0 0 0\n0.0\n"],
    )
    def test_read_rejects_malformed(self, tmp_path: Path, text: str) -> None:
        """A malformed or inconsistent ``.ss`` file is rejected."""
        path = tmp_path / "bad.ss"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(RadiationFitError):
            read_radiation_state_space(path)


class TestEnsureRadiationStateSpace:
    """Generating a deck's missing ``.ss`` file."""

    @pytest.mark.parametrize("density_in", ["fst", "hydro", "seastate"])
    def test_generates_once(self, tmp_path: Path, density_in: str) -> None:
        """The missing file is written next to the WAMIT data, then left alone."""
        fst = _make_deck(tmp_path, density_in=density_in)
        record = ensure_radiation_state_space(fst)
        assert record is not None
        assert record.written
        assert record.path == tmp_path / "HydroData" / "body.ss"
        assert record.path.is_file()
        assert record.max_relative_error < 0.05
        assert set(record.entry_errors) >= {"K11", "K15", "K51", "K66"}
        assert ensure_radiation_state_space(fst) is None
        assert ensure_radiation_state_space(fst, overwrite=True) is not None

    @pytest.mark.parametrize("options", [{"pot_mod": 0}, {"comp_hydro": 0}])
    def test_nothing_to_do(self, tmp_path: Path, options: dict[str, int]) -> None:
        """No potential flow, or HydroDyn off, needs no fit."""
        fst = _make_deck(tmp_path, **options)
        assert ensure_radiation_state_space(fst) is None
        assert not (tmp_path / "HydroData" / "body.ss").exists()

    def test_multiple_bodies_rejected(self, tmp_path: Path) -> None:
        """Only single-body WAMIT data are supported."""
        with pytest.raises(RadiationFitError, match="bodies"):
            ensure_radiation_state_space(_make_deck(tmp_path, n_body=2))

    def test_missing_wamit_data(self, tmp_path: Path) -> None:
        """Without the ``.1`` file nothing can be fitted."""
        fst = _make_deck(tmp_path)
        (tmp_path / "HydroData" / "body.1").unlink()
        with pytest.raises(RadiationFitError, match="not found"):
            ensure_radiation_state_space(fst)

    def test_missing_density(self, tmp_path: Path) -> None:
        """A deck without a water density cannot be dimensionalised."""
        fst = _make_deck(tmp_path, density_in="none")
        with pytest.raises(RadiationFitError, match="WtrDens"):
            ensure_radiation_state_space(fst)

    def test_inaccurate_fit_is_not_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fit outside the acceptance limit is reported but not written."""
        monkeypatch.setattr(radiation_module, "_MAX_ACCEPTED_ERROR", 1e-12)
        fst = _make_deck(tmp_path)
        record = ensure_radiation_state_space(fst)
        assert record is not None
        assert not record.written
        assert not record.accepted
        assert not record.path.exists()

    def test_confined_to_the_working_copy(self, tmp_path: Path) -> None:
        """A PotFile outside the working copy is never written next to."""
        shared = tmp_path / "shared"
        fst = _make_deck(shared)
        case = tmp_path / "case"
        case.mkdir()
        moved = case / "Model.fst"
        moved.write_text(
            fst.read_text(encoding="utf-8").replace(
                '"HD.dat"', f'"{(shared / "HD.dat").as_posix()}"'
            ),
            encoding="utf-8",
        )
        with pytest.raises(RadiationFitError, match="outside"):
            ensure_radiation_state_space(moved, confine_to=case)
        assert not (shared / "HydroData" / "body.ss").exists()
        record = ensure_radiation_state_space(fst, confine_to=shared)
        assert record is not None
        assert record.written

    def test_record_is_json_ready(self, tmp_path: Path) -> None:
        """The provenance dictionary holds strings and numbers only."""
        import json

        record = ensure_radiation_state_space(_make_deck(tmp_path))
        assert record is not None
        data = json.loads(json.dumps(record.to_dict(), allow_nan=False))
        assert data["path"] == str(record.path)
        assert data["source"] == str(record.source)
        assert data["written"] is True
        assert data["entry_errors"]["K11"] == record.entry_errors["K11"]


class TestPreviewRadiationStateSpace:
    """Fitting a deck's WAMIT data without writing anything."""

    def test_preview_writes_nothing(self, tmp_path: Path) -> None:
        """The record says whether the fit is usable; no file is written."""
        fst = _make_deck(tmp_path)
        record = preview_radiation_state_space(fst)
        assert record is not None
        assert record.accepted
        assert not record.written
        assert not record.path.exists()
        written = ensure_radiation_state_space(fst)
        assert written is not None
        assert written.n_states == record.n_states
        assert written.max_relative_error == pytest.approx(record.max_relative_error)

    def test_preview_without_potential_flow(self, tmp_path: Path) -> None:
        """No WAMIT flow: nothing to preview."""
        assert preview_radiation_state_space(_make_deck(tmp_path, pot_mod=0)) is None

    def test_preview_reports_missing_data(self, tmp_path: Path) -> None:
        """A missing ``.1`` file raises as for the generation."""
        fst = _make_deck(tmp_path)
        (tmp_path / "HydroData" / "body.1").unlink()
        with pytest.raises(RadiationFitError, match="not found"):
            preview_radiation_state_space(fst)

    def test_unusable_potfile_is_a_fit_error(self, tmp_path: Path) -> None:
        """A HydroDyn file naming no PotFile cannot be fitted."""
        fst = _make_deck(tmp_path)
        hydro = tmp_path / "HD.dat"
        hydro.write_text(
            "\n".join(
                line
                for line in hydro.read_text(encoding="utf-8").splitlines()
                if "PotFile" not in line
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(RadiationFitError, match="PotFile"):
            preview_radiation_state_space(fst)
