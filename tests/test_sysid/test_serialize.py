# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.sysid.serialize -- save/reload of estimator results."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from vane.sysid import (
    FilterResult,
    RomFamily,
    ScheduledFilterResult,
    StateSpace,
    build_kalman_model,
    initialize_covariances,
    load_result,
    load_rom_family,
    run_filter,
    run_scheduled_filter,
    save_result,
    save_rom_family,
)
from vane.sysid.kalman_filter import ConsistencyReport

if TYPE_CHECKING:
    from pathlib import Path


def _oscillator(freq_hz: float = 0.4) -> StateSpace:
    """Return a continuous 2-state oscillator."""
    wn = 2.0 * np.pi * freq_hz
    return StateSpace(
        a=np.array([[0.0, 1.0], [-(wn**2), -0.1 * wn]]),
        c=np.array([[1.0, 0.0]]),
        state_names=["pos", "vel"],
        output_names=["y"],
    )


def _filter_result(n: int = 30) -> FilterResult:
    """Return a FilterResult from a short run."""
    model = build_kalman_model(
        _oscillator(), initialize_covariances(2, 1, measurement_noise=1e-2), dt=0.05
    )
    rng = np.random.default_rng(0)
    return run_filter(model, rng.normal(0.0, 0.1, (n, 1)))


def _assert_arrays_equal(a: np.ndarray, b: np.ndarray) -> None:
    """Assert two arrays match exactly (NaNs in the same places count as equal)."""
    np.testing.assert_array_equal(a, b)


class TestRoundTrip:
    """Every estimator result reloads exactly as saved."""

    def test_filter_result_round_trip(self, tmp_path: Path) -> None:
        """A FilterResult survives save/load byte-for-byte."""
        result = _filter_result()
        path = tmp_path / "r.npz"
        save_result(result, path)
        loaded = load_result(path)
        assert isinstance(loaded, FilterResult)
        _assert_arrays_equal(loaded.states, result.states)
        _assert_arrays_equal(loaded.covariances, result.covariances)
        _assert_arrays_equal(loaded.nis, result.nis)  # NaNs preserved
        _assert_arrays_equal(loaded.accepted, result.accepted)
        _assert_arrays_equal(loaded.n_active, result.n_active)
        assert loaded.state_names == result.state_names

    def test_round_trips_with_a_suffixless_path(self, tmp_path: Path) -> None:
        """save/load normalize the .npz suffix, so a suffixless path round-trips."""
        result = _filter_result()
        path = tmp_path / "noext"  # no .npz suffix
        save_result(result, path)
        assert (tmp_path / "noext.npz").exists()  # np.savez appended it
        loaded = load_result(path)  # the same suffixless path still loads
        assert isinstance(loaded, FilterResult)
        _assert_arrays_equal(loaded.states, result.states)

    def test_scheduled_filter_result_round_trip(self, tmp_path: Path) -> None:
        """A ScheduledFilterResult (with bool/confidence arrays) round-trips."""
        family = RomFamily(
            operating_points=np.array([8.0, 12.0]),
            systems=[_oscillator(0.3), _oscillator(0.5)],
            frequencies_hz=np.array([[0.3], [0.5]]),
            damping_ratios=np.array([[0.05], [0.05]]),
            tracking_mac=np.array([[1.0], [0.97]]),
        )
        cov = initialize_covariances(2, 1, measurement_noise=1e-2)
        result = run_scheduled_filter(
            family, cov, np.zeros((25, 1)), np.linspace(8.0, 14.0, 25), dt=0.05
        )
        path = tmp_path / "s.npz"
        save_result(result, path)
        loaded = load_result(path)
        assert isinstance(loaded, ScheduledFilterResult)
        _assert_arrays_equal(loaded.min_tracking_mac, result.min_tracking_mac)
        _assert_arrays_equal(loaded.extrapolated, result.extrapolated)
        assert loaded.extrapolated.dtype == np.bool_
        assert loaded.state_names == result.state_names

    def test_rom_family_round_trip(self, tmp_path: Path) -> None:
        """A tracked ROM family preserves every matrix and modal assessment."""
        systems = [_oscillator(0.3), _oscillator(0.5)]
        family = RomFamily(
            operating_points=np.array([8.0, 12.0]),
            systems=systems,
            frequencies_hz=np.array([[0.3], [0.5]]),
            damping_ratios=np.array([[0.05], [0.05]]),
            tracking_mac=np.array([[1.0], [0.97]]),
        )
        path = tmp_path / "family"
        save_rom_family(family, path)
        loaded = load_rom_family(path)
        _assert_arrays_equal(loaded.operating_points, family.operating_points)
        _assert_arrays_equal(loaded.frequencies_hz, family.frequencies_hz)
        _assert_arrays_equal(loaded.damping_ratios, family.damping_ratios)
        _assert_arrays_equal(loaded.tracking_mac, family.tracking_mac)
        assert len(loaded.systems) == len(family.systems)
        for actual, expected in zip(loaded.systems, family.systems, strict=True):
            _assert_arrays_equal(actual.a, expected.a)
            assert actual.b is None and expected.b is None
            _assert_arrays_equal(actual.c, expected.c)  # type: ignore[arg-type]
            assert actual.d is None and expected.d is None
            assert actual.state_names == expected.state_names
            assert actual.output_names == expected.output_names
            assert actual.dt == expected.dt

    def test_rom_family_stabilized_mask_round_trips(self, tmp_path: Path) -> None:
        """The per-mode reflection mask and physical damping survive a round trip."""
        family = RomFamily(
            operating_points=np.array([8.0, 12.0]),
            systems=[_oscillator(0.3), _oscillator(0.5)],
            frequencies_hz=np.array([[0.3], [0.5]]),
            damping_ratios=np.array([[0.05], [-0.02]]),
            tracking_mac=np.array([[1.0], [0.97]]),
            stabilized=np.array([[False], [True]]),
        )
        path = tmp_path / "family"
        save_rom_family(family, path)
        loaded = load_rom_family(path)
        _assert_arrays_equal(loaded.stabilized, family.stabilized)
        assert loaded.stabilized.dtype == np.bool_
        _assert_arrays_equal(loaded.damping_ratios, family.damping_ratios)

    def test_rom_family_archive_without_mask_loads_unstabilized(
        self, tmp_path: Path
    ) -> None:
        """An archive written before the reflection mask existed still loads."""
        family = RomFamily(
            operating_points=np.array([8.0]),
            systems=[_oscillator()],
            frequencies_hz=np.array([[0.4]]),
            damping_ratios=np.array([[0.05]]),
            tracking_mac=np.array([[1.0]]),
        )
        clean = tmp_path / "clean.npz"
        save_rom_family(family, clean)
        with np.load(clean, allow_pickle=False) as data:
            payload: dict[str, Any] = {
                name: data[name] for name in data.files if name != "stabilized"
            }
        legacy = tmp_path / "legacy.npz"
        np.savez(legacy, **payload)
        loaded = load_rom_family(legacy)
        np.testing.assert_array_equal(loaded.stabilized, np.zeros((1, 1), dtype=bool))

    def test_rom_family_rejects_malformed_stabilized_mask(self, tmp_path: Path) -> None:
        """A mask of the wrong shape or dtype is rejected at both archive boundaries."""
        family = RomFamily(
            operating_points=np.array([8.0]),
            systems=[_oscillator()],
            frequencies_hz=np.array([[0.4]]),
            damping_ratios=np.array([[0.05]]),
            tracking_mac=np.array([[1.0]]),
            stabilized=np.zeros((1, 2), dtype=bool),
        )
        with pytest.raises(ValueError, match="stabilized shape"):
            save_rom_family(family, tmp_path / "bad")
        family.stabilized = np.zeros((1, 1), dtype=bool)
        clean = tmp_path / "clean.npz"
        save_rom_family(family, clean)
        with np.load(clean, allow_pickle=False) as data:
            payload: dict[str, Any] = {name: data[name] for name in data.files}
        payload["stabilized"] = np.zeros((1, 1))
        corrupt = tmp_path / "float-mask.npz"
        np.savez(corrupt, **payload)
        with pytest.raises(ValueError, match="boolean mask"):
            load_rom_family(corrupt)

    def test_rom_family_with_mixed_feedthrough_presence_round_trips(
        self, tmp_path: Path
    ) -> None:
        """A family the LPV blend accepts (missing D = zero) is also archivable."""
        b = np.array([[0.0], [1.0]])
        without_d = replace(_oscillator(0.3), b=b)
        with_d = replace(_oscillator(0.5), b=b, d=np.array([[0.5]]))
        family = RomFamily(
            operating_points=np.array([8.0, 12.0]),
            systems=[without_d, with_d],
            frequencies_hz=np.array([[0.3], [0.5]]),
            damping_ratios=np.array([[0.05], [0.05]]),
            tracking_mac=np.array([[1.0], [0.97]]),
        )
        path = tmp_path / "family"
        save_rom_family(family, path)
        loaded = load_rom_family(path)
        assert loaded.systems[0].d is None
        _assert_arrays_equal(loaded.systems[1].d, np.array([[0.5]]))  # type: ignore[arg-type]


class TestProvenanceAndSafety:
    """The archive is version-stamped and loads without executing code."""

    def test_file_records_format_and_vane_version(self, tmp_path: Path) -> None:
        """The saved file carries the format tag and the producing VANE version."""
        path = tmp_path / "r.npz"
        save_result(_filter_result(), path)
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["__meta__"]))
        assert meta["__format__"] == "vane-estimator-result"
        assert meta["__type__"] == "FilterResult"
        assert isinstance(meta["__vane_version__"], str)
        assert meta["__vane_version__"]  # non-empty

    def test_rom_family_archive_rejects_unexpected_array(self, tmp_path: Path) -> None:
        """A ROM archive cannot silently carry undeclared arrays."""
        family = RomFamily(
            operating_points=np.array([8.0]),
            systems=[_oscillator()],
            frequencies_hz=np.array([[0.4]]),
            damping_ratios=np.array([[0.05]]),
            tracking_mac=np.array([[1.0]]),
        )
        clean = tmp_path / "clean.npz"
        save_rom_family(family, clean)
        with np.load(clean, allow_pickle=False) as data:
            payload: dict[str, Any] = {name: data[name] for name in data.files}
        payload["undeclared"] = np.ones(1)
        corrupt = tmp_path / "corrupt.npz"
        np.savez(corrupt, **payload)
        with pytest.raises(ValueError, match="unexpected fields"):
            load_rom_family(corrupt)

    def test_rom_family_rejects_nonmonotonic_schedule(self, tmp_path: Path) -> None:
        """Saving and loading both reject an ambiguous interpolation schedule."""
        systems = [_oscillator(0.3), _oscillator(0.4), _oscillator(0.5)]
        malformed = RomFamily(
            operating_points=np.array([8.0, 16.0, 12.0]),
            systems=systems,
            frequencies_hz=np.array([[0.3], [0.4], [0.5]]),
            damping_ratios=np.full((3, 1), 0.05),
            tracking_mac=np.ones((3, 1)),
        )
        with pytest.raises(ValueError, match="strictly monotonic"):
            save_rom_family(malformed, tmp_path / "malformed")

        valid = RomFamily(
            operating_points=np.array([8.0, 12.0, 16.0]),
            systems=systems,
            frequencies_hz=malformed.frequencies_hz,
            damping_ratios=malformed.damping_ratios,
            tracking_mac=malformed.tracking_mac,
        )
        clean = tmp_path / "clean.npz"
        save_rom_family(valid, clean)
        with np.load(clean, allow_pickle=False) as data:
            payload: dict[str, Any] = {name: data[name] for name in data.files}
        payload["operating_points"] = malformed.operating_points
        corrupt = tmp_path / "nonmonotonic.npz"
        np.savez(corrupt, **payload)
        with pytest.raises(ValueError, match="strictly monotonic"):
            load_rom_family(corrupt)

    def test_rom_family_rejects_tracking_mac_outside_unit_interval(
        self, tmp_path: Path
    ) -> None:
        """Saving and loading both reject nonphysical tracking confidence."""
        family = RomFamily(
            operating_points=np.array([8.0]),
            systems=[_oscillator()],
            frequencies_hz=np.array([[0.4]]),
            damping_ratios=np.array([[0.05]]),
            tracking_mac=np.array([[1.01]]),
        )
        with pytest.raises(ValueError, match="physical interval"):
            save_rom_family(family, tmp_path / "malformed")

        clean = tmp_path / "clean.npz"
        family.tracking_mac[0, 0] = 1.0
        save_rom_family(family, clean)
        with np.load(clean, allow_pickle=False) as data:
            payload: dict[str, Any] = {name: data[name] for name in data.files}
        payload["tracking_mac"] = np.array([[2.0]])
        corrupt = tmp_path / "invalid-mac.npz"
        np.savez(corrupt, **payload)
        with pytest.raises(ValueError, match="physical interval"):
            load_rom_family(corrupt)

    def test_load_rejects_non_vane_archive(self, tmp_path: Path) -> None:
        """An npz without VANE metadata is refused (no silent misread)."""
        path = tmp_path / "plain.npz"
        np.savez(path, x=np.arange(5))
        with pytest.raises(ValueError, match="no metadata"):
            load_result(path)

    def test_load_rejects_wrong_format_tag(self, tmp_path: Path) -> None:
        """A file whose metadata is not the estimator-result format is refused."""
        path = tmp_path / "bad.npz"
        np.savez(
            path, __meta__=np.asarray(json.dumps({"__format__": "something-else"}))
        )
        with pytest.raises(ValueError, match="wrong format tag"):
            load_result(path)

    def test_load_rejects_unknown_type(self, tmp_path: Path) -> None:
        """A known-format file naming an unregistered type is refused."""
        path = tmp_path / "unknown.npz"
        meta = {"__format__": "vane-estimator-result", "__type__": "Nope", "fields": {}}
        np.savez(path, __meta__=np.asarray(json.dumps(meta)))
        with pytest.raises(ValueError, match="unknown estimator-result type"):
            load_result(path)

    def test_load_rejects_nested_only_type_at_top_level(self, tmp_path: Path) -> None:
        """A crafted archive naming a nested-only type at the top level is refused."""
        path = tmp_path / "nested.npz"
        meta = {
            "__format__": "vane-estimator-result",
            "__type__": "ConsistencyReport",  # valid only nested, not at top level
            # (a nested-only dataclass must not be a top-level result)
            "fields": {},
        }
        np.savez(path, __meta__=np.asarray(json.dumps(meta)))
        with pytest.raises(ValueError, match="unknown estimator-result type"):
            load_result(path)

    def test_load_rejects_archive_missing_a_required_field(
        self, tmp_path: Path
    ) -> None:
        """A truncated archive missing a required array is refused, not defaulted."""
        path = tmp_path / "truncated.npz"
        meta = {
            "__format__": "vane-estimator-result",
            "__type__": "FilterResult",
            "fields": {"state_names": []},  # no arrays, so 'states' is absent
        }
        np.savez(path, __meta__=np.asarray(json.dumps(meta)))
        with pytest.raises(ValueError, match="missing required field 'states'"):
            load_result(path)

    def test_load_rejects_required_array_supplied_as_metadata(
        self, tmp_path: Path
    ) -> None:
        """A required array sneaked in as a metadata null (not an array) is refused."""
        path = tmp_path / "tampered.npz"
        # Every required array present except ``states``, which is a null in metadata.
        arrays = {
            "covariances": np.zeros((3, 2, 2)),
            "predicted_states": np.zeros((3, 2)),
            "predicted_covariances": np.zeros((3, 2, 2)),
            "nis": np.zeros(3),
            "accepted": np.zeros(3, dtype=bool),
            "n_active": np.zeros(3, dtype=int),
        }
        meta = {
            "__format__": "vane-estimator-result",
            "__type__": "FilterResult",
            "fields": {"states": None, "state_names": []},
        }
        payload = {"__meta__": np.asarray(json.dumps(meta)), **arrays}
        np.savez(path, **payload)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="'states' must be an array"):
            load_result(path)

    def test_required_array_fields_matches_real_results(self) -> None:
        """The array-field heuristic matches the fields that are actually arrays."""
        from dataclasses import fields as dc_fields

        from vane.sysid.serialize import _required_array_fields

        for result in (_filter_result(),):
            actual = {
                f.name
                for f in dc_fields(result)
                if isinstance(getattr(result, f.name), np.ndarray)
            }
            assert set(_required_array_fields(type(result))) == actual

    def test_save_rejects_non_result(self, tmp_path: Path) -> None:
        """save_result refuses a type it does not handle (e.g. a bare report)."""
        report = ConsistencyReport(
            mean_nis=1.0,
            expected_nis=1.0,
            lower=0.5,
            upper=1.5,
            consistent=True,
            n_accepted=10,
        )
        with pytest.raises(TypeError, match="does not handle"):
            save_result(report, tmp_path / "x.npz")  # type: ignore[arg-type]

    def test_save_rejects_a_subclass(self, tmp_path: Path) -> None:
        """A subclass is refused at save time (its name would be unreloadable)."""
        from dataclasses import fields as dc_fields

        class _SubFilterResult(FilterResult):
            pass

        base = _filter_result()
        kwargs = {f.name: getattr(base, f.name) for f in dc_fields(base)}
        sub = _SubFilterResult(**kwargs)
        with pytest.raises(TypeError, match="does not handle"):
            save_result(sub, tmp_path / "sub.npz")
