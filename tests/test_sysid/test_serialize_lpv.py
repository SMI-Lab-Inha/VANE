# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.sysid.serialize -- save/reload of physical-state LPV families."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from vane._version import vane_version
from vane.export import (
    LPVFamily,
    build_physical_lpv_family,
    load_lpv_family,
    save_lpv_family,
)
from vane.sysid import (
    LPV_FAMILY_SCHEMA_VERSION,
    StateSpace,
    save_rom_family,
)
from vane.sysid.rom_family import RomFamily

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.conftest import LinFileFactory

    # Changes an archive's metadata and arrays in place.
    _Edit = Callable[[dict[str, Any], dict[str, np.ndarray]], object]

_STATES = ["ED PtfmPitch", "ED First time derivative of PtfmPitch", "ED RotSpeed"]
_INPUTS = ["GenTq", "BldPitch1"]
_OUTPUTS = ["RotSpeed", "BldPitch1", "TwrBsMyt"]


def _synthetic_family(n_points: int = 4, seed: int = 7) -> LPVFamily:
    """Return a family with distinct random matrices at each scheduling point."""
    rng = np.random.default_rng(seed)
    n_x, n_u, n_y = len(_STATES), len(_INPUTS), len(_OUTPUTS)
    systems = [
        StateSpace(
            a=rng.standard_normal((n_x, n_x)),
            b=rng.standard_normal((n_x, n_u)),
            c=rng.standard_normal((n_y, n_x)),
            d=rng.standard_normal((n_y, n_u)),
            state_names=list(_STATES),
            input_names=list(_INPUTS),
            output_names=list(_OUTPUTS),
        )
        for _ in range(n_points)
    ]
    return LPVFamily(
        scheduling_points=np.linspace(0.5, 1.25, n_points),
        systems=systems,
        state_names=list(_STATES),
        input_names=list(_INPUTS),
        output_names=list(_OUTPUTS),
        scheduling_variable="RotorSpeed",
        scheduling_unit="rad/s",
        auto_added_states=["ED First time derivative of PtfmPitch"],
        dropped_coupling_a=rng.uniform(0.0, 0.04, n_points),
        dropped_coupling_c=rng.uniform(0.0, 0.04, n_points),
    )


def _assert_same_family(loaded: LPVFamily, original: LPVFamily) -> None:
    """Assert that every field and every matrix round-tripped exactly."""
    assert type(loaded) is LPVFamily
    for name in ("scheduling_points", "dropped_coupling_a", "dropped_coupling_c"):
        expected = getattr(original, name)
        actual = getattr(loaded, name)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    for name in (
        "state_names",
        "input_names",
        "output_names",
        "scheduling_variable",
        "scheduling_unit",
        "auto_added_states",
    ):
        assert getattr(loaded, name) == getattr(original, name)
    assert len(loaded.systems) == len(original.systems)
    for got, want in zip(loaded.systems, original.systems, strict=True):
        assert type(got) is StateSpace
        for matrix in ("a", "b", "c", "d"):
            expected = getattr(want, matrix)
            actual = getattr(got, matrix)
            if expected is None:
                assert actual is None
            else:
                assert actual.dtype == expected.dtype
                np.testing.assert_array_equal(actual, expected)
        assert got.state_names == want.state_names
        assert got.input_names == want.input_names
        assert got.output_names == want.output_names
        assert got.output_mbc_coordinates == want.output_mbc_coordinates
        assert got.dt == want.dt


def _rewrite(path: Path, edit: _Edit) -> None:
    """Load an archive, let ``edit(meta, arrays)`` change it, and write it back."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    meta = json.loads(str(arrays.pop("__meta__")))
    edit(meta, arrays)
    arrays["__meta__"] = np.asarray(json.dumps(meta))
    np.savez(path, **arrays)  # type: ignore[arg-type]


class TestRoundTrip:
    """A saved LPV family reloads exactly as it was saved."""

    def test_synthetic_family_round_trips_exactly(self, tmp_path: Path) -> None:
        """Arrays, names, schedule metadata and the norms all come back unchanged."""
        family = _synthetic_family()
        path = tmp_path / "plant.npz"
        save_lpv_family(family, path)
        _assert_same_family(load_lpv_family(path), family)

    def test_archive_is_pickle_free_and_stamped(self, tmp_path: Path) -> None:
        """No object arrays; the metadata records format, schema and VANE version."""
        path = tmp_path / "plant.npz"
        save_lpv_family(_synthetic_family(), path)
        with np.load(path, allow_pickle=False) as data:
            assert all(data[key].dtype != object for key in data.files)
            meta = json.loads(str(data["__meta__"]))
        assert meta["__format__"] == "vane-lpv-family"
        assert meta["__schema__"] == LPV_FAMILY_SCHEMA_VERSION == 1
        assert meta["__vane_version__"] == vane_version()

    def test_suffix_is_added_on_save_and_load(self, tmp_path: Path) -> None:
        """A path without ``.npz`` writes and reads the same ``.npz`` file."""
        family = _synthetic_family(n_points=2)
        save_lpv_family(family, tmp_path / "plant")
        assert (tmp_path / "plant.npz").is_file()
        _assert_same_family(load_lpv_family(tmp_path / "plant"), family)

    def test_optional_matrices_and_discrete_time(self, tmp_path: Path) -> None:
        """Absent B/C/D, a sample time and empty norms round-trip as well."""
        systems = [
            StateSpace(a=np.diag([-1.0, -2.0]) * k, dt=0.05, state_names=["x1", "x2"])
            for k in (1.0, 2.0, 3.0)
        ]
        family = LPVFamily(
            scheduling_points=np.array([4.0, 8.0, 12.0]),
            systems=systems,
            state_names=["x1", "x2"],
        )
        path = tmp_path / "states_only.npz"
        save_lpv_family(family, path)
        loaded = load_lpv_family(path)
        _assert_same_family(loaded, family)
        assert loaded.dropped_coupling.shape == (0,)

    def test_single_point_family(self, tmp_path: Path) -> None:
        """A one-point family is valid and round-trips."""
        family = _synthetic_family(n_points=1)
        save_lpv_family(family, tmp_path / "one.npz")
        _assert_same_family(load_lpv_family(tmp_path / "one.npz"), family)

    def test_family_from_lin_files_round_trips(
        self, make_lin_file: LinFileFactory, tmp_path: Path
    ) -> None:
        """A family built by build_physical_lpv_family from .lin files round-trips."""
        a = np.array([[0.0, 1.0, 0.0], [-0.4, -0.05, 0.1], [0.0, 0.2, -0.3]])
        b = np.arange(6, dtype=np.float64).reshape(3, 2)
        c = np.arange(6, dtype=np.float64).reshape(2, 3)
        d = np.arange(4, dtype=np.float64).reshape(2, 2)
        states = [
            ("ED PtfmPitch, (deg)", False, 2),
            ("ED PtfmPitch rate, (deg/s)", False, 2),
            ("ED RotSpeed, (rpm)", False, 2),
        ]
        inputs = [("GenTq, (Nm)", False, 0), ("BldPitch1, (rad)", True, 0)]
        outputs = [("RotSpeed, (rpm)", False, 0), ("BldPitch1, (rad)", True, 0)]
        lin_dir = None
        for wind, scale in ((5.0, 1.0), (9.0, 2.0), (13.0, 3.0)):
            path = make_lin_file(
                states=states,
                inputs=inputs,
                outputs=outputs,
                a=a * scale,
                b=b * scale,
                c=c * scale,
                d=d * scale,
                wind_speed=wind,
                rotor_speed=0.5 + 0.05 * wind,
            )
            lin_dir = path.parent
        assert lin_dir is not None
        family = build_physical_lpv_family(
            lin_dir,
            states=["ED PtfmPitch", "ED PtfmPitch rate", "ED RotSpeed"],
            inputs=["GenTq", "BldPitch1"],
            outputs=["RotSpeed", "BldPitch1"],
        )
        archive = tmp_path / "from_lin.npz"
        save_lpv_family(family, archive)
        _assert_same_family(load_lpv_family(archive), family)


class TestSchemaVersion:
    """An archive of another schema version is rejected with both versions named."""

    def test_newer_schema_is_rejected(self, tmp_path: Path) -> None:
        """A newer schema names the writing version and asks for an upgrade."""
        path = tmp_path / "plant.npz"
        save_lpv_family(_synthetic_family(), path)

        def newer(meta: dict[str, Any], _arrays: dict[str, np.ndarray]) -> None:
            meta["__schema__"] = 2
            meta["__vane_version__"] = "9.9.9"

        _rewrite(path, newer)
        with pytest.raises(
            ValueError, match=r"schema 2 \(written by VANE 9\.9\.9\)"
        ) as e:
            load_lpv_family(path)
        assert "reads schema 1 only" in str(e.value)
        assert "upgrade VANE" in str(e.value)

    def test_older_schema_is_rejected(self, tmp_path: Path) -> None:
        """An older schema is rejected with the rebuild remedy."""
        path = tmp_path / "plant.npz"
        save_lpv_family(_synthetic_family(), path)
        _rewrite(path, lambda meta, _arrays: meta.update({"__schema__": 0}))
        with pytest.raises(ValueError, match=r"re-create it from the \.lin files"):
            load_lpv_family(path)

    def test_missing_schema_is_rejected(self, tmp_path: Path) -> None:
        """An archive without a schema version is not read on trust."""
        path = tmp_path / "plant.npz"
        save_lpv_family(_synthetic_family(), path)
        _rewrite(path, lambda meta, _arrays: meta.pop("__schema__"))
        with pytest.raises(ValueError, match="records no schema version"):
            load_lpv_family(path)

    @pytest.mark.parametrize("schema", ["1", True, 1.0, None])
    def test_invalid_schema_value_is_rejected(
        self, tmp_path: Path, schema: object
    ) -> None:
        """A schema that is not an integer (a string, a boolean) is invalid."""
        path = tmp_path / "plant.npz"
        save_lpv_family(_synthetic_family(), path)
        _rewrite(path, lambda meta, _arrays: meta.update({"__schema__": schema}))
        with pytest.raises(ValueError, match="schema"):
            load_lpv_family(path)


class TestInvalidArchives:
    """Anything but an intact LPV-family archive raises ValueError."""

    def test_other_archive_types_are_rejected(self, tmp_path: Path) -> None:
        """A ROM-family archive is not an LPV-family archive."""
        system = StateSpace(a=np.diag([-1.0, -2.0]))
        rom = RomFamily(
            operating_points=np.array([8.0]),
            systems=[system],
            frequencies_hz=np.array([[0.2]]),
            damping_ratios=np.array([[0.05]]),
            tracking_mac=np.array([[1.0]]),
        )
        save_rom_family(rom, tmp_path / "rom.npz")
        with pytest.raises(ValueError, match="wrong format tag"):
            load_lpv_family(tmp_path / "rom.npz")

    def test_archive_without_metadata_is_rejected(self, tmp_path: Path) -> None:
        """A plain npz of arrays is not an LPV-family archive."""
        np.savez(tmp_path / "plain.npz", a=np.eye(2))
        with pytest.raises(ValueError, match="no metadata"):
            load_lpv_family(tmp_path / "plain.npz")

    @pytest.mark.parametrize(
        ("edit", "match"),
        [
            (lambda m, a: a.pop("system_0001_b"), "missing matrix 'system_0001_b'"),
            (lambda m, a: a.pop("system_0002_a"), "missing matrix 'system_0002_a'"),
            (lambda m, a: a.pop("dropped_coupling_a"), "missing fields"),
            (lambda m, a: a.update({"extra": np.zeros(1)}), "unexpected fields"),
            (lambda m, a: m.update({"n_systems": 3}), "invalid system metadata"),
            (lambda m, a: m.update({"n_systems": True}), "invalid system metadata"),
            (
                lambda m, a: a.update({"scheduling_points": np.array([4, 3, 2, 1.0])}),
                "strictly ascending",
            ),
            (
                lambda m, a: a.update({"system_0000_a": np.eye(2)}),
                "does not share system 0's layout|system 0 is invalid",
            ),
            (lambda m, a: m.update({"state_names": "ED RotSpeed"}), "list of strings"),
            (lambda m, a: m.pop("scheduling_unit"), "metadata is invalid"),
            (lambda m, a: m.update({"scheduling_unit": 3}), "must be a string"),
            (
                lambda m, a: m["systems"][0].update({"matrices_present": {"b": True}}),
                "invalid matrix metadata",
            ),
            (
                lambda m, a: m["systems"][0]["matrices_present"].update({"d": "yes"}),
                "matrix flag 'd' is invalid",
            ),
            (lambda m, a: m["systems"].__setitem__(1, []), "metadata 1 is invalid"),
            (lambda m, a: m["systems"][2].update({"dt": -1.0}), "system 2 is invalid"),
            (
                lambda m, a: a.update(
                    {"dropped_coupling_c": np.array([np.nan, 0.0, 0.0, 0.0])}
                ),
                "must be finite",
            ),
        ],
    )
    def test_tampered_archive_is_rejected(
        self, tmp_path: Path, edit: _Edit, match: str
    ) -> None:
        """Each corruption is reported as a ValueError naming the problem."""
        path = tmp_path / "plant.npz"
        save_lpv_family(_synthetic_family(), path)
        _rewrite(path, edit)
        with pytest.raises(ValueError, match=match):
            load_lpv_family(path)


class TestSaveValidation:
    """save_lpv_family refuses a family it could not reload."""

    def test_rejects_other_types(self, tmp_path: Path) -> None:
        """Only an exact LPVFamily is saved."""
        with pytest.raises(TypeError, match="does not handle"):
            save_lpv_family("family", tmp_path / "x.npz")  # type: ignore[arg-type]

    def test_rejects_an_empty_family(self, tmp_path: Path) -> None:
        """A family without systems is rejected."""
        family = LPVFamily(scheduling_points=np.zeros(0), systems=[])
        with pytest.raises(ValueError, match="at least one system"):
            save_lpv_family(family, tmp_path / "x.npz")

    def test_rejects_mismatched_layouts(self, tmp_path: Path) -> None:
        """Members must share the matrix shapes and names."""
        family = _synthetic_family(n_points=2)
        family.systems[1] = replace(family.systems[1], input_names=["u1", "u2"])
        with pytest.raises(ValueError, match="does not share system 0's layout"):
            save_lpv_family(family, tmp_path / "x.npz")

    def test_rejects_unsorted_schedule(self, tmp_path: Path) -> None:
        """The scheduling points must be strictly ascending."""
        family = _synthetic_family(n_points=3)
        family.scheduling_points = family.scheduling_points[::-1].copy()
        with pytest.raises(ValueError, match="strictly ascending"):
            save_lpv_family(family, tmp_path / "x.npz")

    def test_rejects_wrong_length_schedule(self, tmp_path: Path) -> None:
        """There is one scheduling point per system."""
        family = _synthetic_family(n_points=3)
        family.scheduling_points = np.array([0.5, 1.0])
        with pytest.raises(ValueError, match="one value per system, got"):
            save_lpv_family(family, tmp_path / "x.npz")

    def test_rejects_wrong_length_norms(self, tmp_path: Path) -> None:
        """The dropped-coupling norms have one value per point, or none."""
        family = _synthetic_family(n_points=3)
        family.dropped_coupling_a = np.zeros(2)
        with pytest.raises(ValueError, match="one value per system or be empty"):
            save_lpv_family(family, tmp_path / "x.npz")

    def test_rejects_wrong_field_types(self, tmp_path: Path) -> None:
        """Lists instead of arrays, integer arrays and non-StateSpace members fail."""
        family = _synthetic_family(n_points=2)
        family.scheduling_points = [0.5, 1.0]  # type: ignore[assignment]
        with pytest.raises(TypeError, match="must be a NumPy array"):
            save_lpv_family(family, tmp_path / "x.npz")
        family = _synthetic_family(n_points=2)
        family.scheduling_points = np.array([1, 2])  # type: ignore[assignment]
        with pytest.raises(TypeError, match="floating-point array"):
            save_lpv_family(family, tmp_path / "x.npz")
        family = _synthetic_family(n_points=2)
        family.systems[0] = "not a system"  # type: ignore[call-overload]
        with pytest.raises(TypeError, match="not an exact StateSpace"):
            save_lpv_family(family, tmp_path / "x.npz")
        family = _synthetic_family(n_points=2)
        family.scheduling_variable = None  # type: ignore[assignment]
        with pytest.raises(TypeError, match="must be a string"):
            save_lpv_family(family, tmp_path / "x.npz")
        family = _synthetic_family(n_points=2)
        family.auto_added_states = ("x",)  # type: ignore[assignment]
        with pytest.raises(TypeError, match="list of strings"):
            save_lpv_family(family, tmp_path / "x.npz")
