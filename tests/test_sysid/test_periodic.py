# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for fixed-basis rotor-azimuth-periodic modal systems."""

import numpy as np
import numpy.typing as npt
import pytest

from vane.io import MBCResult
from vane.sysid import (
    PeriodicStateSpace,
    StateSpace,
    modal_reduction,
    periodic_state_space_from_mbc,
)


def _present(matrix: npt.NDArray[np.float64] | None) -> npt.NDArray[np.float64]:
    """Return an optional state-space matrix, asserting that it is present."""
    assert matrix is not None
    return matrix


def _system() -> StateSpace:
    return StateSpace(
        a=np.array([[-0.1, -2.0], [2.0, -0.1]]),
        b=np.array([[1.0], [0.25]]),
        c=np.array([[1.0, -0.5]]),
        d=np.array([[0.1]]),
        state_names=["x", "xdot"],
        input_names=["wind"],
        output_names=["load"],
    )


def _mbc(system: StateSpace) -> MBCResult:
    delta_a = np.array([[0.02, 0.03], [-0.01, -0.02]])
    delta_b = np.array([[0.1], [-0.05]])
    delta_c = np.array([[0.2, -0.1]])
    delta_d = np.array([[0.02]])
    assert system.b is not None
    assert system.c is not None
    assert system.d is not None
    return MBCResult(
        avg_a=system.a,
        avg_b=system.b,
        avg_c=system.c,
        avg_d=system.d,
        state_descriptions=list(system.state_names),
        ndof2=1,
        ndof1=0,
        n_blades=0,
        performed_transformation=False,
        rotor_speed_rpm=10.0,
        wind_speed=12.0,
        azimuths_deg=np.array([0.0, 180.0]),
        per_azimuth_a=[system.a + delta_a, system.a - delta_a],
        per_azimuth_b=[system.b + delta_b, system.b - delta_b],
        per_azimuth_c=[system.c + delta_c, system.c - delta_c],
        per_azimuth_d=[system.d + delta_d, system.d - delta_d],
        input_descriptions=list(system.input_names),
        output_descriptions=list(system.output_names),
    )


def test_modal_reduction_retains_biorthogonal_projection_maps() -> None:
    """The retained modal state has explicit full-state projection maps."""
    system = _system()
    reduction = modal_reduction(system, n_modes=1)
    assert reduction.state_projection is not None
    assert reduction.state_reconstruction is not None
    np.testing.assert_allclose(
        reduction.state_projection @ reduction.state_reconstruction,
        np.eye(2),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _present(reduction.system.b),
        reduction.state_projection @ _present(system.b),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _present(reduction.system.c),
        _present(system.c) @ reduction.state_reconstruction,
        atol=1e-12,
    )


def test_periodic_projection_uses_one_fixed_modal_basis_and_wraps() -> None:
    """Azimuth members share a state basis and interpolate across 360 degrees."""
    system = _system()
    reduction = modal_reduction(system, n_modes=1)
    periodic = periodic_state_space_from_mbc(_mbc(system), reduction)
    assert len(periodic.systems) == 2
    at_90 = periodic.at(90.0)
    at_270 = periodic.at(270.0)
    np.testing.assert_allclose(at_90.a, reduction.system.a, atol=1e-12)
    np.testing.assert_allclose(at_270.a, reduction.system.a, atol=1e-12)
    np.testing.assert_allclose(
        _present(at_90.b), _present(reduction.system.b), atol=1e-12
    )
    np.testing.assert_allclose(
        _present(at_270.c), _present(reduction.system.c), atol=1e-12
    )
    np.testing.assert_allclose(
        _present(at_270.d), _present(reduction.system.d), atol=1e-12
    )


def test_periodic_projection_preserves_dc_gain_correction() -> None:
    """A constant family retains the reduced model's corrected feedthrough."""
    system = StateSpace(
        a=np.array(
            [
                [-0.1, -2.0, 0.0, 0.0],
                [2.0, -0.1, 0.0, 0.0],
                [0.0, 0.0, -0.5, -6.0],
                [0.0, 0.0, 6.0, -0.5],
            ]
        ),
        b=np.array([[1.0], [0.25], [0.5], [0.1]]),
        c=np.array([[1.0, -0.5, 0.3, 0.2]]),
        d=np.array([[0.1]]),
        state_names=["x1", "xdot1", "x2", "xdot2"],
        input_names=["wind"],
        output_names=["load"],
    )
    reduction = modal_reduction(system, n_modes=1, match_dc_gain=True)
    assert system.b is not None and system.c is not None and system.d is not None
    assert reduction.system.d is not None
    assert not np.allclose(reduction.system.d, system.d)
    result = MBCResult(
        avg_a=system.a,
        avg_b=system.b,
        avg_c=system.c,
        avg_d=system.d,
        state_descriptions=list(system.state_names),
        ndof2=2,
        ndof1=0,
        n_blades=0,
        performed_transformation=False,
        rotor_speed_rpm=10.0,
        wind_speed=12.0,
        azimuths_deg=np.array([0.0, 180.0]),
        per_azimuth_a=[system.a, system.a],
        per_azimuth_b=[system.b, system.b],
        per_azimuth_c=[system.c, system.c],
        per_azimuth_d=[system.d, system.d],
        input_descriptions=list(system.input_names),
        output_descriptions=list(system.output_names),
    )

    periodic = periodic_state_space_from_mbc(result, reduction)

    for member in periodic.systems:
        np.testing.assert_allclose(member.a, reduction.system.a, atol=1.0e-12)
        np.testing.assert_allclose(
            _present(member.b), _present(reduction.system.b), atol=1.0e-12
        )
        np.testing.assert_allclose(
            _present(member.c), _present(reduction.system.c), atol=1.0e-12
        )
        np.testing.assert_allclose(
            _present(member.d), _present(reduction.system.d), atol=1.0e-12
        )

    result.per_azimuth_b = None
    result.per_azimuth_c = None
    result.per_azimuth_d = None
    constant_io = periodic_state_space_from_mbc(result, reduction)
    for member in constant_io.systems:
        np.testing.assert_allclose(
            _present(member.b), _present(reduction.system.b), atol=1.0e-12
        )
        np.testing.assert_allclose(
            _present(member.c), _present(reduction.system.c), atol=1.0e-12
        )
        np.testing.assert_allclose(
            _present(member.d), _present(reduction.system.d), atol=1.0e-12
        )


def _two_output_mbc(output_names: list[str]) -> tuple[MBCResult, StateSpace]:
    """Return an MBC result with outputs ("load", "tension") and its full system."""
    a = np.array([[-0.1, -2.0], [2.0, -0.1]])
    b = np.array([[1.0], [0.25]])
    c = np.array([[1.0, -0.5], [0.0, 3.0]])
    d = np.array([[0.1], [0.0]])
    delta_c = np.array([[0.5, 0.0], [0.0, 0.0]])  # only "load" varies with azimuth
    delta_d = np.array([[0.02], [0.0]])
    full = StateSpace(
        a=a,
        b=b,
        c=c,
        d=d,
        state_names=["x", "v"],
        input_names=["wind"],
        output_names=["load", "tension"],
    )
    result = MBCResult(
        avg_a=a,
        avg_b=b,
        avg_c=c,
        avg_d=d,
        state_descriptions=["x", "v"],
        ndof2=1,
        ndof1=0,
        n_blades=0,
        performed_transformation=False,
        rotor_speed_rpm=10.0,
        wind_speed=12.0,
        azimuths_deg=np.array([0.0, 180.0]),
        per_azimuth_a=[a, a],
        per_azimuth_b=[b, b],
        per_azimuth_c=[c + delta_c, c - delta_c],
        per_azimuth_d=[d + delta_d, d - delta_d],
        input_descriptions=["wind"],
        output_descriptions=output_names,
    )
    return result, full


def test_periodic_projection_remaps_reordered_outputs_by_name() -> None:
    """A reduction with reordered outputs receives each deviation on its own row."""
    result, full = _two_output_mbc(["load", "tension"])
    assert full.c is not None and full.d is not None
    order = [1, 0]
    reordered = StateSpace(
        a=full.a,
        b=full.b,
        c=full.c[order],
        d=full.d[order],
        state_names=list(full.state_names),
        input_names=list(full.input_names),
        output_names=["tension", "load"],
    )
    reduction = modal_reduction(reordered, n_modes=1)
    assert reduction.state_reconstruction is not None
    periodic = periodic_state_space_from_mbc(result, reduction)
    member = periodic.systems[0]
    assert member.output_names == ["tension", "load"]
    assert result.per_azimuth_c is not None and result.per_azimuth_d is not None
    expected_c = result.per_azimuth_c[0][order] @ reduction.state_reconstruction
    np.testing.assert_allclose(_present(member.c), expected_c, atol=1e-12)
    assert reduction.system.d is not None
    np.testing.assert_allclose(
        _present(member.d), result.per_azimuth_d[0][order], atol=1e-12
    )


@pytest.mark.parametrize(
    ("mbc_outputs", "match"),
    [
        (["load", "strain"], "does not match"),
        (["load", "load"], "ambiguous"),
    ],
)
def test_periodic_projection_rejects_unmatched_output_layout(
    mbc_outputs: list[str], match: str
) -> None:
    """Outputs that are not a permutation of the MBC layout are rejected."""
    result, full = _two_output_mbc(mbc_outputs)
    reduction = modal_reduction(full, n_modes=1)
    with pytest.raises(ValueError, match=match):
        periodic_state_space_from_mbc(result, reduction)


def test_periodic_projection_rejects_mismatched_channel_counts() -> None:
    """An unnamed reduction with a different output count cannot be positioned."""
    result, full = _two_output_mbc(["load", "tension"])
    assert full.c is not None and full.d is not None
    single = StateSpace(a=full.a, b=full.b, c=full.c[:1], d=full.d[:1])
    reduction = modal_reduction(single, n_modes=1)
    with pytest.raises(ValueError, match="1 outputs but the MBC"):
        periodic_state_space_from_mbc(result, reduction)


def test_periodic_projection_rejects_mismatched_output_mbc_coordinates() -> None:
    """Equal names with different MBC coordinates are different physical outputs."""
    result, full = _two_output_mbc(["load", "tension"])
    result.output_mbc_coordinates = ["collective", "cosine"]
    full.output_mbc_coordinates = ["collective", "sine"]
    reduction = modal_reduction(full, n_modes=1)
    with pytest.raises(ValueError, match="does not match"):
        periodic_state_space_from_mbc(result, reduction)


def test_periodic_family_can_be_discretized_before_online_interpolation() -> None:
    """Exact grid-point discretization supports cheap runtime interpolation."""
    system = _system()
    reduction = modal_reduction(system, n_modes=1)
    periodic = periodic_state_space_from_mbc(_mbc(system), reduction)
    discrete = periodic.discretized(0.1)
    assert all(member.dt == 0.1 for member in discrete.systems)
    assert discrete.at(359.0).dt == 0.1


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("state_names", ["x2", "x1"]),
        ("input_names", ["u2", "u1"]),
        ("output_names", ["y2", "y1"]),
        ("output_mbc_coordinates", ["sine", "collective"]),
    ],
)
def test_periodic_family_rejects_mismatched_channel_layout(
    field: str, replacement: list[str]
) -> None:
    """Equal matrix shapes cannot conceal reordered physical channels."""
    names = {
        "state_names": ["x1", "x2"],
        "input_names": ["u1", "u2"],
        "output_names": ["y1", "y2"],
        "output_mbc_coordinates": ["collective", "sine"],
    }
    changed = {name: list(values) for name, values in names.items()}
    changed[field] = replacement

    def member(layout: dict[str, list[str]]) -> StateSpace:
        return StateSpace(
            a=np.diag([-0.1, -0.2]),
            b=np.eye(2),
            c=np.eye(2),
            d=np.zeros((2, 2)),
            state_names=layout["state_names"],
            input_names=layout["input_names"],
            output_names=layout["output_names"],
            output_mbc_coordinates=layout["output_mbc_coordinates"],
        )

    with pytest.raises(ValueError, match="channel layout differs"):
        PeriodicStateSpace(
            azimuths_deg=np.array([0.0, 180.0]),
            systems=(member(names), member(changed)),
        )
