# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.config.system_config — reference FOWT geometry."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from vane.config.system_config import (
    FowtGeometry,
    Member,
    SystemType,
    _rotor_frame,
    default_geometry,
    jacket_geometry,
    monopile_geometry,
    onshore_geometry,
    semisub_geometry,
    spar_geometry,
    tlp_geometry,
)


class TestPublicReExports:
    """The geometry presets are reachable from the public ``vane.config`` package."""

    def test_all_builders_importable_from_vane_config(self) -> None:
        """Every preset re-exports from ``vane.config``, not just ``system_config``."""
        import vane.config as cfg

        for name in (
            "onshore_geometry",
            "monopile_geometry",
            "jacket_geometry",
            "spar_geometry",
            "semisub_geometry",
            "tlp_geometry",
            "default_geometry",
        ):
            assert hasattr(cfg, name), name
            assert name in cfg.__all__


class TestRotorFrame:
    """Tests for the shaft-tilt rotor frame."""

    def test_zero_tilt_is_identity(self) -> None:
        """No shaft tilt leaves the rotor frame axis-aligned."""
        np.testing.assert_allclose(_rotor_frame(0.0), np.eye(3), atol=1e-12)

    def test_positive_tilt_raises_downwind_shaft_end(self) -> None:
        """A positive tilt raises the axial (column 0) shaft end in +z."""
        frame = _rotor_frame(10.0)
        axial = frame[:, 0]
        # Documented convention: axial maps to (cos, 0, +sin) -> positive z component.
        assert axial[2] > 0.0
        np.testing.assert_allclose(
            axial, [np.cos(np.radians(10.0)), 0.0, np.sin(np.radians(10.0))], atol=1e-12
        )

    def test_frame_is_a_proper_rotation(self) -> None:
        """The rotor frame is orthonormal with determinant +1."""
        frame = _rotor_frame(7.0)
        np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(frame), 1.0, atol=1e-12)


class TestRotorSignConventions:
    """Blade layout and bending directions follow the OpenFAST rotor conventions."""

    @staticmethod
    def _blades(
        shaft_tilt_deg: float = -5.0, precone_deg: float = -2.5
    ) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Member]]:
        """Return ``(root, tip, member)`` for each blade of a semi-sub rotor."""
        geometry = semisub_geometry(
            shaft_tilt_deg=shaft_tilt_deg, precone_deg=precone_deg
        )
        return [
            (blade.nodes[0], blade.nodes[-1], blade)
            for blade in geometry.members_of_kind("blade")
        ]

    def test_blade_two_points_to_negative_y_at_zero_azimuth(self) -> None:
        """Positive azimuth rotates about downwind +x, so blade 2 sits at -y."""
        blades = self._blades(shaft_tilt_deg=0.0, precone_deg=0.0)
        assert blades[0][1][2] > blades[0][0][2]  # blade 1 points up
        assert blades[1][1][1] < 0.0
        assert blades[2][1][1] > 0.0

    def test_flap_points_downwind(self) -> None:
        """Positive flap is the ElastoDyn out-of-plane (+x, downwind) direction."""
        for _, _, blade in self._blades():
            assert blade.flap_dir is not None
            assert blade.flap_dir[0] > 0.9

    def test_edge_points_against_rotation(self) -> None:
        """Positive edge points toward the trailing edge (against rotation)."""
        axial = np.array([1.0, 0.0, 0.0])
        for root, tip, blade in self._blades(shaft_tilt_deg=0.0, precone_deg=0.0):
            assert blade.edge_dir is not None
            radial = (tip - root) / np.linalg.norm(tip - root)
            rotation_dir = np.cross(axial, radial)
            np.testing.assert_allclose(blade.edge_dir, -rotation_dir, atol=1e-12)

    def test_flap_edge_radial_is_right_handed(self) -> None:
        """(flap, edge, radial) forms a right-handed triad like ElastoDyn (x, y, z)."""
        for root, tip, blade in self._blades():
            assert blade.flap_dir is not None and blade.edge_dir is not None
            radial = (tip - root) / np.linalg.norm(tip - root)
            np.testing.assert_allclose(
                np.cross(blade.flap_dir, blade.edge_dir), radial, atol=1e-12
            )

    def test_negative_precone_cones_blades_upwind(self) -> None:
        """An OpenFAST-negative precone cones every blade tip upwind (-x)."""
        for root, tip, _ in self._blades(shaft_tilt_deg=0.0, precone_deg=-2.5):
            assert tip[0] < root[0]
        for root, tip, _ in self._blades(shaft_tilt_deg=0.0, precone_deg=2.5):
            assert tip[0] > root[0]

    def test_negative_tilt_is_nose_up(self) -> None:
        """An OpenFAST-negative shaft tilt swings the lower blade tips upwind."""
        blades = self._blades(shaft_tilt_deg=-5.0, precone_deg=0.0)
        assert blades[0][1][0] > blades[0][0][0]  # upper tip leans downwind
        for root, tip, _ in blades[1:]:
            assert tip[0] < root[0]  # lower tips swing away from the tower

    def test_default_rotor_clears_the_tower(self) -> None:
        """The default (upwind 5-MW-like) rotor keeps the lower tips upwind."""
        for build in (semisub_geometry, spar_geometry, tlp_geometry, onshore_geometry):
            geometry = build()
            hub_x = geometry.members_of_kind("nacelle")[0].nodes[-1][0]
            assert hub_x < 0.0  # the hub overhangs upwind of the tower
            for blade in geometry.members_of_kind("blade")[1:]:
                # Lower tips lean upwind of the hub, away from the tower.
                assert blade.nodes[-1][0] < hub_x


class TestSemisubGeometry:
    """Tests for the semi-submersible preset."""

    def test_has_all_component_kinds(self) -> None:
        """A semi-sub has tower, blades, platform, nacelle, and mooring members."""
        geometry = semisub_geometry()
        kinds = {member.kind for member in geometry.members}
        assert {"tower", "blade", "platform", "nacelle", "mooring"} <= kinds

    def test_blade_count_matches(self) -> None:
        """There is exactly one blade member per blade."""
        geometry = semisub_geometry(n_blades=3)
        assert len(geometry.members_of_kind("blade")) == 3
        assert geometry.n_blades == 3

    def test_blades_carry_directions_and_span(self) -> None:
        """Blade members carry unit flap/edge directions and a 0..1 span."""
        geometry = semisub_geometry()
        for blade in geometry.members_of_kind("blade"):
            assert blade.flap_dir is not None
            assert blade.edge_dir is not None
            assert blade.span_fraction is not None
            np.testing.assert_allclose(np.linalg.norm(blade.flap_dir), 1.0)
            np.testing.assert_allclose(np.linalg.norm(blade.edge_dir), 1.0)
            # Flap and edge are orthogonal by construction.
            assert abs(float(blade.flap_dir @ blade.edge_dir)) < 1e-9
            assert blade.span_fraction[0] == 0.0
            np.testing.assert_allclose(blade.span_fraction[-1], 1.0)

    def test_tower_top_is_at_hub_height(self) -> None:
        """The tower top sits at the configured hub height on the centreline."""
        geometry = semisub_geometry(hub_height=120.0)
        np.testing.assert_allclose(geometry.tower_top, [0.0, 0.0, 120.0])
        assert geometry.characteristic_length == 120.0

    def test_mooring_fairlead_is_first_node(self) -> None:
        """Each mooring line starts at the shallow fairlead and ends deep at anchor."""
        geometry = semisub_geometry(fairlead_depth=14.0, water_depth=200.0)
        for mooring in geometry.members_of_kind("mooring"):
            fairlead, anchor = mooring.nodes[0], mooring.nodes[-1]
            assert fairlead[2] > anchor[2]
            np.testing.assert_allclose(anchor[2], -200.0)
            radius = np.hypot(*anchor[:2])
            assert radius > np.hypot(*fairlead[:2])


class TestSparGeometry:
    """Tests for the spar preset."""

    def test_spar_is_deep_and_moored(self) -> None:
        """A spar has a deep single column and three mooring lines."""
        geometry = spar_geometry(draft=120.0)
        assert geometry.system_type is SystemType.SPAR
        assert len(geometry.members_of_kind("mooring")) == 3
        spar = next(m for m in geometry.members if m.name == "Spar")
        assert float(spar.nodes[:, 2].min()) <= -120.0


class TestMonopileGeometry:
    """Tests for the bottom-fixed monopile preset."""

    def test_monopile_has_no_mooring(self) -> None:
        """A fixed monopile carries no mooring lines."""
        geometry = monopile_geometry()
        assert geometry.system_type is SystemType.FIXED
        assert geometry.members_of_kind("mooring") == []

    def test_monopile_extends_below_mudline(self) -> None:
        """The monopile reaches below the seabed (negative z)."""
        geometry = monopile_geometry(embedment=25.0)
        pile = next(m for m in geometry.members if m.name == "Monopile")
        assert float(pile.nodes[:, 2].min()) <= -25.0

    def test_offshore_monopile_seabed_sits_below_the_water_surface(self) -> None:
        """The offshore MONOPILE drops its mudline to -water_depth, below z=0.

        The legacy bottom-fixed preset keeps the seabed at the reference plane,
        but the explicit offshore type draws still water at ``z = 0``; the mudline
        ring and pile bottom must therefore lie below the surface and the pile
        must cross ``z = 0`` up to the tower base.
        """
        geometry = default_geometry(SystemType.MONOPILE)
        assert geometry.system_type is SystemType.MONOPILE
        water_depth = float(geometry.metadata["water_depth"])
        assert water_depth > 0.0
        mudline = next(m for m in geometry.members if m.name == "Mudline")
        np.testing.assert_allclose(mudline.nodes[:, 2], -water_depth)
        pile = next(m for m in geometry.members if m.name == "Monopile")
        assert float(pile.nodes[:, 2].min()) < -water_depth  # embedded below seabed
        assert float(pile.nodes[:, 2].max()) > 0.0  # crosses the waterline


class TestDefaultGeometry:
    """Tests for the system-type dispatch."""

    def test_dispatch_returns_matching_type(self) -> None:
        """Every system type dispatches to a geometry of its own support structure."""
        for system_type in SystemType:
            geometry = default_geometry(system_type)
            assert geometry.system_type is system_type
            assert geometry.tower_length > 0.0
            assert geometry.members_of_kind("tower")
            assert len(geometry.members_of_kind("blade")) == geometry.n_blades

    def test_geometry_is_a_dataclass_instance(self) -> None:
        """The builder returns a populated FowtGeometry."""
        geometry = default_geometry(SystemType.SEMI)
        assert isinstance(geometry, FowtGeometry)
        assert geometry.tower_length > 0.0
        assert len(geometry.members) > 0

    def test_members_of_kind_filters(self) -> None:
        """members_of_kind returns only the requested kind."""
        geometry = default_geometry(SystemType.SEMI)
        towers = geometry.members_of_kind("tower")
        assert len(towers) == 1
        assert all(m.kind == "tower" for m in towers)


class TestOnshoreGeometry:
    """Tests for the land-based geometry."""

    def test_tower_rises_from_the_ground_with_no_water(self) -> None:
        """The onshore tower starts at z=0 and the geometry has no water or mooring."""
        geometry = onshore_geometry()
        assert geometry.system_type is SystemType.LAND
        tower = geometry.members_of_kind("tower")[0]
        assert float(tower.nodes[:, 2].min()) == 0.0
        assert not geometry.members_of_kind("mooring")
        assert "water_depth" not in geometry.metadata


class TestTlpGeometry:
    """Tests for the tension-leg-platform geometry."""

    def test_tendons_are_vertical_and_taut(self) -> None:
        """Each TLP tendon runs straight down: fairlead and anchor share x, y."""
        geometry = tlp_geometry()
        tendons = geometry.members_of_kind("mooring")
        assert len(tendons) == 3
        for tendon in tendons:
            fairlead, anchor = tendon.nodes[0], tendon.nodes[-1]
            np.testing.assert_allclose(fairlead[:2], anchor[:2], atol=1e-9)  # vertical
            assert anchor[2] < fairlead[2]  # anchor is deeper

    def test_is_offshore(self) -> None:
        """A TLP is moored offshore (carries a water depth)."""
        assert "water_depth" in tlp_geometry().metadata


class TestJacketGeometry:
    """Tests for the lattice-jacket geometry."""

    def test_is_a_braced_lattice(self) -> None:
        """The jacket has legs and X-braces and reaches from seabed to above water."""
        geometry = jacket_geometry(n_legs=4, n_bays=4)
        platform = geometry.members_of_kind("platform")
        legs = [m for m in platform if m.name.startswith("Leg")]
        braces = [m for m in platform if m.name.startswith("Brace")]
        assert len(legs) == 4
        assert len(braces) == 4 * 4 * 2  # two crossing braces per leg per bay
        z = np.concatenate([m.nodes[:, 2] for m in platform])
        assert z.min() < 0.0 < z.max()  # spans the waterline

    def test_legs_are_battered_inward(self) -> None:
        """The legs taper from a wide base to a narrow top."""
        geometry = jacket_geometry()
        leg = next(m for m in geometry.members if m.name == "Leg 1")
        base_radius = float(np.hypot(leg.nodes[0, 0], leg.nodes[0, 1]))
        top_radius = float(np.hypot(leg.nodes[-1, 0], leg.nodes[-1, 1]))
        assert base_radius > top_radius

    def test_mudline_stays_at_seabed_when_piles_are_embedded(self) -> None:
        """Embedment drives only the pile toe below the mudline, not the seabed ring.

        The displayed mudline (and the lowest braced bay) must sit at the true
        seabed ``-water_depth``; only each leg's embedded toe reaches deeper.
        """
        water_depth, embedment = 50.0, 20.0
        geometry = jacket_geometry(water_depth=water_depth, embedment=embedment)
        mudline = next(m for m in geometry.members if m.name == "Mudline")
        np.testing.assert_allclose(mudline.nodes[:, 2], -water_depth)
        # No brace dips below the mudline.
        for brace in (m for m in geometry.members if m.name.startswith("Brace")):
            assert float(brace.nodes[:, 2].min()) >= -water_depth - 1e-9
        # Each leg's toe reaches the embedded depth, below the seabed.
        leg = next(m for m in geometry.members if m.name == "Leg 1")
        np.testing.assert_allclose(
            float(leg.nodes[:, 2].min()), -water_depth - embedment
        )
