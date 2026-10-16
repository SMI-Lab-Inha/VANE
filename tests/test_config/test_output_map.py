# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.config.output_map — output channel classification."""

from __future__ import annotations

import pytest

from vane.config.output_map import (
    LOAD_CATEGORIES,
    OutputCategory,
    classify_output,
)


class TestClassifyMooring:
    """Tests for mooring tension channels (MAP++ and MoorDyn)."""

    def test_map_fairlead_tension(self) -> None:
        """MAP ``T[i]`` is a fairlead tension with the line index and unit."""
        info = classify_output("MAP T[1], [N]")
        assert info.category is OutputCategory.MOORING_FAIRLEAD_TENSION
        assert info.is_load
        assert info.component == 1
        assert info.module == "MAP"
        assert info.quantity == "tension"
        assert info.unit == "N"

    def test_map_anchor_tension(self) -> None:
        """MAP ``T_a[i]`` is an anchor tension (not confused with ``T[i]``)."""
        info = classify_output("MAP T_a[2], [N]")
        assert info.category is OutputCategory.MOORING_ANCHOR_TENSION
        assert info.component == 2

    def test_moordyn_fairlead_and_anchor(self) -> None:
        """MoorDyn FAIRTEN/ANCHTEN channels classify with their line index."""
        fair = classify_output("MD FAIRTEN3, (N)")
        anchor = classify_output("MD ANCHTEN3, (N)")
        assert fair.category is OutputCategory.MOORING_FAIRLEAD_TENSION
        assert fair.component == 3
        assert anchor.category is OutputCategory.MOORING_ANCHOR_TENSION
        assert anchor.component == 3

    def test_moordyn_node_tension(self) -> None:
        """A MoorDyn per-node line tension is a generic line tension."""
        info = classify_output("MD L1N4T, (N)")
        assert info.category is OutputCategory.MOORING_LINE_TENSION
        assert info.component == 1

    def test_moordyn_v2_line_and_end_tensions(self) -> None:
        """MoorDyn v2 ``LINE#N#T`` and ``LINE#TENA/B`` names classify as tensions."""
        node = classify_output("MD LINE1N15T, (N)")
        assert node.category is OutputCategory.MOORING_LINE_TENSION
        assert node.component == 1
        for end in ("MD LINE2TENA, (N)", "MD LINE2TENB, (N)"):
            info = classify_output(end)
            assert info.category is OutputCategory.MOORING_LINE_TENSION
            assert info.is_load
            assert info.component == 2


class TestClassifyStructuralLoads:
    """Tests for tower, blade, and shaft load channels."""

    def test_tower_base_moment_axis(self) -> None:
        """A tower-base moment carries its axis label and load flag."""
        info = classify_output("ED TwrBsMyt, (kN-m)")
        assert info.category is OutputCategory.TOWER_BASE_MOMENT
        assert info.axis == "y"
        assert info.quantity == "moment"
        assert info.unit == "kN-m"

    def test_yaw_bearing_moment(self) -> None:
        """A yaw-bearing moment is a tower-top moment with axis."""
        info = classify_output("ED YawBrMzp, (kN-m)")
        assert info.category is OutputCategory.TOWER_TOP_MOMENT
        assert info.axis == "z"

    def test_blade_root_moment_elastodyn(self) -> None:
        """An ElastoDyn blade-root moment carries axis and blade index."""
        info = classify_output("ED RootMxb1, (kN-m)")
        assert info.category is OutputCategory.BLADE_ROOT_MOMENT
        assert info.axis == "x"
        assert info.component == 1

    def test_blade_root_loads_coned_frame(self) -> None:
        """ElastoDyn directional root channels in the coned (c) frame are loads.

        Real ElastoDyn output lists name the directional channels in the coned
        frame (RootMxc1/RootFxc1), not only the blade frame (RootMxb1).
        """
        moment = classify_output("ED RootMyc1, (kN-m)")
        assert moment.category is OutputCategory.BLADE_ROOT_MOMENT
        assert moment.axis == "y"
        assert moment.component == 1
        force = classify_output("ED RootFzc3, (kN)")
        assert force.category is OutputCategory.BLADE_ROOT_FORCE
        assert force.is_load
        assert force.axis == "z"
        assert force.component == 3

    def test_blade_root_in_out_of_plane(self) -> None:
        """The in-plane/out-of-plane root-moment aliases are recognized."""
        for name in ("ED RootMIP2, (kN-m)", "ED RootMOoP2, (kN-m)"):
            info = classify_output(name)
            assert info.category is OutputCategory.BLADE_ROOT_MOMENT
            assert info.component == 2

    def test_blade_root_moment_beamdyn(self) -> None:
        """A BeamDyn blade-root moment carries blade index and axis."""
        info = classify_output("BD_1 B1RootMzr, (N-m)")
        assert info.category is OutputCategory.BLADE_ROOT_MOMENT
        assert info.component == 1
        assert info.axis == "z"
        assert info.module == "BD_1"

    def test_blade_root_force_beamdyn(self) -> None:
        """A BeamDyn blade-root force is a load (not dropped to UNKNOWN)."""
        info = classify_output("BD_1 B1RootFxr, (N)")
        assert info.category is OutputCategory.BLADE_ROOT_FORCE
        assert info.is_load
        assert info.component == 1
        assert info.axis == "x"

    def test_beamdyn_linearization_root_loads_take_blade_from_module(self) -> None:
        """BeamDyn .lin root loads (``BD_2 RootMxr``) take the blade from ``BD_n``."""
        moment = classify_output("BD_2 RootMxr, (N-m)")
        assert moment.category is OutputCategory.BLADE_ROOT_MOMENT
        assert moment.is_load
        assert moment.component == 2
        assert moment.axis == "x"
        assert moment.unit == "N-m"
        force = classify_output("BD_3 RootFzr, (N)")
        assert force.category is OutputCategory.BLADE_ROOT_FORCE
        assert force.is_load
        assert force.component == 3
        assert force.axis == "z"
        # Without an indexed module the blade is unknown, but it is still a load.
        bare = classify_output("BD RootMyr, (N-m)")
        assert bare.category is OutputCategory.BLADE_ROOT_MOMENT
        assert bare.component is None

    def test_blade_root_force_elastodyn(self) -> None:
        """An ElastoDyn blade-root force carries axis and blade index."""
        info = classify_output("ED RootFxb2, (kN)")
        assert info.category is OutputCategory.BLADE_ROOT_FORCE
        assert info.component == 2
        assert info.axis == "x"

    def test_shaft_torque(self) -> None:
        """Rotor torque is classified as shaft torque."""
        info = classify_output("ED RotTorq, (kN-m)")
        assert info.category is OutputCategory.SHAFT_TORQUE
        assert info.is_load


class TestClassifyNonLoads:
    """Tests for channels that are not structural loads."""

    @pytest.mark.parametrize(
        ("description", "category"),
        [
            ("ED PtfmSurge, (m)", OutputCategory.PLATFORM_MOTION),
            ("ED PtfmPitch, (deg)", OutputCategory.PLATFORM_MOTION),
            ("ED TTDspFA, (m)", OutputCategory.DEFLECTION),
            ("SEA Wave1Elev, (m)", OutputCategory.WAVE_ELEVATION),
        ],
    )
    def test_non_load_categories(
        self, description: str, category: OutputCategory
    ) -> None:
        """Motion, deflection, and wave channels are recognized but not loads."""
        info = classify_output(description)
        assert info.category is category
        assert not info.is_load

    def test_unknown_channel(self) -> None:
        """An unrecognized channel is UNKNOWN and not a load, but keeps its unit."""
        info = classify_output("ED GenSpeed, (rpm)")
        assert info.category is OutputCategory.UNKNOWN
        assert not info.is_load
        assert info.unit == "rpm"
        assert info.module == "ED"


class TestLoadCategories:
    """Tests for the LOAD_CATEGORIES membership invariant."""

    def test_load_flag_matches_membership(self) -> None:
        """``is_load`` is true exactly when the category is in LOAD_CATEGORIES."""
        for description in (
            "MAP T[1], [N]",
            "ED TwrBsMxt, (kN-m)",
            "ED PtfmSurge, (m)",
            "ED GenSpeed, (rpm)",
        ):
            info = classify_output(description)
            assert info.is_load == (info.category in LOAD_CATEGORIES)

    def test_unknown_not_a_load(self) -> None:
        """UNKNOWN is never a load category."""
        assert OutputCategory.UNKNOWN not in LOAD_CATEGORIES
