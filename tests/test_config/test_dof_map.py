# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.config.dof_map — DOF description classification."""

from __future__ import annotations

import pytest

from vane.config.dof_map import DofCategory, classify_dof


class TestClassifyDof:
    """Unit tests for classify_dof."""

    @pytest.mark.parametrize(
        ("token", "category"),
        [
            ("DOF_Sg", DofCategory.PLATFORM_SURGE),
            ("DOF_Sw", DofCategory.PLATFORM_SWAY),
            ("DOF_Hv", DofCategory.PLATFORM_HEAVE),
            ("DOF_R", DofCategory.PLATFORM_ROLL),
            ("DOF_P", DofCategory.PLATFORM_PITCH),
            ("DOF_Y", DofCategory.PLATFORM_YAW),
            ("DOF_TFA1", DofCategory.TOWER_FORE_AFT_1),
            ("DOF_TSS1", DofCategory.TOWER_SIDE_SIDE_1),
            ("DOF_TFA2", DofCategory.TOWER_FORE_AFT_2),
            ("DOF_TSS2", DofCategory.TOWER_SIDE_SIDE_2),
            ("DOF_Yaw", DofCategory.NACELLE_YAW),
            ("DOF_GeAz", DofCategory.GENERATOR_AZIMUTH),
            ("DOF_DrTr", DofCategory.DRIVETRAIN_TORSION),
            ("DOF_RFrl", DofCategory.ROTOR_FURL),
            ("DOF_TFrl", DofCategory.TAIL_FURL),
            ("DOF_Teet", DofCategory.TEETER),
        ],
    )
    def test_fixed_tokens(self, token: str, category: DofCategory) -> None:
        """Each fixed DOF token maps to its physical category."""
        desc = f"ED some DOF (internal DOF index = {token}), m"
        assert classify_dof(desc).category is category

    def test_roll_not_confused_with_rotor_furl(self) -> None:
        """`DOF_R` (roll) is not matched as a prefix of `DOF_RFrl`."""
        assert (
            classify_dof("ED platform roll (internal DOF index = DOF_R), rad").category
            is DofCategory.PLATFORM_ROLL
        )
        assert (
            classify_dof("ED rotor-furl (internal DOF index = DOF_RFrl), rad").category
            is DofCategory.ROTOR_FURL
        )

    @pytest.mark.parametrize(
        ("token", "category", "blade"),
        [
            ("DOF_BF(1,1)", DofCategory.BLADE_FLAP_1, 1),
            ("DOF_BF(2,2)", DofCategory.BLADE_FLAP_2, 2),
            ("DOF_BE(3,1)", DofCategory.BLADE_EDGE_1, 3),
            ("DOF_BP(2)", DofCategory.BLADE_PITCH, 2),
        ],
    )
    def test_blade_tokens(self, token: str, category: DofCategory, blade: int) -> None:
        """Blade DOF tokens map to their category and carry the blade index."""
        info = classify_dof(f"ED blade DOF (internal DOF index = {token}), m")
        assert info.category is category
        assert info.blade == blade

    def test_module_and_velocity_flags(self) -> None:
        """The module prefix and velocity marker are extracted."""
        disp = classify_dof("ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m")
        assert disp.module == "ED"
        assert disp.is_velocity is False

        vel = classify_dof(
            "ED First time derivative of 1st tower fore-aft "
            "(internal DOF index = DOF_TFA1), m/s"
        )
        assert vel.is_velocity is True
        assert vel.category is DofCategory.TOWER_FORE_AFT_1

    def test_beamdyn_module_prefix(self) -> None:
        """A BeamDyn instance prefix is captured as the module."""
        info = classify_dof("BD_1 finite element node 2 translational X, m")
        assert info.module == "BD_1"

    @pytest.mark.parametrize(
        ("kind", "axis", "category", "component"),
        [
            ("translational", "X", DofCategory.BLADE_FLAP, "TX"),
            ("rotational", "Y", DofCategory.BLADE_FLAP, "RY"),
            ("translational", "Y", DofCategory.BLADE_EDGE, "TY"),
            ("rotational", "X", DofCategory.BLADE_EDGE, "RX"),
            ("rotational", "Z", DofCategory.BLADE_TORSION, "RZ"),
            ("translational", "Z", DofCategory.BLADE_AXIAL, "TZ"),
        ],
    )
    def test_beamdyn_nodal_states(
        self, kind: str, axis: str, category: DofCategory, component: str
    ) -> None:
        """BeamDyn nodal states map onto the blade-frame flap/edge/torsion/axial."""
        info = classify_dof(
            f"BD_2 finite element node 7 (number of elements = 1; element order = 5) "
            f"{kind} displacement in {axis}, m"
        )
        assert info.category is category
        assert info.module == "BD_2"
        assert info.blade == 2
        assert info.node == 7
        assert info.component == component
        assert info.is_velocity is False

    @pytest.mark.parametrize(
        "description",
        [
            "BD_3 First time derivative of finite element node 4 (number of "
            "elements = 1; element order = 5) translational displacement in Y, m/s",
            "First time derivative of BD_3 finite element node 4 (number of "
            "elements = 1; element order = 5) translational displacement in Y, m/s",
        ],
    )
    def test_beamdyn_velocity_spellings(self, description: str) -> None:
        """Both derivative spellings are the velocity of the same nodal state."""
        info = classify_dof(description)
        assert info.category is DofCategory.BLADE_EDGE
        assert info.blade == 3
        assert info.node == 4
        assert info.is_velocity is True

    def test_bare_beamdyn_prefix_has_no_blade(self) -> None:
        """A bare ``BD`` prefix is a blade state with no blade index."""
        info = classify_dof(
            "BD finite element node 2 (number of elements = 1; element order = 5) "
            "rotational displacement in Z, rad"
        )
        assert info.category is DofCategory.BLADE_TORSION
        assert info.module == "BD"
        assert info.blade is None

    def test_beamdyn_categories_follow_unknown(self) -> None:
        """The BeamDyn categories are appended after UNKNOWN, keeping prior order."""
        members = list(DofCategory)
        assert members.index(DofCategory.UNKNOWN) == 20
        assert members[21:] == [
            DofCategory.BLADE_FLAP,
            DofCategory.BLADE_EDGE,
            DofCategory.BLADE_TORSION,
            DofCategory.BLADE_AXIAL,
        ]

    def test_elastodyn_states_carry_no_node(self) -> None:
        """ElastoDyn states have no BeamDyn node or component."""
        info = classify_dof("ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m")
        assert info.node is None
        assert info.component is None

    def test_unknown_description(self) -> None:
        """A description without a known DOF token is UNKNOWN with no blade."""
        info = classify_dof("AD B1N001 axial induction, -")
        assert info.category is DofCategory.UNKNOWN
        assert info.blade is None
