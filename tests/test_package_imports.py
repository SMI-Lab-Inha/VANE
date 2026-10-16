# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Smoke tests that confirm every declared VANE subpackage imports cleanly."""

import importlib
import pkgutil

import pytest

SUBPACKAGES = [
    "vane.campbell",
    "vane.config",
    "vane.io",
    "vane.modal",
    "vane.run",
    "vane.sysid",
    "vane.utils",
    "vane.viz",
]


class TestSubpackageImports:
    """Verify all declared subpackages are importable from a clean interpreter."""

    @pytest.mark.parametrize("name", SUBPACKAGES)
    def test_subpackage_imports(self, name: str) -> None:
        """Each subpackage must import without error and expose a module docstring."""
        module = importlib.import_module(name)
        assert module.__doc__ is not None
        assert module.__doc__.strip()


class TestRunReExports:
    """``vane.run`` re-exports every public name of its modules."""

    def test_every_module_name_is_re_exported(self) -> None:
        """Each name in a ``vane.run`` module's ``__all__`` is the same object there."""
        package = importlib.import_module("vane.run")
        exported = set(package.__all__)
        missing: list[str] = []
        for info in pkgutil.iter_modules(package.__path__):
            if info.name.startswith("_"):
                continue
            module = importlib.import_module(f"vane.run.{info.name}")
            for name in module.__all__:
                if name not in exported:
                    missing.append(f"{info.name}.{name}")
                else:
                    assert getattr(package, name) is getattr(module, name)
        assert not missing, f"not re-exported from vane.run: {missing}"

    def test_curve_power_coefficient_is_public(self) -> None:
        """The cp reader used by the sweep tutorial imports from ``vane.run``."""
        from vane.run import curve_power_coefficient
        from vane.run.auto_config import curve_power_coefficient as defined

        assert curve_power_coefficient is defined
