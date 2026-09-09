"""Contracts for registry-backed, immutable host configuration."""

import json
import math
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from pathlib import Path
from types import ModuleType

import pytest


def configuration_modules() -> tuple[ModuleType, ModuleType]:
    """Load the public contracts so missing implementations fail explicitly."""
    import importlib.util

    for name in ("veris.configuration", "veris.physical_constants"):
        assert importlib.util.find_spec(name) is not None, f"missing {name}"
    return (
        importlib.import_module("veris.configuration"),
        importlib.import_module("veris.physical_constants"),
    )


def test_registry_defaults_are_disjoint_complete_and_frozen() -> None:
    config, physical = configuration_modules()
    legacy = json.loads(
        (
            Path(__file__).parent / "reference_data/configuration_pre_dataclass.json"
        ).read_text()
    )["defaults"]

    assert not config.SETTINGS.keys() & physical.PHYSICALCONSTANTS.keys()
    combined = {**config.SETTINGS, **physical.PHYSICALCONSTANTS}
    assert legacy.keys() <= combined.keys()
    for name, value in legacy.items():
        assert combined[name].default == value, name
    for cls, registry in (
        (config.Settings, config.SETTINGS),
        (physical.PhysicalConstants, physical.PHYSICALCONSTANTS),
    ):
        instance = cls()
        assert is_dataclass(instance)
        assert {field.name for field in fields(instance)} == registry.keys()
        assert hash(instance) == hash(cls())
        for name, metadata in registry.items():
            assert isinstance(metadata, tuple)
            assert metadata.description
            assert isinstance(getattr(instance, name), metadata.type)
            assert getattr(instance, name) == metadata.default
        with pytest.raises(FrozenInstanceError):
            setattr(instance, next(iter(registry)), 1)


def test_replace_recomputes_dependencies_and_preserves_rounded_constants() -> None:
    config, physical = configuration_modules()
    settings = replace(config.Settings(), deltatDyn=600, deltatTherm=900, nITC=3)
    assert settings.recip_deltatDyn == 1 / 600
    assert settings.recip_deltatTherm == 1 / 900
    assert settings.recip_nITC == 1 / 3
    constants = replace(
        physical.PhysicalConstants(),
        rhoIce=910,
        rhoSnow=300,
        rhoFresh=999,
        rhoSea=1020,
        waterTurnAngle=30,
        h0=0.4,
        h0_south=0.25,
        lhFusion=334001,
    )
    assert constants.rhoIce2rhoSnow == 910 / 300
    assert constants.rhoIce2rhoFresh == 910 / 999
    assert constants.rhoFresh2rhoSnow == 999 / 300
    assert constants.recip_rhoFresh == 1 / 999
    assert constants.recip_rhoSea == 1 / 1020
    assert constants.sinWat == pytest.approx(0.5)
    assert constants.cosWat == pytest.approx(math.sqrt(3) / 2)
    assert constants.recip_h0 == 2.5
    assert constants.recip_h0_south == 4
    assert constants.lhSublim == constants.lhFusion + constants.lhEvap
    assert constants.rgas == 8314.47
    assert constants.rdair == 287.042
    assert constants.rwv == 461.505
    assert constants.latvap == 2501000
    with pytest.raises((TypeError, ValueError), match="init=False"):
        replace(settings, recip_deltatDyn=2)


@pytest.mark.parametrize(
    "name,value",
    [
        ("deltatDyn", 0),
        ("deltatTherm", -1),
        ("nITC", 0),
        ("nEVPsteps", -1),
        ("nITC", 1.5),
        ("nITC", True),
        ("useEVP", 1),
        ("evpAlpha", 0),
        ("evpBeta", 0),
        ("deltatDyn", float("nan")),
        ("deltatDyn", float("inf")),
        ("zref", 0),
        ("Area_reg", -1),
        ("eps2", 0),
    ],
)
def test_invalid_settings_are_rejected(name: str, value: object) -> None:
    config, _ = configuration_modules()
    with pytest.raises((TypeError, ValueError), match=name):
        config.Settings(**{name: value})


@pytest.mark.parametrize(
    "name,value",
    [
        ("rhoIce", 0),
        ("rhoSea", 0),
        ("rhoSnow", -1),
        ("h0", 0),
        ("rhoFresh", True),
        ("cpAir", "1005"),
        ("waterTurnAngle", float("nan")),
        ("gravity", float("inf")),
        ("mwdair", 0),
        ("PlasDefCoeff", 0),
    ],
)
def test_invalid_constants_are_rejected(name: str, value: object) -> None:
    _, physical = configuration_modules()
    with pytest.raises((TypeError, ValueError), match=name):
        physical.PhysicalConstants(**{name: value})


def test_derived_nonfinite_values_are_rejected() -> None:
    """Finite inputs must not produce infinite dependent constants or reciprocals."""
    config, physical = configuration_modules()
    with pytest.raises(ValueError, match="recip_deltatDyn"):
        config.Settings(deltatDyn=1e-320)
    with pytest.raises(ValueError, match="lhSublim"):
        physical.PhysicalConstants(lhFusion=1e308, lhEvap=1e308)


def test_evp_print_control_is_instance_owned() -> None:
    """Residual reporting is configurable without a mutable module global."""
    from veris.configuration import SETTINGS, Settings

    assert "printEvpResidual" in SETTINGS
    assert Settings().printEvpResidual is False
    assert replace(Settings(), printEvpResidual=True).printEvpResidual is True
