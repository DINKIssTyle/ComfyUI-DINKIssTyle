"""Plug-in model families: one FamilyDescription per family, read by the GUI's generic paths.

See description.py. Existing families (Klein, Krea 2, MiniMax H3) are not described here by design.
"""
from fizgig.families.description import (  # noqa: F401
    FamilyDescription, LoRAFormat, ModelFile, SamplingSettings, SpeedLoRA,
)


def __getattr__(name):
    """The registry's names, loaded on first use: importing the package (for launch / checks, which every family's
    Start uses) must not depend on every family description loading."""
    if name in ("FAMILIES", "by_arch_id", "by_gui_label", "get", "training_families"):
        from fizgig.families import registry
        return getattr(registry, name)
    raise AttributeError(name)
