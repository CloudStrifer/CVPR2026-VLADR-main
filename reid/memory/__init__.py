"""Persistent summaries for category-progressive ReID (no image replay)."""

from .ecpm import IdentityPrototypeAccumulator, IdentityPrototypeMemory
from .ecpm_modes import ECPMMemory
from .finch_modes import FinchConfig

__all__ = ['IdentityPrototypeAccumulator', 'IdentityPrototypeMemory', 'ECPMMemory', 'FinchConfig']
