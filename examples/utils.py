"""
Compatibility shim for the NPAP example notebooks.

The dataset helper now lives in the installable package as :mod:`npap.datasets`,
so the documentation examples work after a plain ``pip install npap`` without
needing a copy of this file. New code should import from there directly:

    from npap.datasets import fetch_pypsa_eur

This module is kept so that older copies of the notebooks keep running.
"""

from __future__ import annotations

from npap.datasets import (
    PYPSA_EUR_FILES as FILES,
)
from npap.datasets import (
    ZENODO_BASE,
    ZENODO_RECORD,
    clear_data_home,
    fetch_pypsa_eur,
    get_data_home,
)

#: Deprecated alias kept for the original notebook API.
download_and_preprocess_zenodo_data = fetch_pypsa_eur

__all__ = [
    "FILES",
    "ZENODO_BASE",
    "ZENODO_RECORD",
    "clear_data_home",
    "download_and_preprocess_zenodo_data",
    "fetch_pypsa_eur",
    "get_data_home",
]
