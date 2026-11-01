# SPDX-FileCopyrightText: Contributors to NPAP
# SPDX-License-Identifier: MIT

"""
Example datasets used throughout the NPAP documentation.

The datasets are not shipped with the package. They are downloaded on first
use and cached locally, so every example in the documentation runs after a
plain ``pip install npap`` without any manual data preparation.

Data source
-----------
Xiong, B., Fioriti, D., Neumann, F., Riepin, I., & Brown, T. (2026).
Prebuilt electricity network for PyPSA-Eur based on OpenStreetMap data.
Zenodo. https://doi.org/10.5281/ZENODO.18619025

Described in: Xiong, B., Fioriti, D., Neumann, F., Riepin, I., & Brown, T.
(2025). Modelling the high-voltage grid using open data for Europe and beyond.
Scientific Data, 12(1). https://doi.org/10.1038/s41597-025-04550-7

The underlying data derives from OpenStreetMap and is licensed under the Open
Database License (ODbL). Please credit OpenStreetMap contributors and cite the
references above when using it.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

import pandas as pd

from npap._logging import LogCategory, log_info

#: Zenodo record holding the prebuilt PyPSA-Eur network.
ZENODO_RECORD: str = "18619025"

#: Zenodo file-listing endpoint for :data:`ZENODO_RECORD`.
ZENODO_BASE: str = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"

#: CSV files that make up the PyPSA-Eur network.
PYPSA_EUR_FILES: tuple[str, ...] = (
    "buses.csv",
    "lines.csv",
    "transformers.csv",
    "converters.csv",
    "links.csv",
)

#: Environment variable overriding the download cache location.
DATA_HOME_ENV: str = "NPAP_DATA_HOME"

_BAR_WIDTH: int = 40


# ---------------------------------------------------------------------------
# Cache location
# ---------------------------------------------------------------------------


def get_data_home(data_dir: str | Path | None = None) -> Path:
    """
    Return the directory used to cache downloaded datasets.

    Resolution order: the explicit `data_dir` argument, then the
    ``NPAP_DATA_HOME`` environment variable, then a per-user cache directory
    (``$XDG_CACHE_HOME/npap`` or ``~/.cache/npap`` on Linux and macOS,
    ``%LOCALAPPDATA%`` + ``/npap`` on Windows).

    The directory is created if it does not exist.

    Parameters
    ----------
    data_dir : str or Path or None
        Explicit cache directory. Overrides every other source.

    Returns
    -------
    Path
        Existing cache directory.
    """
    if data_dir is not None:
        home = Path(data_dir)
    elif os.environ.get(DATA_HOME_ENV):
        home = Path(os.environ[DATA_HOME_ENV])
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        home = Path(os.environ["LOCALAPPDATA"]) / "npap"
    else:
        cache_root = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
        home = Path(cache_root) / "npap"

    home = home.expanduser()
    home.mkdir(parents=True, exist_ok=True)
    return home


def clear_data_home(data_dir: str | Path | None = None) -> None:
    """
    Delete the dataset cache so the next fetch downloads everything again.

    Parameters
    ----------
    data_dir : str or Path or None
        Cache directory to clear. Defaults to :func:`get_data_home`.
    """
    home = get_data_home(data_dir)
    shutil.rmtree(home, ignore_errors=True)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _fetch_remote_sizes() -> dict[str, int]:
    """
    Query the Zenodo file listing for file sizes.

    Returns
    -------
    dict[str, int]
        Mapping of filename to size in bytes, empty if the request fails.
    """
    try:
        with urllib.request.urlopen(ZENODO_BASE) as resp:
            metadata = json.loads(resp.read())
        # v3 API returns {"entries": [...]}, v1 returns a bare list
        entries = metadata.get("entries", metadata) if isinstance(metadata, dict) else metadata
        return {e["key"]: e["size"] for e in entries if "key" in e and "size" in e}
    except Exception:
        return {}


def _format_size(n_bytes: int) -> str:
    """
    Format a byte count as a human-readable string.

    Parameters
    ----------
    n_bytes : int
        Size in bytes.

    Returns
    -------
    str
        Size rendered as ``'X.X MB'`` or ``'X KB'``.
    """
    if n_bytes >= 1024**2:
        return f"{n_bytes / 1024**2:.1f} MB"
    return f"{n_bytes / 1024:.0f} KB"


def _cache_summary(paths: dict[str, Path], home: Path) -> str:
    """
    Build the message describing where the dataset lives and how to remove it.

    Parameters
    ----------
    paths : dict[str, Path]
        Mapping of filename to local path.
    home : Path
        Cache directory holding them.

    Returns
    -------
    str
        Multi-line message naming the location, the total size on disk and the
        call that clears it.
    """
    total = sum(p.stat().st_size for p in paths.values() if p.exists())
    return (
        f"Dataset cached in {home} ({_format_size(total)}).\n"
        "These files are kept for future runs and are never removed "
        "automatically. To delete them, run:\n"
        "    from npap.datasets import clear_data_home; clear_data_home()"
    )


def _download_file(url: str, dest: Path, total_bytes: int = 0, show_progress: bool = True) -> None:
    """
    Download a single file, optionally rendering a progress bar.

    Parameters
    ----------
    url : str
        Source URL.
    dest : Path
        Destination path. Written atomically via a temporary file.
    total_bytes : int
        Expected size, used for the progress bar. 0 disables the percentage.
    show_progress : bool
        Whether to print progress to stdout.
    """
    hook = None
    if show_progress:
        last = [0.0]

        def hook(block_count: int, block_size: int, _reported: int) -> None:
            now = time.monotonic()
            if now - last[0] < 0.1:
                return
            last[0] = now
            done = block_count * block_size
            if total_bytes > 0:
                done = min(done, total_bytes)
                pct = done / total_bytes * 100
                filled = int(pct / 100 * _BAR_WIDTH)
                bar = "=" * filled + "-" * (_BAR_WIDTH - filled)
                line = f"  {dest.name}: [{bar}] {pct:5.1f}%"
            else:
                line = f"  {dest.name}: {_format_size(done)}"
            print(line, end="\r", flush=True)

    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    tmp.replace(dest)

    if show_progress:
        size = _format_size(dest.stat().st_size)
        print(f"  {dest.name}: downloaded ({size}){' ' * _BAR_WIDTH}")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _harmonise_columns(data_files: dict[str, Path]) -> dict[str, Path]:
    """
    Rename PyPSA-Eur/Zenodo columns to the names NPAP's loaders expect.

    The operation is idempotent: once renamed, a second call leaves the files
    unchanged.

    Renames applied
    ---------------
    buses.csv
        ``x`` -> ``lon``, ``y`` -> ``lat``, ``v_nom`` -> ``voltage``
    lines.csv
        ``type`` -> ``line_type``. Zenodo's ``type`` is a cable hardware
        specification; renaming avoids colliding with NPAP's own edge-type
        attribute, which the visualizer expects to be ``"line"``, ``"trafo"``
        or ``"dc_link"``.
    transformers.csv
        ``voltage_bus0`` -> ``primary_voltage``,
        ``voltage_bus1`` -> ``secondary_voltage``, plus a placeholder
        reactance when the dataset does not carry one.

    Parameters
    ----------
    data_files : dict[str, Path]
        Mapping of filename to local path.

    Returns
    -------
    dict[str, Path]
        The same mapping; files are rewritten in place.
    """
    buses_path = data_files["buses.csv"]
    buses = pd.read_csv(buses_path)
    buses = buses.rename(columns={"x": "lon", "y": "lat", "v_nom": "voltage"})
    buses.to_csv(buses_path, index=False)

    # The geometry column is single-quoted in the raw Zenodo CSVs, e.g.
    # 'LINESTRING (x y, x y)'. Rewriting it with the default double-quote
    # quotechar would leave its internal commas unprotected and shift every
    # later column when the loaders re-read the file. NPAP never uses the
    # geometry, so dropping it removes the quoting mismatch entirely.
    lines_path = data_files["lines.csv"]
    lines = pd.read_csv(lines_path, quotechar="'")
    lines = lines.rename(columns={"type": "line_type"})
    lines = lines.drop(columns=["geometry"], errors="ignore")
    lines.to_csv(lines_path, index=False)

    trafos_path = data_files["transformers.csv"]
    trafos = pd.read_csv(trafos_path, quotechar="'")
    trafos = trafos.rename(
        columns={"voltage_bus0": "primary_voltage", "voltage_bus1": "secondary_voltage"}
    )
    trafos = trafos.drop(columns=["geometry"], errors="ignore")
    if "x" not in trafos.columns:
        trafos["x"] = 20
    trafos.to_csv(trafos_path, index=False)

    return data_files


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_pypsa_eur(
    data_dir: str | Path | None = None,
    show_progress: bool = True,
) -> dict[str, Path]:
    """
    Download the prebuilt PyPSA-Eur network and prepare it for NPAP.

    The European high-voltage grid (roughly 6,800 buses across 35 countries),
    derived from OpenStreetMap. Files are cached after the first call, so
    subsequent calls return immediately. Column names are harmonised with
    NPAP's loader conventions, so the returned paths can be passed straight to
    :meth:`~npap.PartitionAggregatorManager.load_data`.

    Parameters
    ----------
    data_dir : str or Path or None
        Cache directory. Defaults to :func:`get_data_home`.
    show_progress : bool
        Whether to print download progress. Set to False in scripts and tests.

    Returns
    -------
    dict[str, Path]
        Mapping of filename to local path, with keys ``"buses.csv"``,
        ``"lines.csv"``, ``"transformers.csv"``, ``"converters.csv"`` and
        ``"links.csv"``.

    Notes
    -----
    Requires an internet connection on first use. The data is licensed under
    the ODbL; see the module docstring for attribution and citation details.

    Examples
    --------
    >>> import npap
    >>> from npap.datasets import fetch_pypsa_eur
    >>> files = fetch_pypsa_eur()
    >>> manager = npap.PartitionAggregatorManager()
    >>> graph = manager.load_data(
    ...     strategy="csv_files",
    ...     node_file=str(files["buses.csv"]),
    ...     edge_file=str(files["lines.csv"]),
    ... )
    """
    home = get_data_home(data_dir)
    targets = {name: home / name for name in PYPSA_EUR_FILES}

    missing = [name for name, path in targets.items() if not path.exists()]
    if not missing:
        log_info(f"PyPSA-Eur dataset already cached in {home}", LogCategory.INPUT)
        if show_progress:
            print(_cache_summary(targets, home))
        return targets

    if show_progress:
        print(f"Fetching PyPSA-Eur network data from Zenodo into {home}")

    remote_sizes = _fetch_remote_sizes()
    for name in missing:
        _download_file(
            url=f"{ZENODO_BASE}/{name}/content",
            dest=targets[name],
            total_bytes=remote_sizes.get(name, 0),
            show_progress=show_progress,
        )

    _harmonise_columns(targets)
    log_info(f"PyPSA-Eur dataset ready in {home}", LogCategory.INPUT)
    if show_progress:
        print(_cache_summary(targets, home))
    return targets
