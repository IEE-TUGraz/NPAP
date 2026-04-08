"""
Shared utilities for NPAP example notebooks.

Provides general-purpose helpers used across examples.

Data preparation
--------------
Xiong, B., Fioriti, D., Neumann, F., Riepin, I., & Brown, T. (2025).
Modelling the high-voltage grid using open data for Europe and beyond.
Scientific Data, 12(1). https://doi.org/10.1038/s41597-025-04550-7

Zenodo Dataset
--------------
Xiong, B., Fioriti, D., Neumann, F., Riepin, I., & Brown, T. (2026).
Prebuilt electricity network for PyPSA-Eur based on OpenStreetMap data.
Zenodo. https://doi.org/10.5281/ZENODO.18619025

Public API
----------
download_and_preprocess_zenodo_data : Download all network CSV files,
    apply column renames to match NPAP conventions, and return their paths.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZENODO_RECORD: str = "18619025"
ZENODO_BASE: str = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"

DATA_DIR: Path = Path(__file__).parent / "data"

FILES: list[str] = [
    "buses.csv",
    "lines.csv",
    "transformers.csv",
    "converters.csv",
    "links.csv",
]

_BAR_WIDTH: int = 40


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_zenodo_sizes() -> dict[str, int]:
    """
    Query the Zenodo file-listing endpoint for file sizes.

    Returns
    -------
    dict[str, int]
        Mapping of filename to size in bytes.
        Returns an empty dict if the request fails.
    """
    try:
        with urllib.request.urlopen(ZENODO_BASE) as resp:
            metadata = json.loads(resp.read())
        # v3 API: {"entries": [...]}  /  v1 API: [...]
        entries: list[dict] = (
            metadata.get("entries", metadata) if isinstance(metadata, dict) else metadata
        )
        return {e["key"]: e["size"] for e in entries if "key" in e and "size" in e}
    except Exception:
        return {}


def _fmt_size(n_bytes: int) -> str:
    """
    Format a byte count as a human-readable string.

    Parameters
    ----------
    n_bytes : int
        Size in bytes.

    Returns
    -------
    str
        Size formatted as ``'X.X MB'`` or ``'X KB'``.
    """
    if n_bytes >= 1024**2:
        return f"{n_bytes / 1024**2:.1f} MB"
    return f"{n_bytes / 1024:.0f} KB"


def _preprocess_zenodo_files(data_files: dict[str, Path]) -> dict[str, Path]:
    """
    Rename PyPSA-Eur/Zenodo CSV columns to match NPAP naming conventions.

    The Zenodo dataset uses column names that differ from what NPAP expects.
    This function applies the necessary renames in-place so the files can be
    passed directly to any NPAP loading strategy without further remapping.

    Renames applied
    ---------------
    buses.csv
        ``x`` → ``lon``, ``y`` → ``lat``, ``v_nom`` → ``voltage``
    lines.csv
        ``type`` → ``line_type``
        (Zenodo's ``type`` is a cable hardware spec string; renaming avoids
        collision with NPAP's internal edge-type attribute used by the
        visualizer, which expects ``type`` to be ``"line"``, ``"trafo"``, or
        ``"dc_link"``.)
    transformers.csv
        ``voltage_bus0`` → ``primary_voltage``,
        ``voltage_bus1`` → ``secondary_voltage``

    The operation is idempotent: if the target column names already exist the
    file is left unchanged.

    Parameters
    ----------
    data_files : dict[str, Path]
        Mapping returned by :func:`download_and_preprocess_zenodo_data`.

    Returns
    -------
    dict[str, Path]
        The same mapping (files modified in-place).
    """
    # buses: coordinate and voltage column names
    buses_path = data_files["buses.csv"]
    buses_df = pd.read_csv(buses_path)
    buses_df = buses_df.rename(columns={"x": "lon", "y": "lat", "v_nom": "voltage"})
    buses_df.to_csv(buses_path, index=False)

    # lines: rename 'type' (cable type string) to 'line_type' so it does not
    # collide with NPAP's internal edge-type attribute used by the visualizer.
    # Read with quotechar="'" for the raw Zenodo file whose geometry column is
    # single-quoted (e.g. 'LINESTRING (x y, x y)').  Dropping geometry avoids
    # the quoting mismatch on subsequent idempotent runs, consistent with the
    # transformer preprocessing.
    lines_path = data_files["lines.csv"]
    lines_df = pd.read_csv(lines_path, quotechar="'")
    lines_df = lines_df.rename(columns={"type": "line_type"})
    lines_df = lines_df.drop(columns=["geometry"], errors="ignore")
    lines_df.to_csv(lines_path, index=False)

    # transformers: voltage column names + dummy reactance
    # Read with quotechar="'" because the geometry field is single-quoted in the
    # Zenodo CSV (e.g. 'LINESTRING (x1 y1, x2 y2)'). Writing back with the default
    # double-quote quotechar would leave the geometry's internal comma unprotected,
    # shifting all subsequent columns when the va_loader re-reads the file.
    # Dropping geometry avoids the quoting mismatch entirely; NPAP never uses it.
    trafos_path = data_files["transformers.csv"]
    trafos_df = pd.read_csv(trafos_path, quotechar="'")
    trafos_df = trafos_df.rename(
        columns={"voltage_bus0": "primary_voltage", "voltage_bus1": "secondary_voltage"}
    )
    trafos_df = trafos_df.drop(columns=["geometry"], errors="ignore")
    if "x" not in trafos_df.columns:
        trafos_df["x"] = 20
    trafos_df.to_csv(trafos_path, index=False)

    return data_files


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_and_preprocess_zenodo_data(
    data_dir: Path | str | None = None,
) -> dict[str, Path]:
    """
    Download all PyPSA-Eur network CSV files from Zenodo and preprocess them.

    Files that already exist on disk are skipped.  A live progress display
    shows each file's size upfront and a progress bar during the transfer.
    After downloading, column names are renamed to match NPAP conventions
    (see :func:`_preprocess_zenodo_files`).

    Parameters
    ----------
    data_dir : Path or str or None, optional
        Directory where files are saved.  Defaults to ``<examples>/data/``.

    Returns
    -------
    dict[str, Path]
        Mapping of filename to the local ``Path`` of the downloaded file.

    Examples
    --------
    >>> from utils import download_and_preprocess_zenodo_data
    >>> data_files = download_and_preprocess_zenodo_data()
    >>> data_files["buses.csv"]
    PosixPath('.../examples/data/buses.csv')
    """
    dest_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    file_sizes = _fetch_zenodo_sizes()

    statuses: dict[str, str] = {}
    for filename in FILES:
        dest = dest_dir / filename
        if dest.exists():
            statuses[filename] = f"already cached ({dest.stat().st_size / 1024**2:.1f} MB)"
        else:
            size_hint = f"  [{_fmt_size(file_sizes[filename])}]" if filename in file_sizes else ""
            statuses[filename] = f"pending{size_hint}"

    def _render() -> None:
        try:
            from IPython.display import clear_output

            clear_output(wait=True)
        except ImportError:
            pass
        print("Fetching PyPSA-Eur network data from Zenodo:")
        for fn, status in statuses.items():
            print(f"  {fn}: {status}")

    _render()

    for filename in FILES:
        dest = dest_dir / filename
        if dest.exists():
            continue

        url = f"{ZENODO_BASE}/{filename}/content"
        total = file_sizes.get(filename, 0)

        def _make_hook(fn: str, total_bytes: int):
            last_t: list[float] = [0.0]

            def _hook(block_count: int, block_size: int, _reported: int) -> None:
                now = time.monotonic()
                if now - last_t[0] < 0.1:
                    return
                last_t[0] = now
                downloaded = (
                    min(block_count * block_size, total_bytes)
                    if total_bytes
                    else block_count * block_size
                )
                if total_bytes > 0:
                    pct = downloaded / total_bytes * 100
                    filled = int(pct / 100 * _BAR_WIDTH)
                    bar = "=" * filled + "-" * (_BAR_WIDTH - filled)
                    statuses[fn] = (
                        f"[{bar}] {pct:5.1f}%  ({_fmt_size(downloaded)} / {_fmt_size(total_bytes)})"
                    )
                else:
                    statuses[fn] = f"{_fmt_size(downloaded)} downloaded"
                _render()

            return _hook

        urllib.request.urlretrieve(url, dest, reporthook=_make_hook(filename, total))
        statuses[filename] = f"done ({dest.stat().st_size / 1024**2:.1f} MB)"
        _render()

    data_files = {f: dest_dir / f for f in FILES}
    _preprocess_zenodo_files(data_files)
    return data_files
