# SPDX-FileCopyrightText: Contributors to NPAP
# SPDX-License-Identifier: MIT

"""Tests for npap.datasets. The network layer is stubbed out throughout."""

from pathlib import Path

import pandas as pd
import pytest

from npap.datasets import (
    DATA_HOME_ENV,
    PYPSA_EUR_FILES,
    _download_file,
    _fetch_remote_sizes,
    _format_size,
    _harmonise_columns,
    clear_data_home,
    fetch_pypsa_eur,
    get_data_home,
)

RAW_BUSES = "bus_id,x,y,v_nom,country\nb1,15.4,47.0,380,AT\nb2,16.3,48.2,220,AT\n"
RAW_LINES = (
    "line_id,bus0,bus1,x,r,s_nom,type,geometry\n"
    "l1,b1,b2,0.01,0.001,500,Al/St 240/40,'LINESTRING (15.4 47.0, 16.3 48.2)'\n"
)
RAW_TRAFOS = (
    "trafo_id,bus0,bus1,voltage_bus0,voltage_bus1,geometry\nt1,b1,b2,380,220,'POINT (15.4 47.0)'\n"
)


def _write_raw(directory):
    """Write raw Zenodo-shaped CSVs into `directory` and return the mapping."""
    files = {}
    for name, content in [
        ("buses.csv", RAW_BUSES),
        ("lines.csv", RAW_LINES),
        ("transformers.csv", RAW_TRAFOS),
        ("converters.csv", "converter_id,bus\nc1,b1\n"),
        ("links.csv", "link_id,bus0,bus1\nk1,b1,b2\n"),
    ]:
        path = directory / name
        path.write_text(content, encoding="utf-8")
        files[name] = path
    return files


class TestDataHome:
    """Cache directory resolution."""

    def test_explicit_dir_wins(self, tmp_path):
        target = tmp_path / "explicit"
        assert get_data_home(target) == target
        assert target.is_dir()

    def test_environment_variable(self, tmp_path, monkeypatch):
        monkeypatch.setenv(DATA_HOME_ENV, str(tmp_path / "from_env"))
        assert get_data_home() == tmp_path / "from_env"

    def test_explicit_overrides_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv(DATA_HOME_ENV, str(tmp_path / "from_env"))
        assert get_data_home(tmp_path / "explicit") == tmp_path / "explicit"

    def test_falls_back_to_user_cache(self, tmp_path, monkeypatch):
        monkeypatch.delenv(DATA_HOME_ENV, raising=False)
        monkeypatch.setattr("os.name", "posix")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert get_data_home() == tmp_path / "npap"

    def test_never_writes_inside_the_package(self, tmp_path, monkeypatch):
        """Regression: the cache must not land in site-packages."""
        import npap

        monkeypatch.delenv(DATA_HOME_ENV, raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        package_dir = Path(npap.__file__).parent
        assert package_dir not in get_data_home().parents

    def test_clear_removes_cache(self, tmp_path):
        home = get_data_home(tmp_path / "cache")
        (home / "buses.csv").write_text("x", encoding="utf-8")
        clear_data_home(home)
        assert not home.exists()


class TestFormatting:
    """Human-readable byte formatting."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(1024, "1 KB"), (900 * 1024, "900 KB"), (5 * 1024**2, "5.0 MB")],
    )
    def test_format_size(self, value, expected):
        assert _format_size(value) == expected


class TestRemoteSizes:
    """Parsing of the Zenodo file listing."""

    def _stub_urlopen(self, monkeypatch, payload):
        import contextlib
        import json as json_mod

        @contextlib.contextmanager
        def fake_urlopen(url):
            class Resp:
                def read(self):
                    return json_mod.dumps(payload).encode()

            yield Resp()

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    def test_parses_v3_entries_shape(self, monkeypatch):
        self._stub_urlopen(monkeypatch, {"entries": [{"key": "buses.csv", "size": 2048}]})
        assert _fetch_remote_sizes() == {"buses.csv": 2048}

    def test_parses_v1_list_shape(self, monkeypatch):
        self._stub_urlopen(monkeypatch, [{"key": "lines.csv", "size": 4096}])
        assert _fetch_remote_sizes() == {"lines.csv": 4096}

    def test_returns_empty_on_failure(self, monkeypatch):
        def boom(url):
            raise OSError("network down")

        monkeypatch.setattr("urllib.request.urlopen", boom)
        assert _fetch_remote_sizes() == {}


class TestDownloadFile:
    """The download layer, with urlretrieve stubbed."""

    def test_writes_atomically_via_part_file(self, tmp_path, monkeypatch):
        seen = {}

        def fake_urlretrieve(url, filename, reporthook=None):
            seen["tmp"] = Path(filename)
            Path(filename).write_text("payload", encoding="utf-8")

        monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)
        dest = tmp_path / "buses.csv"
        _download_file("https://example.invalid/f", dest, show_progress=False)

        assert seen["tmp"].suffix == ".part"
        assert not seen["tmp"].exists()
        assert dest.read_text(encoding="utf-8") == "payload"

    def test_progress_reports_to_stdout(self, tmp_path, monkeypatch, capsys):
        def fake_urlretrieve(url, filename, reporthook=None):
            Path(filename).write_text("x" * 2048, encoding="utf-8")
            if reporthook:
                reporthook(1, 1024, 2048)

        monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)
        _download_file(
            "https://example.invalid/f",
            tmp_path / "lines.csv",
            total_bytes=2048,
            show_progress=True,
        )
        assert "lines.csv" in capsys.readouterr().out


class TestHarmoniseColumns:
    """Column renaming applied after download."""

    def test_renames_bus_columns(self, tmp_path):
        files = _write_raw(tmp_path)
        _harmonise_columns(files)
        buses = pd.read_csv(files["buses.csv"])
        assert {"lon", "lat", "voltage"} <= set(buses.columns)
        assert not {"x", "y", "v_nom"} & set(buses.columns)

    def test_renames_line_type_and_drops_geometry(self, tmp_path):
        files = _write_raw(tmp_path)
        _harmonise_columns(files)
        lines = pd.read_csv(files["lines.csv"])
        assert "line_type" in lines.columns
        assert "geometry" not in lines.columns
        # 'x' here is reactance and must survive the bus-column rename
        assert lines.loc[0, "x"] == 0.01

    def test_renames_transformer_voltages(self, tmp_path):
        files = _write_raw(tmp_path)
        _harmonise_columns(files)
        trafos = pd.read_csv(files["transformers.csv"])
        assert {"primary_voltage", "secondary_voltage"} <= set(trafos.columns)
        assert "geometry" not in trafos.columns

    def test_adds_placeholder_reactance(self, tmp_path):
        files = _write_raw(tmp_path)
        _harmonise_columns(files)
        assert pd.read_csv(files["transformers.csv"]).loc[0, "x"] == 20

    def test_is_idempotent(self, tmp_path):
        files = _write_raw(tmp_path)
        _harmonise_columns(files)
        first = {n: p.read_text(encoding="utf-8") for n, p in files.items()}
        _harmonise_columns(files)
        assert {n: p.read_text(encoding="utf-8") for n, p in files.items()} == first


class TestFetch:
    """fetch_pypsa_eur with the download layer stubbed out."""

    def test_downloads_then_harmonises(self, tmp_path, monkeypatch):
        calls = []

        def fake_download(url, dest, total_bytes=0, show_progress=True):
            calls.append(dest.name)
            _write_raw(dest.parent)

        monkeypatch.setattr("npap.datasets._download_file", fake_download)
        monkeypatch.setattr("npap.datasets._fetch_remote_sizes", dict)

        files = fetch_pypsa_eur(data_dir=tmp_path, show_progress=False)

        assert set(files) == set(PYPSA_EUR_FILES)
        assert all(p.exists() for p in files.values())
        assert calls  # the download layer was exercised
        assert "lon" in pd.read_csv(files["buses.csv"]).columns

    def test_skips_download_when_cached(self, tmp_path, monkeypatch):
        _write_raw(tmp_path)

        def fail(*args, **kwargs):
            raise AssertionError("should not download when files are cached")

        monkeypatch.setattr("npap.datasets._download_file", fail)
        files = fetch_pypsa_eur(data_dir=tmp_path, show_progress=False)
        assert set(files) == set(PYPSA_EUR_FILES)

    def test_returns_absolute_paths(self, tmp_path):
        _write_raw(tmp_path)
        files = fetch_pypsa_eur(data_dir=tmp_path, show_progress=False)
        assert all(p.is_absolute() for p in files.values())

    def test_quiet_mode_prints_nothing(self, tmp_path, capsys):
        _write_raw(tmp_path)
        fetch_pypsa_eur(data_dir=tmp_path, show_progress=False)
        assert capsys.readouterr().out == ""


class TestCacheReporting:
    """The user is told where the data lives and that it persists."""

    def test_cache_hit_reports_location_and_how_to_clear(self, tmp_path, capsys):
        _write_raw(tmp_path)
        fetch_pypsa_eur(data_dir=tmp_path, show_progress=True)
        out = capsys.readouterr().out
        assert str(tmp_path) in out
        assert "never removed" in out
        assert "clear_data_home" in out

    def test_after_download_reports_location_and_how_to_clear(self, tmp_path, monkeypatch, capsys):
        def fake_download(url, dest, total_bytes=0, show_progress=True):
            _write_raw(dest.parent)

        monkeypatch.setattr("npap.datasets._download_file", fake_download)
        monkeypatch.setattr("npap.datasets._fetch_remote_sizes", dict)

        fetch_pypsa_eur(data_dir=tmp_path, show_progress=True)
        out = capsys.readouterr().out
        assert str(tmp_path) in out
        assert "clear_data_home" in out

    def test_summary_reports_total_size(self, tmp_path, capsys):
        _write_raw(tmp_path)
        fetch_pypsa_eur(data_dir=tmp_path, show_progress=True)
        # Sizes are tiny in the fixtures, so the summary uses the KB branch
        assert "KB)" in capsys.readouterr().out
