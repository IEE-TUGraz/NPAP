from npap.visualization import (
    PlotConfig,
    PlotPreset,
    _apply_preset_overrides,
    _normalize_plot_preset,
)


def test_transmission_study_preset_overrides_high_voltage():
    config = PlotConfig()
    updated = _apply_preset_overrides(config, PlotPreset.TRANSMISSION_STUDY)

    assert updated.title == "Transmission Study"
    assert updated.line_voltage_threshold == 450.0
    assert updated.width == 1200
    assert updated.height == 750


def test_distribution_study_preset_focuses_on_dense_grid():
    config = PlotConfig()
    updated = _apply_preset_overrides(config, PlotPreset.DISTRIBUTION_STUDY)

    assert updated.title == "Distribution Study"
    assert updated.map_zoom == 8.8
    assert updated.cluster_colorscale == "YlOrBr"


def test_e_mobility_preset_highlights_nodes():
    config = PlotConfig()
    updated = _apply_preset_overrides(config, "e_mobility_planning")

    assert updated.title == "E-Mobility Planning"
    assert updated.node_color == "#FF6F61"
    assert updated.node_size == 10
    assert updated.map_zoom == 10.5


def test_preset_normalization_accepts_human_friendly_names():
    assert _normalize_plot_preset("Transmission Study") == PlotPreset.TRANSMISSION_STUDY
    assert _normalize_plot_preset("distribution study") == PlotPreset.DISTRIBUTION_STUDY
    assert _normalize_plot_preset("E-Mobility Planning") == PlotPreset.E_MOBILITY_PLANNING
