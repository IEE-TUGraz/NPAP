# Installation

## From PyPI

The simplest way to install NPAP is via pip:

```bash
pip install npap
```

This installs NPAP with its core dependencies: NetworkX, NumPy, Pandas, SciPy, scikit-learn, and Plotly.

## From Source

To install the latest development version:

```bash
git clone https://github.com/IEE-TUGraz/NPAP.git
cd NPAP
pip install -e .
```

## Development Installation

For contributors and developers who want to run tests, build documentation, or use linting tools:

```bash
# Clone the repository
git clone https://github.com/IEE-TUGraz/NPAP.git
cd NPAP

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with all development dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

## Optional Dependencies

NPAP has optional dependency groups for different use cases:

| Group | Command | Includes |
|-------|---------|----------|
| `test` | `pip install -e ".[test]"` | pytest, pytest-cov |
| `dev` | `pip install -e ".[dev]"` | ruff, pre-commit |
| `docs` | `pip install -e ".[docs]"` | sphinx, pydata-sphinx-theme |
| All | `pip install -e ".[dev,test,docs]"` | Everything |

## Verifying Installation

After installation, verify NPAP is working correctly:

```python
import npap

# Check version
print(npap.__version__)

# Quick test
manager = npap.PartitionAggregatorManager()
print("NPAP installed successfully!")
```

## Requirements

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, or Linux

### Core Dependencies

| Package | Minimum Version | Purpose |
|---------|-----------------|---------|
| networkx | 3.0 | Graph data structures |
| numpy | 1.24 | Numerical operations |
| pandas | 2.0 | Data manipulation |
| scipy | 1.10 | Scientific computing |
| scikit-learn | 1.3 | Clustering algorithms |
| plotly | 5.0 | Interactive visualization |

## Troubleshooting

### Import Error

If you encounter an import error:

```python
>>> import npap
ModuleNotFoundError: No module named 'npap'
```

Ensure NPAP is installed in your active Python environment:

```bash
pip show npap
```

### Version Conflicts

If you experience dependency conflicts, create a fresh virtual environment:

```bash
python -m venv npap_env
source npap_env/bin/activate  # On Windows: npap_env\Scripts\activate
pip install npap
```

## Next Steps

Once installed, proceed to the [Quick Start](quick-start.md) guide to learn the NPAP workflow.
