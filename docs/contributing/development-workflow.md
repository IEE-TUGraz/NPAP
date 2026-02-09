# Development Workflow

This guide covers the day-to-day development practices for contributing to NPAP.

## Code Style

NPAP uses [**Ruff**](https://docs.astral.sh/ruff/) for both linting and formatting. This ensures consistent code style across the project.

### Configuration

Ruff is configured in `pyproject.toml`:
- **Line length**: 100 characters
- **Docstring style**: NumPy convention
- **Type hints**: Expected throughout

### Commands

| Command | Description |
|---------|-------------|
| `ruff check .` | Check for linting issues |
| `ruff check --fix .` | Auto-fix linting issues |
| `ruff format .` | Format code |
| `ruff format --check .` | Check formatting without changes |

### Pre-commit Integration

Pre-commit hooks run Ruff automatically before each commit:

```bash
# Install hooks (one-time)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

## Testing

NPAP uses [**pytest**](https://docs.pytest.org/en/stable/) for testing.

### Running Tests

| Command | Description |
|---------|-------------|
| `pytest` | Run all tests |
| `pytest --cov=npap` | Run with coverage report |
| `pytest test/test_partitioning.py` | Run specific test file |
| `pytest -k "test_kmeans"` | Run tests matching pattern |
| `pytest -v` | Verbose output |
| `pytest -x` | Stop on first failure |

### Writing Tests

Place tests in the `test/` directory following the naming convention `test_*.py`:

```python
# test/test_my_feature.py
import pytest
import networkx as nx
from npap import PartitionAggregatorManager

class TestMyFeature:
    def test_basic_functionality(self):
        """Test the basic case."""
        manager = PartitionAggregatorManager()
        # ... test code ...
        assert result == expected

    def test_edge_case(self):
        """Test edge case handling."""
        # ...

    def test_error_handling(self):
        """Test that appropriate errors are raised."""
        with pytest.raises(ValueError):
            # Code that should raise ValueError
            pass
```

### Test Coverage

Aim for high test coverage on new code:

```bash
# Generate coverage report
pytest --cov=npap --cov-report=html

# Open the HTML report
open htmlcov/index.html  # or just browse to the file
```

## Building Documentation

Documentation is built with [**Sphinx**](https://www.sphinx-doc.org/en/master/) using the [**pydata-sphinx-theme**](https://pydata-sphinx-theme.readthedocs.io/en/stable/).

### Local Build

```bash
cd docs
sphinx-build -b html . _build/html
```

Open `_build/html/index.html` in your browser to view.

### Live Reload (Development)

For live reload during documentation development:

```bash
pip install sphinx-autobuild
cd docs
sphinx-autobuild . _build/html
```

This starts a local server at `http://127.0.0.1:8000` that auto-refreshes on changes.

### Documentation Style

- **Format**: Markdown (MyST) for user guides, RST for API docs
- **Docstrings**: [NumPy convention](https://numpydoc.readthedocs.io/en/latest/format.html)
- **Code examples**: Include runnable examples where possible

Example docstring:

```python
def partition(self, strategy: str, n_clusters: int, **kwargs) -> PartitionResult:
    """Partition the loaded graph into clusters.

    Parameters
    ----------
    strategy : str
        Name of the partitioning strategy to use.
    n_clusters : int
        Number of clusters to create.
    **kwargs
        Additional arguments passed to the strategy.

    Returns
    -------
    PartitionResult
        Object containing cluster assignments and metadata.

    Raises
    ------
    PartitioningError
        If the partitioning algorithm fails.
    ValidationError
        If required node attributes are missing.

    Examples
    --------
    >>> manager = PartitionAggregatorManager()
    >>> manager.load_data("networkx_direct", graph=G)
    >>> partition = manager.partition("geographical_kmeans", n_clusters=10)
    >>> print(partition.n_clusters)
    10
    """
```

## Type Hints

Use [type hints](https://docs.python.org/3/library/typing.html) throughout your code:

```python
from typing import Any
import networkx as nx

def my_function(
    graph: nx.DiGraph,
    n_clusters: int = 10,
    **kwargs: Any
) -> dict[int, list[str]]:
    """Function with type hints."""
    ...
```

## Project Structure

Understanding the project structure helps when making changes:

```
npap/
├── __init__.py          # Public API exports
├── interfaces.py        # Abstract base classes and dataclasses
├── managers.py          # Manager classes
├── exceptions.py        # Custom exceptions
├── visualization.py     # Plotting functionality
├── input/               # Data loading strategies
│   ├── csv_loader.py
│   ├── networkx_loader.py
│   └── va_loader.py
├── partitioning/        # Partitioning strategies
│   ├── geographical.py
│   ├── electrical.py
│   └── va_geographical.py
└── aggregation/         # Aggregation strategies
    ├── topology.py
    ├── physical.py
    └── properties.py
```

## Common Tasks

### Adding a New Partitioning Strategy

1. Create strategy class inheriting from `PartitioningStrategy`
2. Implement `required_attributes` property and `partition` method
3. Add tests in `test/test_partitioning.py`
4. Register in `PartitioningManager` initialization
5. Update documentation

See [Extending NPAP](extending-npap.md) for detailed instructions.

### Fixing a Bug

1. Write a failing test that reproduces the bug
2. Fix the bug
3. Verify the test passes
4. Check no other tests broke

### Updating Documentation

1. Edit the relevant `.md` or `.rst` file in `docs/`
2. Build locally to verify: `sphinx-build -b html docs docs/_build/html`
3. Check for any Sphinx warnings

## Next Steps

- [Extending NPAP](extending-npap.md) - Creating custom strategies
- [Pull Requests](pull-requests.md) - Submitting your changes
