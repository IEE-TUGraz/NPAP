# Contributing to NPAP 🎉

First off, **thank you** for considering contributing to NPAP! It's people like you who make open-source such a wonderful place to learn, create, and collaborate.

> [!TIP]
> For the full contributing documentation — including how to extend NPAP with custom strategies, detailed code style guidelines, and citing instructions — visit
> **[npap.readthedocs.io/en/latest/contributing.html](https://npap.readthedocs.io/en/latest/contributing.html)**

---

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
  - [Fork and Clone](#1-fork-and-clone)
  - [Set Up Environment](#2-set-up-development-environment)
  - [Create a Branch](#3-create-a-branch)
- [Development Workflow](#development-workflow)
  - [Code Style](#code-style)
  - [Running Tests](#running-tests)
  - [Building Documentation](#building-documentation)
- [Submitting Changes](#submitting-changes)
  - [Pull Request Process](#pull-request-process)
  - [Commit Messages](#commit-messages)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)
- [Code of Conduct](#code-of-conduct)

---

## Ways to Contribute

We value **every type of contribution**, not just code!


- **Bug Reports** - Found something that doesn't work? Let us know!
- **Feature Ideas** - Have an idea to improve NPAP? We'd love to hear it
- **Documentation** - Spotted a typo or unclear explanation? Help us fix it
- **Code** - Bug fixes, new features, or performance improvements
- **Spread the Word** - Tell others about NPAP if you find it useful

---

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/IEE-TUGraz/NPAP.git
cd NPAP
```

### 2. Set Up Development Environment

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

---

## Development Workflow

### Code Style

We use **Ruff** for linting and formatting. Pre-commit hooks run automatically, but you can also check manually:

| Command | Description |
|:--------|:------------|
| `ruff check .` | Check for linting issues |
| `ruff check --fix .` | Auto-fix linting issues |
| `ruff format .` | Format code |

### Running Tests

| Command | Description |
|:--------|:------------|
| `pytest` | Run all tests |
| `pytest --cov=npap` | Run with coverage report |
| `pytest test/test_partitioning.py` | Run specific test file |
| `pytest -k "test_kmeans"` | Run tests matching pattern |

### Building Documentation

```bash
cd docs
sphinx-build -b html . _build/html
# Open _build/html/index.html in your browser
```

---

## Submitting Changes

### Pull Request Process

1. **Update documentation** if you changed any public APIs
2. **Add tests** for new functionality
3. **Ensure all tests pass** and pre-commit hooks are happy
4. **Write a clear PR description** explaining your changes

### Commit Messages

We follow a simple, descriptive commit style:

```
Short summary of change (50 chars or less)

More detailed explanation if needed. Explain the "why"
rather than the "what" — the code shows what changed.
```

---

## Reporting Bugs

Bugs, questions and feature requests all go through GitHub Issues:
**[open a new issue](https://github.com/IEE-TUGraz/NPAP/issues/new/choose)** and
pick the matching template. Please
[search the existing issues](https://github.com/IEE-TUGraz/NPAP/issues) first, in
case it has already been reported.

When reporting bugs, please include:

| Information | How to Get It |
|:------------|:--------------|
| Python version | `python --version` |
| NPAP version | `python -c "import npap; print(npap.__version__)"` |
| Minimal example | Code that reproduces the issue |
| Error traceback | Full error message |

---

## Suggesting Features

Feature requests are welcome — open one with the
[feature request template](https://github.com/IEE-TUGraz/NPAP/issues/new/choose).
When suggesting a feature:

- Explain the **problem** you're trying to solve
- Describe your **proposed solution**
- Consider if it fits NPAP's scope (network partitioning & aggregation)

---

## Code of Conduct

Be **kind and respectful**. We're all here to learn and build something useful together. Harassment or exclusionary behavior is not tolerated.

---

<p align="center">
<b>Questions?</b> Don't hesitate to open an issue. There are no silly questions!
<br><br>
<i>Thank you for contributing — every contribution helps make NPAP better for everyone.</i>
<br><br>
📖 <a href="https://npap.readthedocs.io/en/latest/contributing.html">Full contributing docs on ReadTheDocs</a>
</p>
