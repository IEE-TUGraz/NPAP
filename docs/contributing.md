# Contributing to NPAP

```{toctree}
:hidden:
:maxdepth: 2

contributing/getting-started
contributing/development-workflow
contributing/extending-npap
contributing/pull-requests
contributing/reporting-issues
contributing/code-of-conduct
contributing/citing
```

First off, **thank you** for considering contributing to NPAP! It's people like you who make open-source such a wonderful place to learn, create, and collaborate.

---

## Ways to Contribute

We value **every type of contribution**, not just code!

::::{grid} 2
:gutter: 3

:::{grid-item-card} Bug Reports
:class-card: contrib-card

Found something that doesn't work? Let us know! Open an issue with details about the problem.
:::

:::{grid-item-card} Feature Ideas
:class-card: contrib-card

Have an idea to improve NPAP? We'd love to hear it. Describe your use case and proposed solution.
:::

:::{grid-item-card} Documentation
:class-card: contrib-card

Spotted a typo or unclear explanation? Help us fix it. Good docs make everyone's life easier.
:::

:::{grid-item-card} Code
:class-card: contrib-card

Bug fixes, new features, or performance improvements. All code contributions are welcome!
:::

::::

---

## Contribution Workflow Overview

```{mermaid}
flowchart LR
    A[Fork Repository] --> B[Clone & Setup]
    B --> C[Create Branch]
    C --> D[Make Changes]
    D --> E[Test & Lint]
    E --> F[Commit]
    F --> G[Push & PR]
    G --> H[Review & Merge]

    style A fill:#2993B5,stroke:#1d6f8a,color:#fff
    style B fill:#2993B5,stroke:#1d6f8a,color:#fff
    style C fill:#2993B5,stroke:#1d6f8a,color:#fff
    style D fill:#2993B5,stroke:#1d6f8a,color:#fff
    style E fill:#0fad6b,stroke:#076b3f,color:#fff
    style F fill:#2993B5,stroke:#1d6f8a,color:#fff
    style G fill:#2993B5,stroke:#1d6f8a,color:#fff
    style H fill:#0fad6b,stroke:#076b3f,color:#fff
```

## Quick Links

| Topic | Description |
|-------|-------------|
| [Getting Started](contributing/getting-started.md) | Fork, clone, and set up your development environment |
| [Development Workflow](contributing/development-workflow.md) | Code style, testing, and building documentation |
| [Extending NPAP](contributing/extending-npap.md) | Create custom strategies for data loading, partitioning, and aggregation |
| [Pull Requests](contributing/pull-requests.md) | PR guidelines and commit message conventions |
| [Reporting Issues](contributing/reporting-issues.md) | How to report bugs and suggest features |
| [Code of Conduct](contributing/code-of-conduct.md) | Community guidelines |
| [Citing NPAP](contributing/citing.md) | How to cite NPAP in academic work |

---

## Quick Start for Contributors

### 1. Fork and Clone

```bash
# Fork on GitHub first, then:
git clone https://github.com/YOUR-USERNAME/NPAP.git
cd NPAP
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with all dependencies
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

### 4. Make Your Changes

- Follow the [code style guidelines](contributing/development-workflow.md#code-style)
- Add tests for new functionality
- Update documentation if needed

### 5. Test and Lint

```bash
# Run tests
pytest

# Check linting
ruff check .

# Format code
ruff format .
```

### 6. Commit and Push

```bash
git add .
git commit -m "Short description of changes"
git push origin feature/your-feature-name
```

### 7. Open a Pull Request

Go to GitHub and open a pull request from your branch to `main`.

---

```{admonition} Questions?
:class: tip

Don't hesitate to open an issue. There are no silly questions!

**Thank you for contributing** — every contribution helps make NPAP better for everyone.
```
