# Getting Started

This guide walks you through setting up your development environment for contributing to NPAP.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** installed
- **Git** installed and configured
- A **GitHub account**

## Step 1: Fork the Repository

1. Navigate to the [NPAP repository](https://github.com/IEE-TUGraz/NPAP) on GitHub
2. Click the **Fork** button in the top-right corner
3. This creates a copy of the repository under your GitHub account

## Step 2: Clone Your Fork

```bash
# Clone your forked repository
git clone https://github.com/YOUR-USERNAME/NPAP.git
cd NPAP

# Add the upstream remote (original repository)
git remote add upstream https://github.com/IEE-TUGraz/NPAP.git

# Verify remotes
git remote -v
# origin    https://github.com/YOUR-USERNAME/NPAP.git (fetch)
# origin    https://github.com/YOUR-USERNAME/NPAP.git (push)
# upstream  https://github.com/IEE-TUGraz/NPAP.git (fetch)
# upstream  https://github.com/IEE-TUGraz/NPAP.git (push)
```

## Step 3: Create a Virtual Environment

We recommend using a virtual environment to isolate dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

## Step 4: Install Development Dependencies

Install NPAP in development mode with all optional dependencies:

```bash
pip install -e ".[dev,test,docs]"
```

This installs:
- **dev**: Ruff (linting/formatting), pre-commit
- **test**: pytest, pytest-cov
- **docs**: Sphinx, pydata-sphinx-theme, myst-parser

## Step 5: Install Pre-commit Hooks

Pre-commit hooks automatically check your code before each commit:

```bash
pre-commit install
```

The hooks will:
- Format code with Ruff
- Check for linting errors
- Validate YAML/JSON files

## Step 6: Verify Installation

Test that everything is set up correctly:

```bash
# Run tests
pytest

# Check linting
ruff check .

# Import NPAP
python -c "import npap; print(npap.__version__)"
```

## Step 7: Create a Branch

Create a new branch for your contribution:

```bash
# Sync with upstream first
git fetch upstream
git checkout main
git merge upstream/main

# Create your feature branch
git checkout -b feature/your-feature-name
```

### Branch Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/description` | `feature/add-lmp-partitioning` |
| Bug fix | `fix/description` | `fix/kmeans-convergence` |
| Documentation | `docs/description` | `docs/improve-quick-start` |
| Refactoring | `refactor/description` | `refactor/manager-structure` |

## Keeping Your Fork Updated

Periodically sync your fork with the upstream repository:

```bash
# Fetch upstream changes
git fetch upstream

# Merge into your main branch
git checkout main
git merge upstream/main

# Push to your fork
git push origin main
```

## Next Steps

Now that your environment is set up:

- Read the [Development Workflow](development-workflow.md) guide for code style and testing
- Check [Extending NPAP](extending-npap.md) if you're adding new strategies
- Review [Pull Requests](pull-requests.md) before submitting your changes
