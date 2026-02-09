# Reporting Issues

Help us improve NPAP by reporting bugs and suggesting features.

## Reporting Bugs

### Before Reporting

1. **Check existing issues**: Search [GitHub Issues](https://github.com/IEE-TUGraz/NPAP/issues) to see if it's already reported
2. **Try the latest version**: Update to the latest development NPAP version
3. **Minimal reproduction**: Create a minimal example that reproduces the issue

### Information to Include

| Information | How to Get It |
|-------------|---------------|
| Python version | `python --version` |
| NPAP version | `python -c "import npap; print(npap.__version__)"` |
| Operating system | Windows/macOS/Linux |
| Minimal example | Code that reproduces the issue |
| Full traceback | Complete error message |

### Bug Report Template

```markdown
## Description

Brief description of the bug.

## Steps to Reproduce

1. Step one
2. Step two
3. ...

## Minimal Example

.```python
import npap

# Minimal code to reproduce
manager = npap.PartitionAggregatorManager()
# ...
.```

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened.

## Error Message

.```
Paste full error traceback here
.```

## Environment

- Python version: 3.11.0
- NPAP version: 0.1.0
- OS: Ubuntu 22.04
```

### Where to Report

Open an issue at: [https://github.com/IEE-TUGraz/NPAP/issues/new](https://github.com/IEE-TUGraz/NPAP/issues/new)

## Suggesting Features

### Before Suggesting

1. **Check the roadmap**: See if it's already planned
2. **Search existing issues**: Someone may have suggested it already
3. **Consider scope**: Does it fit NPAP's focus on network partitioning and aggregation?

### Feature Request Template

```markdown
## Feature Description

Brief description of the feature.

## Problem/Use Case

What problem does this solve? What's your use case?

## Proposed Solution

How do you envision this working?

## Alternatives Considered

Other approaches you've thought about.

## Additional Context

Any other context, examples, or references.
```

## Security Issues

For security vulnerabilities, **do not** open a public issue. Instead:

1. Email the maintainers directly
2. Include details about the vulnerability
3. Allow time for a fix before public disclosure

## Labels

Maintainers use labels to categorize issues:

| Label | Description |
|-------|-------------|
| `bug` | Something isn't working |
| `enhancement` | New feature request |
| `documentation` | Documentation improvements |
| `good first issue` | Good for newcomers |
| `help wanted` | Extra attention needed |
| `question` | General questions |

## What Happens Next

After you submit an issue:

1. **Triage**: A maintainer will review and label your issue
2. **Discussion**: We may ask clarifying questions
3. **Resolution**: The issue will be assigned or marked for community contribution
4. **Updates**: You'll be notified of progress

## Contributing a Fix

Found a bug you can fix? We encourage you to:

1. Comment on the issue that you'd like to work on it
2. Fork the repository and create a branch
3. Submit a pull request referencing the issue

See [Pull Requests](pull-requests.md) for details.
