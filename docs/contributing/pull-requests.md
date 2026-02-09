# Pull Requests

This guide covers how to submit pull requests to NPAP and the review process.

## Before You Submit

### Checklist

Before opening a pull request, ensure:

- [ ] All tests pass: `pytest`
- [ ] Code is formatted: `ruff format .`
- [ ] No linting errors: `ruff check .`
- [ ] Documentation updated (if applicable)
- [ ] Commit messages are clear and descriptive

### Sync with Upstream

Make sure your branch is up to date:

```bash
git fetch upstream
git rebase upstream/main
```

## Creating a Pull Request

### Step 1: Push Your Branch

```bash
git push origin feature/your-feature-name
```

### Step 2: Open the PR on GitHub

1. Go to your fork on GitHub
2. Click **"Compare & pull request"**
3. Select `IEE-TUGraz/NPAP` as the base repository and `main` as the base branch

### Step 3: Write a Good PR Description

Use this template:

```markdown
## Summary

Brief description of what this PR does.

## Changes

- Added X feature
- Fixed Y bug
- Updated Z documentation

## Testing

Describe how you tested your changes.

## Related Issues

Closes #123 (if applicable)
```

## Commit Message Guidelines

### Format

```
Short summary (50 chars or less)

More detailed explanation if needed. Explain the "why"
rather than the "what" — the code shows what changed.
```

### Examples

**Good:**
```
Add LMP-based partitioning strategy

Implements a new partitioning strategy that clusters nodes
based on Locational Marginal Prices. This is useful for
market-based network aggregation.
```

**Bad:**
```
update code
```

### Conventions

| Type | Prefix | Example                                          |
|------|--------|--------------------------------------------------|
| Feature | `Add` | `Add AC-island detection`                        |
| Bug fix | `Fix` | `Fix kmeans convergence issue`                   |
| Documentation | `Docs` | `docs: update quick start guide`                 |
| Refactoring | `Refactor` | `Refactor manager structure`                     |
| Testing | `Test` | `test: add coverage for electrical partitioning` |

## The Review Process

### What to Expect

1. **Automated checks**: CI runs tests and linting
2. **Maintainer review**: A maintainer will review your code
3. **Feedback**: You may receive suggestions for changes
4. **Iteration**: Address feedback and push updates
5. **Merge**: Once approved, your PR will be merged

### Responding to Feedback

- Be open to suggestions
- Ask for clarification if needed
- Push additional commits to address feedback
- Mark conversations as resolved when addressed

### Updating Your PR

If changes are requested:

```bash
# Make changes
git add .
git commit -m "Address review feedback"
git push origin feature/your-feature-name
```

The PR will automatically update.

## PR Types

### Feature PRs

For new features:

- Include documentation
- Add comprehensive tests
- Consider backward compatibility

### Bug Fix PRs

For bug fixes:

- Include a test that reproduces the bug
- Reference the issue number if one exists
- Keep changes focused on the fix

### Documentation PRs

For documentation improvements:

- Build docs locally to verify
- Check for broken links
- Follow existing style

## After Merge

Once your PR is merged:

1. Delete your feature branch (GitHub offers this option)
2. Sync your fork:
   ```bash
   git checkout main
   git pull upstream main
   git push origin main
   ```
3. Celebrate your contribution! 🎉

## Getting Help

If you're stuck or unsure about something:

- Open a draft PR and ask for guidance
- Open an issue to discuss your approach
- Tag maintainers for help

We're here to help you succeed!
