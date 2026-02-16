# Transition Guide: From Placeholder to Full Package

This document explains how to transition from the placeholder `flowbook-python` package to publishing the full FlowBook package.

## Current Setup (Placeholder Branch)

The `placeholder` branch contains:
- `placeholder-package/` directory with minimal package structure
- `.github/workflows/release.yaml` workflow for PyPI publishing
- Package name: `flowbook-python`
- Version: `0.0.1`

## Transition Steps

When ready to publish the full FlowBook package to PyPI:

### Option 1: Update the Placeholder Branch (Recommended for seamless transition)

1. **Merge main into placeholder branch:**
   ```bash
   git checkout placeholder
   git merge main
   ```

2. **Update the workflow to build from root:**
   - Edit `.github/workflows/release.yaml`
   - Change `cd placeholder-package` to build from repository root
   - Update the `packages-dir` from `placeholder-package/dist/` to `dist/`

3. **Update pyproject.toml:**
   - Change package name from `flowbook-python` to `flowbook` (or keep as `flowbook-python`)
   - Update version to the actual version (e.g., `0.1.0`)
   - Update dependencies and metadata

4. **Remove placeholder-package directory:**
   ```bash
   git rm -r placeholder-package/
   ```

5. **Test the build:**
   ```bash
   python -m build
   ```

6. **Create a release tag:**
   ```bash
   git tag v0.1.0
   git push origin placeholder v0.1.0
   ```

### Option 2: Configure Main Branch for Releases

1. **Copy the workflow to main branch:**
   ```bash
   git checkout main
   git checkout placeholder -- .github/workflows/release.yaml
   ```

2. **Adjust the workflow:**
   - Remove the `cd placeholder-package` line
   - Update `packages-dir` to `dist/`

3. **Update pyproject.toml on main:**
   - Change `name = "flowbook"` to `name = "flowbook-python"` if keeping the same PyPI name
   - Or keep separate names for different packages

4. **Push and tag:**
   ```bash
   git push origin main
   git tag v0.1.0
   git push origin v0.1.0
   ```

## PyPI Trusted Publishing Setup

The workflow uses PyPI's trusted publishing feature. To enable it:

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new publisher:
   - PyPI Project Name: `flowbook-python`
   - Owner: `stephenfreund`
   - Repository: `FlowBook`
   - Workflow: `release.yaml`
   - Environment: (leave blank or set to `release`)

If not using trusted publishing, you'll need to:
1. Create a PyPI API token
2. Add it to GitHub Secrets as `PYPI_API_TOKEN`
3. Uncomment the password line in the workflow

## Testing Before Release

Test the package build locally:

```bash
cd placeholder-package  # Or repository root for full package
python -m build
pip install dist/flowbook_python-0.0.1-py3-none-any.whl  # Test installation
```

## Version Management

- Placeholder versions: `0.0.x`
- Beta/development versions: `0.1.0b1`, `0.1.0rc1`, etc.
- Stable versions: `0.1.0`, `0.2.0`, `1.0.0`, etc.

Use semantic versioning: `MAJOR.MINOR.PATCH`

## Notes

- The placeholder package reserves the PyPI name `flowbook-python`
- Users who install the placeholder will get a clear message about the project status
- When updating to the full package, increment the version number appropriately
- Consider adding deprecation notices if changing package names
