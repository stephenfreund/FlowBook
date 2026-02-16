# FlowBook Placeholder Branch

This branch contains the setup for publishing a placeholder package to PyPI under the name `flowbook-python`.

## Purpose

This placeholder package reserves the `flowbook-python` name on PyPI while the main FlowBook project is under active development. When the main project is ready for release, the package can be seamlessly transitioned to the full version.

## What's Included

### Package Structure

```
placeholder-package/
├── LICENSE                    # BSD 3-Clause license
├── README.md                  # User-facing documentation
├── pyproject.toml            # Package configuration
└── flowbook/                 # Python package (imports as "flowbook")
    └── __init__.py           # Minimal module with version info
```

### Automation

- **`.github/workflows/release.yaml`** - Automated PyPI publishing
  - Triggers when you create a new release in GitHub
  - Builds the package automatically
  - Publishes to PyPI using trusted publishing
  - Uploads built packages as release assets

### Documentation

- **`PLACEHOLDER_SETUP.md`** - Complete implementation details, usage instructions, and verification steps
- **`TRANSITION.md`** - Guide for transitioning to the full FlowBook package
- **`setup-placeholder.sh`** - Helper script for testing the setup

## Quick Start

### 1. Set Up PyPI Trusted Publishing (One-Time)

Go to https://pypi.org/manage/account/publishing/ and add:
- PyPI Project Name: `flowbook-python`
- Owner: `stephenfreund`
- Repository: `FlowBook`
- Workflow: `release.yaml`
- Environment: (leave blank)

### 2. Publish the Placeholder

1. Push this branch to GitHub (if not already pushed):
   ```bash
   git push origin placeholder
   ```

2. Create a new release:
   - Go to https://github.com/stephenfreund/FlowBook/releases/new
   - Select the `placeholder` branch
   - Create tag: `v0.0.1`
   - Title: `v0.0.1 - Placeholder Package`
   - Description: "Initial placeholder package to reserve flowbook-python name on PyPI"
   - Click "Publish release"

3. The workflow will automatically:
   - Build the package
   - Publish to PyPI
   - Attach build artifacts to the release

### 3. Verify Publication

After the workflow completes:
```bash
pip install flowbook-python
python -c "import flowbook; print(flowbook.__version__)"
```

## Testing Locally

Test the package build before publishing:

```bash
cd placeholder-package
python -m pip install build
python -m build
pip install dist/flowbook-0.0.1-py3-none-any.whl
python -c "import flowbook; print(flowbook.__version__)"
```

Or use the helper script:
```bash
./setup-placeholder.sh
```

## Package Details

- **PyPI Name**: `flowbook-python`
- **Import Name**: `flowbook`
- **Version**: `0.0.1`
- **Python Support**: 3.8+
- **License**: BSD 3-Clause
- **Type**: Pure Python wheel

## Transitioning to Full Package

When ready to publish the full FlowBook package, see `TRANSITION.md` for detailed instructions. The key steps are:

1. Merge main branch into placeholder
2. Update workflow to build from repository root
3. Update version in pyproject.toml
4. Remove placeholder-package directory
5. Create a new release with updated version

The transition is designed to be seamless for users - they simply upgrade their installed package.

## File Descriptions

| File | Purpose |
|------|---------|
| `placeholder-package/pyproject.toml` | Package configuration (name, version, dependencies) |
| `placeholder-package/README.md` | User-facing documentation explaining the placeholder |
| `placeholder-package/flowbook/__init__.py` | Minimal Python module with version info |
| `placeholder-package/LICENSE` | BSD 3-Clause license |
| `.github/workflows/release.yaml` | GitHub Actions workflow for PyPI publishing |
| `PLACEHOLDER_SETUP.md` | Implementation details and usage guide |
| `TRANSITION.md` | Guide for transitioning to full package |
| `setup-placeholder.sh` | Helper script for testing the setup |

## Important Notes

- ⚠️ This is a **placeholder branch** - do not use for main development
- ✅ The placeholder package installs without errors and can be imported
- ✅ Users get a clear message that this is a placeholder
- ✅ The transition path to the full package is well-documented
- ✅ The workflow only publishes when you explicitly create a release
- ✅ Uses PyPI trusted publishing (no API tokens needed)

## Support

For questions or issues:
- See `PLACEHOLDER_SETUP.md` for detailed setup information
- See `TRANSITION.md` for transition guidance
- GitHub: https://github.com/stephenfreund/FlowBook

## Next Steps After Publishing

1. Monitor PyPI for the package: https://pypi.org/project/flowbook-python/
2. Verify the package installs correctly: `pip install flowbook-python`
3. When ready, follow `TRANSITION.md` to publish the full package
4. Consider setting up a simple project website or documentation

---

**Branch Status**: Ready for publication  
**Package Status**: Placeholder (v0.0.1)  
**Last Updated**: 2026-02-16
