#!/bin/bash
# Setup script for creating the placeholder branch
# This script can be run to create or verify the placeholder branch setup

set -e  # Exit on error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=================================================="
echo "FlowBook Placeholder Branch Setup Script"
echo "=================================================="
echo ""

# Check if we're in the right repository
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: This script must be run from the FlowBook repository root"
    exit 1
fi

echo "Current branch: $(git branch --show-current)"
echo ""

# Ask user if they want to create or switch to placeholder branch
read -p "Create or switch to placeholder branch? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Check if placeholder branch exists
if git show-ref --verify --quiet refs/heads/placeholder; then
    echo "Placeholder branch already exists. Switching to it..."
    git checkout placeholder
else
    echo "Creating new placeholder branch..."
    git checkout -b placeholder
fi

echo ""
echo "Verifying placeholder-package directory..."

if [ ! -d "placeholder-package" ]; then
    echo "ERROR: placeholder-package directory not found!"
    echo "Please ensure the placeholder branch is properly set up."
    exit 1
fi

echo "✓ placeholder-package directory exists"

# List the contents
echo ""
echo "Placeholder package structure:"
tree -L 2 placeholder-package/ 2>/dev/null || find placeholder-package -type f -o -type d | sort

echo ""
echo "Verifying workflow file..."

if [ ! -f ".github/workflows/release.yaml" ]; then
    echo "ERROR: .github/workflows/release.yaml not found!"
    exit 1
fi

echo "✓ .github/workflows/release.yaml exists"

echo ""
echo "Testing package build..."

cd placeholder-package

# Install build dependencies if needed
if ! python -c "import build" 2>/dev/null; then
    echo "Installing build dependencies..."
    pip install --quiet build hatchling
fi

# Build the package
echo "Building package..."
python -m build

if [ -d "dist" ]; then
    echo "✓ Package built successfully"
    echo ""
    echo "Built packages:"
    ls -lh dist/
    
    # Optionally test installation
    echo ""
    read -p "Test package installation? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        WHEEL_FILE=$(ls dist/*.whl | head -1)
        if [ -f "$WHEEL_FILE" ]; then
            echo "Installing $WHEEL_FILE..."
            pip install "$WHEEL_FILE"
            echo ""
            echo "Testing import..."
            python -c "import flowbook; print('Version:', flowbook.__version__)"
            echo "✓ Package installation successful"
        fi
    fi
else
    echo "ERROR: Build failed - dist/ directory not created"
    exit 1
fi

cd "$REPO_ROOT"

echo ""
echo "=================================================="
echo "Setup verification complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Set up PyPI trusted publishing (see PLACEHOLDER_SETUP.md)"
echo "2. Push the placeholder branch: git push origin placeholder"
echo "3. Create a new release in GitHub UI with tag v0.0.1"
echo "   (GitHub Releases → New Release → Select placeholder branch)"
echo "4. Monitor GitHub Actions for successful PyPI upload"
echo ""
echo "For more details, see:"
echo "  - PLACEHOLDER_SETUP.md (implementation details)"
echo "  - TRANSITION.md (how to switch to full package)"
echo ""
