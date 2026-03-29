# Publishing ommi-llm to PyPI

This guide walks you through publishing ommi-llm to PyPI (Python Package Index).

## Prerequisites

1. **PyPI Account**: Create one at https://pypi.org/account/register/
2. **TestPyPI Account** (optional but recommended): https://test.pypi.org/account/register/

## Method 1: Manual Publishing

### Step 1: Install Build Tools

```bash
cd /home/dih/opencode-tomfoolery/ommi-llm
pip install build twine
```

### Step 2: Build the Package

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build wheel and source distribution
python -m build
```

This creates:
- `dist/ommi_llm-0.1.0-py3-none-any.whl` (wheel)
- `dist/ommi_llm-0.1.0.tar.gz` (source)

### Step 3: Test on TestPyPI (Recommended)

```bash
# Upload to test server
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ ommi-llm
```

### Step 4: Upload to Real PyPI

```bash
# Upload to production PyPI
python -m twine upload dist/*

# Or with explicit repository
python -m twine upload --repository pypi dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your PyPI API token

### Step 5: Verify Installation

```bash
# Install from PyPI
pip install ommi-llm

# Test
ommi --version
```

## Method 2: Using API Tokens (Recommended)

### Create API Token

1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Name: `ommi-llm-upload`
4. Scope: "Entire account" or specific project
5. Copy the token (starts with `pypi-`)

### Configure Twine

Create `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-XXXXXXXXXXXXXXX-your-token-here

[testpypi]
username = __token__
password = pypi-XXXXXXXXXXXXXXX-your-test-token-here
```

Or use environment variables:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-XXXXXXXXXXXXXXX-your-token-here
python -m twine upload dist/*
```

## Method 3: GitHub Actions (Automatic)

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - 'v*'

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      
      - name: Build package
        run: python -m build
      
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

### Add PyPI Token to GitHub

1. Go to https://github.com/Zapdev-labs/ommi-llm/settings/secrets/actions
2. Click "New repository secret"
3. Name: `PYPI_API_TOKEN`
4. Value: Your PyPI API token

### Automatic Publishing

Now just create a git tag and push:

```bash
# Update version in pyproject.toml first!
git add pyproject.toml
git commit -m "Bump version to 0.1.1"
git tag v0.1.1
git push origin v0.1.1
```

GitHub Actions will automatically build and publish!

## Pre-Publish Checklist

- [ ] Version updated in `pyproject.toml` and `__init__.py`
- [ ] `README.md` is complete
- [ ] `LICENSE` file exists
- [ ] All tests pass (if you have tests)
- [ ] Package installs locally: `pip install -e .`
- [ ] CLI works: `ommi --help`

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- `0.1.0` → `0.1.1` (bug fixes)
- `0.1.0` → `0.2.0` (new features, backward compatible)
- `0.1.0` → `1.0.0` (breaking changes)

## Troubleshooting

### Error: File already exists

```
HTTPError: 400 Bad Request from https://upload.pypi.org/legacy/
File already exists.
```

**Fix**: You cannot overwrite versions on PyPI. Increment the version number:

```toml
# pyproject.toml
version = "0.1.1"  # Change this
```

### Error: Invalid distribution

**Fix**: Clean and rebuild:

```bash
rm -rf dist/ build/
python -m build
```

### Error: Authentication failed

**Fix**: Make sure you're using `__token__` as username, not your actual username.

## Quick Reference

```bash
# One-liner publish
cd /home/dih/opencode-tomfoolery/ommi-llm && \
  rm -rf dist/ && \
  python -m build && \
  python -m twine upload dist/*
```

## After Publishing

Your package will be available at:
- **PyPI**: https://pypi.org/project/ommi-llm/
- **Install**: `pip install ommi-llm`

## Next Steps

1. Add project URLs to `pyproject.toml`:
```toml
[project.urls]
Homepage = "https://github.com/Zapdev-labs/ommi-llm"
Documentation = "https://github.com/Zapdev-labs/ommi-llm#readme"
Repository = "https://github.com/Zapdev-labs/ommi-llm"
Issues = "https://github.com/Zapdev-labs/ommi-llm/issues"
```

2. Create a release on GitHub with release notes
3. Announce on social media / forums
4. Monitor downloads at https://pypistats.org/packages/ommi-llm
