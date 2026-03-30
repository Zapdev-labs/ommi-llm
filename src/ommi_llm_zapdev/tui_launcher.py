"""
TUI Launcher for Ommi LLM.

Provides a terminal user interface for managing and running models.
"""

import os
import subprocess
import sys
from pathlib import Path


def get_tui_dir() -> Path:
    """Get the directory containing the TUI files."""
    return Path(__file__).parent / "tui"


def ensure_bun() -> str:
    """Ensure Bun is available, return the path to bun."""
    # Check for bun in PATH
    bun_paths = [
        "bun",
        "/usr/local/bin/bun",
        "/opt/homebrew/bin/bun",
        os.path.expanduser("~/.bun/bin/bun"),
    ]

    for path in bun_paths:
        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return path
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    raise RuntimeError(
        "Bun is not installed or not found in PATH.\n"
        "Please install Bun: https://bun.sh/docs/installation\n"
        "Or ensure 'bun' is in your PATH."
    )


def install_dependencies(tui_dir: Path, bun_path: str) -> None:
    """Install TUI dependencies if needed."""
    node_modules = tui_dir / "node_modules"
    if not node_modules.exists():
        print("Installing TUI dependencies...")
        result = subprocess.run(
            [bun_path, "install"],
            cwd=tui_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error installing dependencies: {result.stderr}", file=sys.stderr)
            raise RuntimeError("Failed to install TUI dependencies")


def launch_tui() -> int:
    """Launch the TUI application."""
    tui_dir = get_tui_dir()

    if not tui_dir.exists():
        print(f"TUI directory not found: {tui_dir}", file=sys.stderr)
        return 1

    try:
        bun_path = ensure_bun()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    # Install dependencies if needed
    try:
        install_dependencies(tui_dir, bun_path)
    except RuntimeError:
        return 1

    # Set environment variable to point to Python executable
    env = os.environ.copy()
    env["OMMI_PYTHON"] = sys.executable

    # Launch the TUI
    try:
        result = subprocess.run(
            [bun_path, "run", "index.ts"],
            cwd=tui_dir,
            env=env,
        )
        return result.returncode
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"Error launching TUI: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(launch_tui())
