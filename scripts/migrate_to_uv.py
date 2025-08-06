#!/usr/bin/env python3
"""
Migration script to help transition from pip to uv.

This script helps developers migrate their existing development environment
to use uv instead of pip.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def check_uv_installed() -> bool:
    """Check if uv is installed."""
    try:
        result = run_command(["uv", "--version"], check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_uv() -> None:
    """Install uv using the official installer."""
    print("Installing uv...")
    try:
        # Try the official installer
        result = run_command([
            "curl", "-LsSf", "https://astral.sh/uv/install.sh"
        ], check=False)
        if result.returncode == 0:
            subprocess.run(["sh"], input=result.stdout, text=True, check=True)
        else:
            # Fallback to pip
            print("Falling back to pip installation...")
            run_command([sys.executable, "-m", "pip", "install", "uv"])
    except Exception as e:
        print(f"Failed to install uv: {e}")
        print("Please install uv manually: https://docs.astral.sh/uv/getting-started/installation/")
        sys.exit(1)


def migrate_environment() -> None:
    """Migrate from pip virtual environment to uv."""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("Migrating to uv environment...")
    
    # Remove old virtual environment if it exists
    old_venv_paths = ["venv", "env", ".env"]
    for venv_path in old_venv_paths:
        if Path(venv_path).exists():
            print(f"Found old virtual environment: {venv_path}")
            choice = input(f"Remove {venv_path}? (y/N): ").lower()
            if choice == 'y':
                import shutil
                shutil.rmtree(venv_path)
                print(f"Removed {venv_path}")
    
    # Create new uv environment
    print("Creating new uv virtual environment...")
    run_command(["uv", "venv"])
    
    # Install dependencies
    print("Installing dependencies with uv...")
    run_command(["uv", "sync"])
    
    # Generate lock file
    print("Generating lock file...")
    run_command(["uv", "lock"])
    
    print("\nMigration complete!")
    print("\nNext steps:")
    print("1. Activate the new environment: source .venv/bin/activate")
    print("2. Or use uv run for commands: uv run python script.py")
    print("3. Add new dependencies with: uv add package-name")
    print("4. Consider removing requirements.txt after testing")


def main() -> None:
    """Main migration function."""
    print("VibeCode AI Mentor - UV Migration Script")
    print("=" * 40)
    
    # Check if uv is installed
    if not check_uv_installed():
        print("uv is not installed.")
        choice = input("Install uv now? (Y/n): ").lower()
        if choice != 'n':
            install_uv()
        else:
            print("Please install uv first: https://docs.astral.sh/uv/")
            sys.exit(1)
    else:
        print("✓ uv is already installed")
    
    # Run migration
    migrate_environment()


if __name__ == "__main__":
    main()