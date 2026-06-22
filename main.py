"""Convenience entry point for running the package from the repository root."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from heritage_perception_analysis.cli import main


if __name__ == "__main__":
    main()
