"""Smoke-test the Python environment required by the project notebooks."""

from __future__ import annotations

import importlib
import platform
import sys


IMPORTS = {
    "numpy": "NumPy",
    "pandas": "pandas",
    "scipy": "SciPy",
    "matplotlib": "Matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "imblearn": "imbalanced-learn",
    "joblib": "joblib",
    "astropy": "Astropy",
    "astroquery": "astroquery",
    "h5py": "h5py",
    "missingno": "missingno",
    "cv2": "OpenCV",
    "torch": "PyTorch",
    "torchvision": "torchvision",
    "PIL": "Pillow",
    "IPython": "IPython",
}


def main() -> int:
    print(f"Python {platform.python_version()} ({sys.executable})")
    if sys.version_info[:2] != (3, 10):
        print("ERROR: this project is tested with Python 3.10.", file=sys.stderr)
        return 1

    failures: list[str] = []
    for module_name, display_name in IMPORTS.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # report every failed dependency in one run
            failures.append(f"{display_name}: {exc!r}")
            continue

        version = getattr(module, "__version__", "version unavailable")
        print(f"OK  {display_name}: {version}")

    if failures:
        print("\nEnvironment validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nEnvironment validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
