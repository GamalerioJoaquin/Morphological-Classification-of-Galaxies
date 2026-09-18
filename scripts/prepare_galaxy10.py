"""Download Galaxy10 DECaLS and create deterministic splits."""

from __future__ import annotations

import argparse
import sys
import urllib.request
import warnings
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_classification.galaxy10_data import (  # noqa: E402
    GALAXY10_FILENAME,
    GALAXY10_SHA256,
    build_split_manifest,
    sha256_file,
    validate_arrays,
)


GALAXY10_URL = "https://www.astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5"
ASTRONN_CACHE_DIR = Path.home() / ".astroNN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/verify Galaxy10 DECaLS and write 70/15/15 splits."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "image-manifests" / "galaxy10-decals.csv",
    )
    parser.add_argument(
        "--array-cache",
        type=Path,
        default=ROOT / "data" / "processed" / "galaxy10-images.npy",
        help="Contiguous local array used for fast shuffled training.",
    )
    return parser.parse_args()


def download_galaxy10(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    print(f"Downloading Galaxy10 DECaLS to {destination.resolve()}")
    try:
        urllib.request.urlretrieve(GALAXY10_URL, temporary_path)
        temporary_path.replace(destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def ensure_galaxy10_h5() -> Path:
    h5_path = ASTRONN_CACHE_DIR / "datasets" / GALAXY10_FILENAME
    if h5_path.exists():
        checksum = sha256_file(h5_path)
        if checksum == GALAXY10_SHA256:
            print(f"{h5_path} was found!")
            return h5_path
        print("Cached Galaxy10 file failed checksum validation; downloading again.")

    download_galaxy10(h5_path)
    checksum = sha256_file(h5_path)
    if checksum != GALAXY10_SHA256:
        raise RuntimeError(f"Galaxy10 checksum mismatch: {checksum}")
    return h5_path


def load_galaxy10_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`product` is deprecated as of NumPy 1.25.0",
                category=DeprecationWarning,
            )
            images = np.array(handle["images"])
            labels = np.array(handle["ans"])
    return images, labels


def main() -> int:
    args = parse_args()
    try:
        h5_path = ensure_galaxy10_h5()
        checksum = sha256_file(h5_path)
        images, labels = load_galaxy10_h5(h5_path)
    except (OSError, RuntimeError, urllib.error.URLError) as exc:
        print(
            f"ERROR: Galaxy10 DECaLS could not be downloaded or loaded: {exc}",
            file=sys.stderr,
        )
        return 1

    print(images.shape)
    print(labels.shape)
    try:
        validate_arrays(images, labels)
    except ValueError as exc:
        print(
            f"ERROR: Galaxy10 array validation failed: {exc}",
            file=sys.stderr,
        )
        return 1

    args.array_cache.parent.mkdir(parents=True, exist_ok=True)
    if not args.array_cache.exists():
        print(f"Writing contiguous training cache: {args.array_cache.resolve()}")
        np.save(args.array_cache, images, allow_pickle=False)
    cached_images = np.load(args.array_cache, mmap_mode="r")
    if cached_images.shape != images.shape or cached_images.dtype != images.dtype:
        print("ERROR: Existing NumPy image cache is incompatible.", file=sys.stderr)
        return 1

    manifest = build_split_manifest(labels)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    print(f"HDF5: {h5_path}")
    print(f"SHA256: {checksum}")
    print(f"Manifest: {args.manifest.resolve()}")
    print(f"Training array: {args.array_cache.resolve()}")
    print(manifest.groupby(["split", "label"]).size().unstack(fill_value=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
