"""Recover SDSS identities by position and download reproducible JPEG cutouts."""

from __future__ import annotations

import csv
import hashlib
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from PIL import Image


SKYSERVER_RELEASE = "dr18"
SKYSERVER_ROOT = f"https://skyserver.sdss.org/{SKYSERVER_RELEASE}/SkyServerWS"
SQL_ENDPOINT = f"{SKYSERVER_ROOT}/SearchTools/SqlSearch"
JPEG_ENDPOINT = f"{SKYSERVER_ROOT}/ImgCutout/getjpeg"
USER_AGENT = "galaxy-classification-dataset-builder/1.0"
MATCH_COLUMNS = (
    "coordinate_id",
    "sdss_objid",
    "sdss_ra",
    "sdss_dec",
    "match_distance_arcsec",
    "run",
    "rerun",
    "camcol",
    "field",
    "sdss_modelMag_u",
    "sdss_modelMag_g",
    "sdss_modelMag_r",
    "sdss_modelMag_i",
    "sdss_modelMag_z",
)


class SdssImageError(RuntimeError):
    """Raised when catalog matching or cutout retrieval cannot be completed."""


@dataclass(frozen=True)
class CutoutConfig:
    width: int = 128
    height: int = 128
    scale_arcsec_per_pixel: float = 0.396
    workers: int = 4
    retries: int = 3

    def __post_init__(self) -> None:
        if not 1 <= self.width <= 2048 or not 1 <= self.height <= 2048:
            raise ValueError("Cutout width and height must be between 1 and 2048.")
        if self.scale_arcsec_per_pixel <= 0:
            raise ValueError("Cutout scale must be positive.")
        if self.workers < 1 or self.retries < 1:
            raise ValueError("workers and retries must be positive.")


def build_targets(prepared: pd.DataFrame, labels: Iterable[str]) -> pd.DataFrame:
    """Return one non-conflicting target per exact coordinate group."""

    selected_labels = set(labels)
    targets = prepared.loc[
        prepared["source_label"].isin(selected_labels)
        & ~prepared["coordinate_has_label_conflict"]
    ].copy()
    targets = targets.drop_duplicates("coordinate_id", keep="first")
    targets = targets[
        [
            "coordinate_id",
            "legacy_objID",
            "ra",
            "dec",
            "source_label",
            "coordinate_group_size",
        ]
    ].reset_index(drop=True)
    return targets


def stable_split(object_id: str) -> str:
    """Assign a deterministic 80/10/10 split without global dataset state."""

    bucket = int(hashlib.sha256(object_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "validation"
    return "test"


def legacy_objid_is_compatible(legacy_objid: str, recovered_objid: str) -> bool:
    """Check whether an exact ID rounds to the precision-damaged legacy value."""

    try:
        damaged = Decimal(str(legacy_objid).replace(",", "."))
        recovered = Decimal(str(recovered_objid))
        return recovered.quantize(damaged) == damaged
    except (InvalidOperation, ValueError):
        return False


def _values_clause(targets: pd.DataFrame) -> str:
    rows = []
    for row in targets.itertuples(index=False):
        coordinate_id = str(row.coordinate_id).replace("'", "''")
        rows.append(
            f"('{coordinate_id}',{float(row.ra):.17g},{float(row.dec):.17g})"
        )
    return ",".join(rows)


def build_match_query(targets: pd.DataFrame, radius_arcsec: float) -> str:
    """Build one SkyServer SQL query for a small batch of positions."""

    if targets.empty:
        raise ValueError("Cannot build a match query for an empty batch.")
    if radius_arcsec <= 0:
        raise ValueError("Match radius must be positive.")
    radius_arcmin = radius_arcsec / 60.0
    return (
        "SELECT t.coordinate_id,p.objID AS sdss_objid,p.ra AS sdss_ra,"
        "p.dec AS sdss_dec,n.distance*60 AS match_distance_arcsec,"
        "p.run,p.rerun,p.camcol,p.field,"
        "p.modelMag_u AS sdss_modelMag_u,p.modelMag_g AS sdss_modelMag_g,"
        "p.modelMag_r AS sdss_modelMag_r,p.modelMag_i AS sdss_modelMag_i,"
        "p.modelMag_z AS sdss_modelMag_z "
        f"FROM (VALUES {_values_clause(targets)}) AS t(coordinate_id,ra,dec) "
        f"CROSS APPLY dbo.fGetNearestObjEq(t.ra,t.dec,{radius_arcmin:.17g}) AS n "
        "JOIN PhotoPrimary AS p ON p.objID=n.objID"
    )


def _read_url(url: str, timeout: float = 60.0) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_skyserver_csv(payload: bytes) -> pd.DataFrame:
    """Parse SkyServer CSV, whose table-name comment precedes the header."""

    text = payload.decode("utf-8-sig")
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    if not lines:
        return pd.DataFrame(columns=MATCH_COLUMNS)
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    return pd.DataFrame(rows)


def match_batch(
    targets: pd.DataFrame,
    radius_arcsec: float = 1.0,
    reader: Callable[[str], bytes] = _read_url,
) -> pd.DataFrame:
    """Resolve one target batch against the nearest DR18 primary photo object."""

    query = build_match_query(targets, radius_arcsec)
    url = SQL_ENDPOINT + "?" + urlencode({"cmd": query, "format": "csv"})
    matches = parse_skyserver_csv(reader(url))
    if matches.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)
    missing = set(MATCH_COLUMNS).difference(matches.columns)
    if missing:
        raise SdssImageError(f"SkyServer response is missing columns: {sorted(missing)}")
    return matches.loc[:, MATCH_COLUMNS]


def match_catalog(
    targets: pd.DataFrame,
    *,
    radius_arcsec: float = 1.0,
    batch_size: int = 100,
    reader: Callable[[str], bytes] = _read_url,
) -> pd.DataFrame:
    """Resolve all targets in bounded URL-sized batches."""

    if batch_size < 1 or batch_size > 200:
        raise ValueError("batch_size must be between 1 and 200.")
    batches = []
    for start in range(0, len(targets), batch_size):
        batches.append(
            match_batch(
                targets.iloc[start : start + batch_size],
                radius_arcsec=radius_arcsec,
                reader=reader,
            )
        )
    if not batches:
        return pd.DataFrame(columns=MATCH_COLUMNS)
    return pd.concat(batches, ignore_index=True)


def cutout_url(ra: float, dec: float, config: CutoutConfig) -> str:
    parameters = {
        "ra": f"{ra:.17g}",
        "dec": f"{dec:.17g}",
        "scale": f"{config.scale_arcsec_per_pixel:.6g}",
        "width": str(config.width),
        "height": str(config.height),
    }
    return JPEG_ENDPOINT + "?" + urlencode(parameters)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_jpeg(path: Path, config: CutoutConfig) -> None:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.format != "JPEG":
            raise SdssImageError(f"SkyServer did not return JPEG data for {path.name}.")
        if image.size != (config.width, config.height):
            raise SdssImageError(
                f"Unexpected cutout size for {path.name}: {image.size}."
            )


def download_cutout(
    row: Mapping[str, object],
    output_dir: Path,
    config: CutoutConfig,
    reader: Callable[[str], bytes] = _read_url,
) -> dict[str, str]:
    """Download or verify one cutout and return its manifest fields."""

    output_dir.mkdir(parents=True, exist_ok=True)
    coordinate_id = str(row["coordinate_id"])
    destination = output_dir / f"{coordinate_id}.jpg"
    if destination.is_file():
        _validate_jpeg(destination, config)
    else:
        temporary = destination.with_suffix(".jpg.part")
        last_error: Exception | None = None
        for attempt in range(config.retries):
            try:
                temporary.write_bytes(
                    reader(cutout_url(float(row["ra"]), float(row["dec"]), config))
                )
                _validate_jpeg(temporary, config)
                temporary.replace(destination)
                break
            except (HTTPError, URLError, OSError, SdssImageError) as exc:
                last_error = exc
                temporary.unlink(missing_ok=True)
                if attempt + 1 < config.retries:
                    time.sleep(2**attempt)
        else:
            raise SdssImageError(f"Could not download {coordinate_id}: {last_error}")

    object_id = (
        f"sdss_dr18_{row['sdss_objid']}"
        if str(row.get("sdss_objid", "")).strip()
        else coordinate_id
    )
    return {
        "coordinate_id": coordinate_id,
        "object_id": object_id,
        "split": stable_split(object_id),
        "image_path": destination.name,
        "sha256": sha256_file(destination),
        "source": "SDSS DR18 SkyServer JPEG cutout",
        "cutout_url": cutout_url(float(row["ra"]), float(row["dec"]), config),
    }


def download_cutouts(
    targets: pd.DataFrame,
    output_dir: Path,
    config: CutoutConfig,
    reader: Callable[[str], bytes] = _read_url,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Download targets concurrently while isolating per-object failures."""

    completed: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        future_rows = {
            executor.submit(
                download_cutout,
                row._asdict(),
                output_dir,
                config,
                reader,
            ): row
            for row in targets.itertuples(index=False)
        }
        for future in as_completed(future_rows):
            row = future_rows[future]
            try:
                completed.append(future.result())
            except Exception as exc:  # keep a full-dataset run resumable
                failures.append(
                    {"coordinate_id": str(row.coordinate_id), "error": str(exc)}
                )
    return pd.DataFrame(completed), failures
