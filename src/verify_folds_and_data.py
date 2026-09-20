"""Verify CSV structure, persisted folds, and optional DICOM scanner/site metadata.

Local CSV-only usage:
    python src/verify_folds_and_data.py

With competition DICOM files mounted:
    python src/verify_folds_and_data.py --dicom-root /path/to/competition/data
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

LABEL_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
DICOM_FIELDS = (
    "Manufacturer",
    "InstitutionName",
    "ManufacturerModelName",
    "MagneticFieldStrength",
)
PLANE_TERMS = ("plane", "anatom", "orientation", "sequence", "view", "protocol")


def find_first_dicom(study_id: str, dicom_root: str | Path) -> Path | None:
    """Return the first DICOM file found below a study directory.

    Supports roots containing either ``<study_id>/...`` or the common Kaggle
    ``train_images/<study_id>/...`` and ``test_images/<study_id>/...`` layouts.
    """
    root = Path(dicom_root)
    candidates = (
        root / study_id,
        root / "train_images" / study_id,
        root / "test_images" / study_id,
    )
    for study_directory in candidates:
        if study_directory.is_dir():
            first_file = next(study_directory.rglob("*.dcm"), None)
            if first_file is not None:
                return first_file
    return None


def read_dicom_headers(study_id: str, dicom_root: str | Path) -> dict[str, Any]:
    """Read selected scanner/site fields from a study's first DICOM file.

    Missing files, missing tags, and tag conversion errors become ``None`` for
    the affected field instead of stopping the verification run.
    """
    result: dict[str, Any] = {"StudyInstanceUID": study_id, "DICOMPath": None}
    result.update({field: None for field in DICOM_FIELDS})
    dicom_path = find_first_dicom(study_id, dicom_root)
    if dicom_path is None:
        return result
    result["DICOMPath"] = str(dicom_path)

    try:
        import pydicom
    except ImportError:
        return result

    try:
        dataset = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
    except Exception:
        return result

    for field in DICOM_FIELDS:
        try:
            value = getattr(dataset, field)
            result[field] = str(value) if value is not None else None
        except Exception:
            result[field] = None
    return result


def inspect_series(series: pd.DataFrame) -> None:
    """Print row, study, multiplicity, and view/sequence metadata checks."""
    print("\n=== train_series.csv checks ===")
    total_rows = len(series)
    unique_studies = series["StudyInstanceUID"].nunique()
    counts = series.groupby("StudyInstanceUID").size()
    print(f"Total train_series.csv rows: {total_rows:,}")
    print(f"Unique StudyInstanceUID values: {unique_studies:,}")
    if total_rows == unique_studies:
        print("Each study has exactly one train_series.csv row.")
    else:
        print("Multiple series rows exist for some studies.")
        print("Series rows per study distribution:")
        print(counts.value_counts().sort_index().rename("number_of_studies").to_string())

    descriptive_columns = [
        column for column in series.columns
        if any(term in column.lower() for term in PLANE_TERMS)
    ]
    if descriptive_columns:
        print(f"Columns that look like plane/sequence metadata: {descriptive_columns}")
        for column in descriptive_columns:
            values = series[column].dropna().astype(str).drop_duplicates().tolist()
            print(f"Distinct values in {column}: {values}")
    else:
        print("No anatomical-plane or sequence-like columns were detected.")


def inspect_labels(train: pd.DataFrame) -> None:
    """Print explicit-label and report-only counts."""
    print("\n=== train.csv label checks ===")
    missing = [column for column in LABEL_COLUMNS if column not in train.columns]
    if missing:
        raise KeyError(f"Missing expected label columns: {missing}")
    if "Report" not in train.columns:
        raise KeyError("train.csv is missing the Report column")

    label_frame = train[LABEL_COLUMNS]
    has_any_label = label_frame.notna().any(axis=1)
    has_all_labels = label_frame.notna().all(axis=1)
    has_report = train["Report"].fillna("").astype(str).str.strip().ne("")
    report_only = has_report & ~has_any_label

    print(f"Total unique studies: {train['StudyInstanceUID'].nunique():,}")
    print(f"Explicitly labeled studies with at least one non-null label: {has_any_label.sum():,}")
    print(f"Studies with all 12 labels non-null: {has_all_labels.sum():,}")
    print(f"Report-only studies (Report filled, no labels): {report_only.sum():,}")
    print(f"Label columns ({len(LABEL_COLUMNS)}): {LABEL_COLUMNS}")
    print("Report column: Report")


def sample_fold_studies(folds: pd.DataFrame, sample_size: int = 30, seed: int = 2026) -> pd.DataFrame:
    """Select a fixed-seed sample spread as evenly as possible across folds."""
    required = {"StudyInstanceUID", "fold"}
    missing = required - set(folds.columns)
    if missing:
        raise KeyError(f"folds.csv is missing columns: {sorted(missing)}")
    fold_values = sorted(folds["fold"].dropna().unique())
    if len(fold_values) != 5:
        raise ValueError(f"Expected five existing folds, found {fold_values}")
    if len(folds) < sample_size:
        raise ValueError(f"Cannot sample {sample_size} studies from {len(folds)} rows.")

    base_count, remainder = divmod(sample_size, len(fold_values))
    rng = folds.sample(frac=1, random_state=seed)
    selected = []
    for position, fold_value in enumerate(fold_values):
        count = base_count + (position < remainder)
        fold_rows = rng[rng["fold"] == fold_value].head(count)
        if len(fold_rows) != count:
            raise ValueError(f"Fold {fold_value} does not contain {count} studies to sample.")
        selected.append(fold_rows)
    return pd.concat(selected, ignore_index=True).sort_values(["fold", "StudyInstanceUID"])


def inspect_dicom_sample(folds: pd.DataFrame, dicom_root: str | Path) -> None:
    """Read and print scanner/site fields for 30 balanced sampled studies."""
    print("\n=== DICOM header sample ===")
    sample = sample_fold_studies(folds)
    print("Selected 30 studies with a fixed seed, distributed six per fold.")
    records = [read_dicom_headers(study_id, dicom_root) for study_id in sample["StudyInstanceUID"]]
    results = sample.merge(pd.DataFrame(records), on="StudyInstanceUID", how="left")
    print(results[["StudyInstanceUID", "fold", *DICOM_FIELDS]].to_string(index=False))
    print("\nDistinct non-null header values:")
    for field in DICOM_FIELDS:
        values = results[field].dropna().drop_duplicates().tolist()
        print(f"{field}: {values if values else 'none found'}")
    if results["DICOMPath"].notna().sum() == 0:
        print("No DICOM files were found below the supplied root; run this part on Kaggle with images mounted.")
    elif results[list(DICOM_FIELDS)].notna().any().any():
        print("At least one usable scanner/site-related header value was found.")
    else:
        print("DICOM files were found, but none of the requested fields were populated.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--dicom-root",
        type=Path,
        default=None,
        help="Mounted competition data root; omit for CSV-only verification.",
    )
    args = parser.parse_args()

    print("Starting data and fold verification. folds.csv will only be read, never rebuilt.")
    train = pd.read_csv(args.data_dir / "train.csv")
    train_series = pd.read_csv(args.data_dir / "train_series.csv")
    folds = pd.read_csv(args.data_dir / "folds.csv")
    inspect_series(train_series)
    inspect_labels(train)

    print("\n=== folds.csv check ===")
    print(f"Fold assignment rows: {len(folds):,}")
    print(f"Fold values: {sorted(folds['fold'].dropna().unique().tolist())}")
    print("This script does not rebuild folds.csv.")

    if args.dicom_root is None:
        print("\nDICOM header check skipped: no --dicom-root was provided.")
        print("On Kaggle, rerun with --dicom-root pointing at the mounted competition data.")
    else:
        inspect_dicom_sample(folds, args.dicom_root)

    print("\nVerification complete. Use the DICOM sample output to decide whether study-based folds are sufficient.")


if __name__ == "__main__":
    main()
