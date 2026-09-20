"""Dataset and DICOM loading utilities for the RSNA knee task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

LABEL_COLUMNS = (
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
)


@dataclass(frozen=True)
class StudyView:
    study_id: str
    series_id: str
    anatomical_plane: str
    pixels: Any


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError("Loading dataset tables requires pandas.") from error
    return pd


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise ImportError("Loading DICOM pixels requires numpy.") from error
    return np


def load_metadata(data_dir: str | Path) -> tuple[Any, Any]:
    """Load the study labels and series metadata tables."""
    pandas = _require_pandas()
    data_path = Path(data_dir)
    train = pandas.read_csv(data_path / "train.csv")
    train_series = pandas.read_csv(data_path / "train_series.csv")
    missing = [column for column in LABEL_COLUMNS if column not in train.columns]
    if missing:
        raise ValueError(f"train.csv is missing label columns: {missing}")
    return train, train_series


def _dicom_sort_key(dataset: Any) -> tuple[int, float, str]:
    instance_number = getattr(dataset, "InstanceNumber", 0)
    try:
        instance_number = int(instance_number)
    except (TypeError, ValueError):
        instance_number = 0
    position = getattr(dataset, "ImagePositionPatient", [0.0, 0.0, 0.0])
    try:
        position_value = float(position[-1])
    except (TypeError, ValueError, IndexError):
        position_value = 0.0
    return instance_number, position_value, str(getattr(dataset, "SOPInstanceUID", ""))


def _series_directory(data_dir: Path, study_id: str, series_id: str) -> Path:
    candidates = (
        data_dir / "train_images" / study_id / series_id,
        data_dir / "test_images" / study_id / series_id,
        data_dir / study_id / series_id,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find DICOM series {series_id} for study {study_id} under {data_dir}."
    )


def load_study_view(
    data_dir: str | Path,
    study_id: str,
    series_metadata: Any,
    anatomical_plane: str | None = None,
) -> StudyView:
    """Load one study's DICOM slices for one anatomical view as a pixel stack."""
    numpy = _require_numpy()
    try:
        import pydicom
    except ImportError as error:
        raise ImportError("DICOM loading requires pydicom.") from error

    study_rows = series_metadata[series_metadata["StudyInstanceUID"].astype(str) == str(study_id)]
    if anatomical_plane is not None:
        study_rows = study_rows[study_rows["Anatomical_Plane"] == anatomical_plane]
    if study_rows.empty:
        raise ValueError(f"No series metadata found for study {study_id!r} and view {anatomical_plane!r}.")

    selected_row = study_rows.iloc[0]
    series_id = str(selected_row["SeriesInstanceUID"])
    series_directory = _series_directory(Path(data_dir), str(study_id), series_id)
    datasets = [pydicom.dcmread(str(path)) for path in series_directory.glob("*.dcm")]
    if not datasets:
        raise FileNotFoundError(f"No DICOM files found in {series_directory}.")
    datasets.sort(key=_dicom_sort_key)
    pixels = numpy.stack([dataset.pixel_array for dataset in datasets]).astype("float32")
    plane = str(selected_row.get("Anatomical_Plane", "unknown"))
    return StudyView(str(study_id), series_id, plane, pixels)


def _normalise_stack(pixels: Any, image_size: int) -> Any:
    numpy = _require_numpy()
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise ImportError("Converting image stacks to tensors requires torch.") from error

    stack = pixels.astype("float32")
    lower, upper = numpy.percentile(stack, (1, 99))
    stack = numpy.clip((stack - lower) / max(upper - lower, 1e-6), 0.0, 1.0)
    middle = stack.shape[0] // 2
    indices = numpy.clip(numpy.array([middle - 1, middle, middle + 1]), 0, stack.shape[0] - 1)
    image = torch.from_numpy(stack[indices]).unsqueeze(0)
    return functional.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)


class KneeStudyDataset:
    """Torch-compatible study dataset without importing torch at module load time."""

    def __init__(
        self,
        studies: Any,
        series_metadata: Any,
        data_dir: str | Path,
        image_size: int = 224,
        anatomical_plane: str | None = None,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self.studies = studies.reset_index(drop=True)
        self.series_metadata = series_metadata
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.anatomical_plane = anatomical_plane
        self.transform = transform

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.studies.iloc[index]
        study_id = str(row["StudyInstanceUID"])
        view = load_study_view(self.data_dir, study_id, self.series_metadata, self.anatomical_plane)
        image = _normalise_stack(view.pixels, self.image_size)
        if self.transform is not None:
            image = self.transform(image)
        targets = row[list(LABEL_COLUMNS)].astype("float32").to_numpy()
        mask = ~row[list(LABEL_COLUMNS)].isna().to_numpy()
        return {
            "image": image,
            "target": targets,
            "mask": mask,
            "study_id": study_id,
        }
