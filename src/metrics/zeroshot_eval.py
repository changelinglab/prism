"""Evaluate zero-shot prediction results against ground truth.

This module evaluates JSONL prediction files from the Gemini direct-prompt runner
using **the same metric definitions as PRiSM probing** (src/recipe/common/classification_module.py).

Supports:
- classification (class_id): cmul2arcticl1, edacc, fleurs, ultrasuite_child
- ordinal/regression (score): speechocean, easycall, uaspeech
- geolocation (lat, lon in decimal degrees): vaanigeo

Usage:
    python -m src.metrics.zeroshot_eval \
        --dataset vaanigeo \
        --run_dir exp/runs/vaani/20251230_201815

    python -m src.metrics.zeroshot_eval \
        --dataset cmul2arcticl1 \
        --predictions "exp/runs/dp_gemini_l1cls/prediction.*.jsonl"

    python -m src.metrics.zeroshot_eval \
        --dataset speechocean \
        --run_dir exp/runs/speechocean/20251231_012345 \
        --output exp/eval_results/speechocean_report
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.metrics.geolocation import GeolocationDistanceError, GeolocationMissRate

# ---------------------------------------------------------------------------
# Lazy imports for metrics (scipy may not be installed)
# ---------------------------------------------------------------------------


def _import_torchmetrics():
    from torchmetrics.classification import (
        CohenKappa,
        MulticlassAccuracy,
        MulticlassF1Score,
    )
    from torchmetrics.regression import MeanAbsoluteError, PearsonCorrCoef

    return {
        "CohenKappa": CohenKappa,
        "MulticlassAccuracy": MulticlassAccuracy,
        "MulticlassF1Score": MulticlassF1Score,
        "MeanAbsoluteError": MeanAbsoluteError,
        "PearsonCorrCoef": PearsonCorrCoef,
    }


def _import_kendalltau():
    """Import KendallTau metric from src/metrics (requires scipy)."""
    try:
        from src.metrics.kendalltau import KendallTau

        return KendallTau
    except ImportError as e:
        print(
            f"Warning: KendallTau import failed ({e}).\n"
            "  Install scipy: uv pip install scipy\n"
            "  KendallTau metric will be skipped."
        )
        return None


# ---------------------------------------------------------------------------
# Task specifications (same as DOWNSTREAM_TASK_SPECS.md)
# ---------------------------------------------------------------------------


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GEOLOCATION = "geolocation"


@dataclass
class TaskSpec:
    task_type: TaskType
    num_classes: int  # K (for cls: #classes, for reg: max+1, for geo: 2)
    min_value: int = 0
    max_value: int = 0  # only for regression
    pred_key: str = "processed_transcript"  # key in pred dict


TASK_SPECS: Dict[str, TaskSpec] = {
    # Classification
    "cmul2arcticl1": TaskSpec(
        TaskType.CLASSIFICATION, num_classes=7, pred_key="processed_transcript"
    ),
    "edacc": TaskSpec(
        TaskType.CLASSIFICATION, num_classes=13, pred_key="processed_transcript"
    ),
    "fleurs": TaskSpec(
        TaskType.CLASSIFICATION, num_classes=24, pred_key="processed_transcript"
    ),
    "ultrasuite_child": TaskSpec(
        TaskType.CLASSIFICATION, num_classes=2, pred_key="processed_transcript"
    ),
    # Ordinal/Regression
    "speechocean": TaskSpec(
        TaskType.REGRESSION,
        num_classes=11,
        min_value=0,
        max_value=10,
        pred_key="processed_transcript",
    ),
    "easycall": TaskSpec(
        TaskType.REGRESSION,
        num_classes=4,
        min_value=0,
        max_value=3,
        pred_key="processed_transcript",
    ),
    "uaspeech": TaskSpec(
        TaskType.REGRESSION,
        num_classes=5,
        min_value=0,
        max_value=4,
        pred_key="processed_transcript",
    ),
    # Geolocation (lat/lon in degrees)
    "vaanigeo": TaskSpec(
        TaskType.GEOLOCATION, num_classes=2, pred_key="predicted_transcript"
    ),
}


# ---------------------------------------------------------------------------
# Record normalization (distributed_inference.py format)
# ---------------------------------------------------------------------------


def normalize_record(record: Any) -> Optional[Dict[str, Any]]:
    """Normalize one JSON record from distributed inference outputs.

    Supported formats:
    - Flat format: {"pred": ..., "passthrough": {...}, ...}
    - Sharded format: {"<idx>": {"pred": ..., "passthrough": {...}}}

    Returns:
        A flat dict (optionally with "idx") or None if the record is not usable.
    """
    if not isinstance(record, dict):
        return None

    # Already flat
    if "pred" in record or "passthrough" in record:
        return record

    # distributed_inference.py writes one-key dict per line: {i: {...}}
    if len(record) == 1:
        ((idx, payload),) = record.items()
        if isinstance(payload, dict) and (
            "pred" in payload or "passthrough" in payload
        ):
            out: Dict[str, Any] = {"idx": idx}
            out.update(payload)
            return out

    # Unknown dict shape: keep as-is (will likely be counted invalid later)
    return record


def load_predictions(
    pattern: str, exclude_cache_error: bool = True
) -> List[Dict[str, Any]]:
    """Load predictions from JSONL file(s) matching the pattern.

    Args:
        pattern: Glob pattern for JSONL files
        exclude_cache_error: If True, exclude files matching *cache*.jsonl and *error(s)*.jsonl
    """
    files = sorted(glob(pattern))
    if exclude_cache_error:
        # Exclude cache + error shards (e.g., prediction.cache.jsonl, prediction.errors.0.jsonl)
        files = [
            f
            for f in files
            if ".cache" not in Path(f).name
            and ".error" not in Path(f).name
            and ".errors" not in Path(f).name
        ]
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    records = []
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    norm = normalize_record(record)
                    if norm is not None:
                        records.append(norm)
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Skipping malformed JSON at {filepath}:{line_num}: {e}"
                    )

    print(f"Loaded {len(records)} records from {len(files)} file(s)")
    return records


def filter_test_split(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter records to keep only test split (if split info is available)."""
    filtered = []
    for rec in records:
        passthrough = rec.get("passthrough", {})
        split = passthrough.get("split")
        # If split is available, keep only test; otherwise keep all
        if split is None or split == "test":
            filtered.append(rec)
    if len(filtered) < len(records):
        print(f"Filtered to {len(filtered)} test records (from {len(records)} total)")
    return filtered


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------


def extract_classification(
    records: List[Dict[str, Any]], spec: TaskSpec
) -> Tuple[List[int], List[int], int, int]:
    """Extract classification predictions and targets."""
    preds, targets = [], []
    invalid = 0
    for rec in records:
        pred = rec.get("pred")
        if pred and isinstance(pred, list):
            pred = pred[0]
        passthrough = rec.get("passthrough", {})
        target = passthrough.get("target")

        if target is None:
            invalid += 1
            continue

        if pred is None or not isinstance(pred, dict):
            invalid += 1
            continue

        # Direct-prompt (Qwen/Gemini) may store scalar under "result"; transcript uses spec.pred_key
        class_id = pred.get(spec.pred_key) or pred.get("result") or pred.get("class_id")
        if class_id is None or not isinstance(class_id, (int, float, str)):
            invalid += 1
            continue

        try:
            class_id = int(class_id)
        except ValueError:
            invalid += 1
            continue
        if class_id < 0 or class_id >= spec.num_classes:
            invalid += 1
            continue

        preds.append(class_id)
        targets.append(int(target))

    return preds, targets, len(preds), invalid


def extract_regression(
    records: List[Dict[str, Any]], spec: TaskSpec, clamp: bool = True
) -> Tuple[List[float], List[int], List[int], int, int]:
    """Extract regression predictions and targets.

    Returns:
        (raw_preds, int_preds, targets, valid_count, invalid_count)
        - raw_preds: float scores (for PCC)
        - int_preds: round+clamp int scores (for acc/f1/mae/kappa/kendalltau)
    """
    raw_preds, int_preds, targets = [], [], []
    invalid = 0

    for rec in records:
        pred = rec.get("pred")
        if pred and isinstance(pred, list):
            pred = pred[0]
        passthrough = rec.get("passthrough", {})
        target = passthrough.get("target")

        if target is None:
            invalid += 1
            continue

        if pred is None or not isinstance(pred, dict):
            invalid += 1
            continue

        # Direct-prompt (Qwen/Gemini) may store scalar under "result"; transcript uses spec.pred_key
        score = pred.get(spec.pred_key) or pred.get("result") or pred.get("score")
        if score is None or not isinstance(score, (int, float, str)):
            invalid += 1
            continue

        try:
            raw_score = float(score)
        except ValueError:
            invalid += 1
            continue
        int_score = int(round(raw_score))
        if clamp:
            int_score = max(spec.min_value, min(spec.max_value, int_score))
        elif int_score < spec.min_value or int_score > spec.max_value:
            invalid += 1
            continue

        raw_preds.append(raw_score)
        int_preds.append(int_score)
        targets.append(int(target))

    return raw_preds, int_preds, targets, len(targets), invalid


def extract_geolocation(
    records: List[Dict[str, Any]], spec: TaskSpec = TASK_SPECS["vaanigeo"]
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float]], int, int]:
    """Extract geolocation predictions (lat/lon in degrees) and targets (lat_rad, lon_rad).

    - The model outputs {"lat": <deg>, "lon": <deg>}.
    - Targets are [lat_rad, lon_rad].
    - We convert predicted (lat_deg, lon_deg)
      -> (lat_rad, lon_rad) -> (x, y, z) unit vector.
    """
    preds_xyz, targets_latlon = [], []
    invalid = 0

    for rec in records:
        pred = rec.get("pred")
        if pred and isinstance(pred, list):
            pred = pred[0]
        passthrough = rec.get("passthrough", {})
        target = passthrough.get("target")

        if target is None:
            invalid += 1
            continue

        if pred is None or not isinstance(pred, dict):
            invalid += 1
            continue

        try:
            pred = json.loads(pred[spec.pred_key])
        except Exception:
            pred = pred
        lat_deg = pred.get("lat")
        lon_deg = pred.get("lon")
        if any(
            v is None or not isinstance(v, (int, float)) for v in [lat_deg, lon_deg]
        ):
            invalid += 1
            continue

        lat_deg = float(lat_deg)
        lon_deg = float(lon_deg)
        if not (-90.0 <= lat_deg <= 90.0) or not (-180.0 <= lon_deg <= 180.0):
            invalid += 1
            continue

        # degrees -> radians
        lat_rad_pred = math.radians(lat_deg)
        lon_rad_pred = math.radians(lon_deg)

        # radians -> unit sphere cartesian
        x = math.cos(lat_rad_pred) * math.cos(lon_rad_pred)
        y = math.cos(lat_rad_pred) * math.sin(lon_rad_pred)
        z = math.sin(lat_rad_pred)

        # Target should be [lat_rad, lon_rad]
        if not isinstance(target, (list, tuple)) or len(target) != 2:
            invalid += 1
            continue

        lat_rad, lon_rad = float(target[0]), float(target[1])

        preds_xyz.append((x, y, z))
        targets_latlon.append((lat_rad, lon_rad))

    return preds_xyz, targets_latlon, len(targets_latlon), invalid


# ---------------------------------------------------------------------------
# Metric computation (matching src/recipe/common/classification_module.py)
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    preds: List[int], targets: List[int], num_classes: int
) -> Dict[str, float]:
    """Compute classification metrics (same as ClassificationModel)."""
    tm = _import_torchmetrics()
    preds_t = torch.tensor(preds)
    targets_t = torch.tensor(targets)

    acc = tm["MulticlassAccuracy"](num_classes=num_classes)
    f1 = tm["MulticlassF1Score"](num_classes=num_classes, average="macro")

    return {
        "acc": acc(preds_t, targets_t).item(),
        "f1": f1(preds_t, targets_t).item(),
    }


def compute_regression_metrics(
    raw_preds: List[float],
    int_preds: List[int],
    targets: List[int],
    num_classes: int,
) -> Dict[str, float]:
    """Compute ordinal/regression metrics (same as ClassificationModel).

    Uses:
    - int_preds for acc, f1, mae, cohenkappa, kendalltau
    - raw_preds for pcc (continuous)
    """
    tm = _import_torchmetrics()
    KendallTau = _import_kendalltau()

    int_preds_t = torch.tensor(int_preds)
    raw_preds_t = torch.tensor(raw_preds)
    targets_t = torch.tensor(targets)

    acc = tm["MulticlassAccuracy"](num_classes=num_classes)
    f1 = tm["MulticlassF1Score"](num_classes=num_classes, average="macro")
    mae = tm["MeanAbsoluteError"]()
    cohenkappa = tm["CohenKappa"](
        task="multiclass", num_classes=num_classes, weights="quadratic"
    )
    pcc = tm["PearsonCorrCoef"]()

    metrics = {
        "acc": acc(int_preds_t, targets_t).item(),
        "f1": f1(int_preds_t, targets_t).item(),
        "mae": mae(int_preds_t.float(), targets_t.float()).item(),
        "cohenkappa": cohenkappa(int_preds_t, targets_t).item(),
        "pcc": pcc(raw_preds_t, targets_t.float()).item(),
    }

    if KendallTau is not None:
        kt = KendallTau()
        kt.update(int_preds_t, targets_t)
        metrics["kendalltau"] = kt.compute().item()
    else:
        metrics["kendalltau"] = None

    return metrics


def compute_geolocation_metrics(
    preds_xyz: List[Tuple[float, float, float]],
    targets_latlon: List[Tuple[float, float]],
) -> Dict[str, float]:
    """Compute geolocation metrics (same as ClassificationModel)."""
    preds_t = torch.tensor(preds_xyz)  # (N, 3)
    targets_t = torch.tensor(targets_latlon)  # (N, 2) - lat_rad, lon_rad

    err_km = GeolocationDistanceError()
    miss_rate_1 = GeolocationMissRate(k=1)
    miss_rate_5 = GeolocationMissRate(k=5)
    miss_rate_10 = GeolocationMissRate(k=10)

    err_km.update(preds_t, targets_t)
    miss_rate_1.update(preds_t, targets_t)
    miss_rate_5.update(preds_t, targets_t)
    miss_rate_10.update(preds_t, targets_t)

    return {
        "err_km": err_km.compute().item(),
        "miss_rate_top1": miss_rate_1.compute().item(),
        "miss_rate_top5": miss_rate_5.compute().item(),
        "miss_rate_top10": miss_rate_10.compute().item(),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def get_primary_metrics(task_type: TaskType, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Table 4 primary metrics."""
    if task_type == TaskType.CLASSIFICATION:
        return {"acc": metrics.get("acc"), "f1": metrics.get("f1")}
    elif task_type == TaskType.REGRESSION:
        return {
            "mae": metrics.get("mae"),
            "pcc": metrics.get("pcc"),
            "kendalltau": metrics.get("kendalltau"),
        }
    elif task_type == TaskType.GEOLOCATION:
        return {
            "err_km": metrics.get("err_km"),
            "miss_rate_top1": metrics.get("miss_rate_top1"),
        }
    return {}


def save_report(
    results: Dict[str, Any],
    output_path: Path,
    dataset: str,
    format: str = "both",
) -> None:
    """Save verification report to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format in ["json", "both"]:
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON report: {json_path}")

    if format in ["md", "both"]:
        md_path = output_path.with_suffix(".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Direct Prompt Verification Report\n\n")
            f.write(f"**Dataset**: {dataset}\n\n")
            f.write(f"**Task Type**: {results['task_type']}\n\n")

            f.write("## Summary\n\n")
            f.write(f"- Valid samples: {results['valid_count']}\n")
            f.write(f"- Invalid/skipped samples: {results['invalid_count']}\n\n")

            f.write("## Table 4 Primary Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for k, v in results["primary_metrics"].items():
                if v is None:
                    f.write(f"| {k} | N/A |\n")
                elif isinstance(v, float):
                    f.write(f"| {k} | {v:.4f} |\n")
                else:
                    f.write(f"| {k} | {v} |\n")

            f.write("\n## All Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for k, v in results["metrics"].items():
                if v is None:
                    f.write(f"| {k} | N/A |\n")
                elif isinstance(v, float):
                    f.write(f"| {k} | {v:.4f} |\n")
                else:
                    f.write(f"| {k} | {v} |\n")
        print(f"Saved Markdown report: {md_path}")


def print_summary(results: Dict[str, Any], dataset: str) -> None:
    """Print summary to stdout."""
    print("\n" + "=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Task Type: {results['task_type']}")
    print(
        f"Valid samples: {results['valid_count']}, Invalid/skipped: {results['invalid_count']}"
    )
    print("=" * 60)
    print("\n--- Table 4 Primary Metrics ---")
    for k, v in results["primary_metrics"].items():
        if v is None:
            print(f"  {k}: N/A")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("\n--- All Metrics ---")
    for k, v in results["metrics"].items():
        if v is None:
            print(f"  {k}: N/A")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI (PhoneRecognition style: add_args + main)
# ---------------------------------------------------------------------------


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add zero-shot prediction evaluation arguments to an argparse parser."""
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(TASK_SPECS.keys()),
        help="Dataset name (determines task type and metric configuration)",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run directory containing prediction*.jsonl files",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Glob pattern for prediction JSONL file(s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for report (default: run_dir/verification_report)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "md", "both"],
        default="both",
        help="Output format",
    )
    parser.add_argument(
        "--no-filter-test",
        action="store_true",
        help="Do not filter to test split only (use all records)",
    )


def main() -> None:
    """Main entry point for zero-shot prediction evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot predictions against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_args(parser)
    args = parser.parse_args()

    # Determine predictions pattern
    if args.predictions:
        pattern = args.predictions
    elif args.run_dir:
        pattern = str(Path(args.run_dir) / "prediction*.jsonl")
    else:
        parser.error("Either --run_dir or --predictions must be specified")

    spec = TASK_SPECS[args.dataset]

    # Load and filter records
    records = load_predictions(pattern)
    if not args.no_filter_test:
        records = filter_test_split(records)

    if not records:
        print("Error: No valid records found after filtering")
        return

    # Extract and compute metrics based on task type
    if spec.task_type == TaskType.CLASSIFICATION:
        preds, targets, valid, invalid = extract_classification(records, spec)
        if valid == 0:
            print("Error: No valid classification predictions found")
            return
        metrics = compute_classification_metrics(preds, targets, spec.num_classes)

    elif spec.task_type == TaskType.REGRESSION:
        raw_preds, int_preds, targets, valid, invalid = extract_regression(
            records, spec
        )
        if valid == 0:
            print("Error: No valid regression predictions found")
            return
        metrics = compute_regression_metrics(
            raw_preds, int_preds, targets, spec.num_classes
        )

    elif spec.task_type == TaskType.GEOLOCATION:
        preds_xyz, targets_latlon, valid, invalid = extract_geolocation(records)
        if valid == 0:
            print("Error: No valid geolocation predictions found")
            return
        metrics = compute_geolocation_metrics(preds_xyz, targets_latlon)

    # Build results
    primary_metrics = get_primary_metrics(spec.task_type, metrics)
    results = {
        "dataset": args.dataset,
        "task_type": spec.task_type.value,
        "valid_count": valid,
        "invalid_count": invalid,
        "primary_metrics": primary_metrics,
        "metrics": metrics,
    }

    # Print summary
    print_summary(results, args.dataset)

    # Save report
    if args.output:
        output_path = Path(args.output)
    elif args.run_dir:
        output_path = Path(args.run_dir) / "verification_report"
    else:
        # Default: same directory as predictions
        first_file = sorted(glob(pattern))[0]
        output_path = Path(first_file).parent / "verification_report"

    save_report(results, output_path, args.dataset, args.format)


if __name__ == "__main__":
    main()
