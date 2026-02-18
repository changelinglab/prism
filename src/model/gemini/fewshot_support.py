"""
Few-shot support builder for Gemini direct prompt inference.
"""

import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@dataclass
class FewShotSample:
    """Container for one few-shot support example."""

    audio_path: str
    answer: dict[str, Any]


class FewShotSupportBuilder:
    """
    Build reusable few-shot supports from train split only.

    The builder instantiates a datamodule from config, resolves its train dataset,
    scans targets once to create candidate pools, samples deterministic support
    indices by policy, and materializes only selected samples.
    """

    @staticmethod
    def build(
        support_data_cfg: Any,
        few_shot_cfg: dict[str, Any],
        seed: int,
        response_schema: Optional[dict[str, Any]],
    ) -> list[FewShotSample]:
        """Build shared few-shot examples."""
        if not support_data_cfg:
            raise ValueError("few_shot mode requires non-empty prompting.support_data_cfg")

        policy = str((few_shot_cfg or {}).get("policy", "")).strip()
        k = (few_shot_cfg or {}).get("k")
        if not policy:
            raise ValueError("few_shot mode requires prompt_config.few_shot.policy")

        datamodule = FewShotSupportBuilder._resolve_datamodule(support_data_cfg)
        if hasattr(datamodule, "prepare_data"):
            datamodule.prepare_data()
        if hasattr(datamodule, "setup"):
            datamodule.setup(stage="fit")

        train_ds = FewShotSupportBuilder._resolve_train_dataset(datamodule)
        scan = FewShotSupportBuilder._scan_train_targets_once(train_ds)
        rng = random.Random(int(seed))
        selected_indices = FewShotSupportBuilder._select_indices(
            policy=policy,
            k=k,
            all_indices=scan["all_indices"],
            class_to_indices=scan["class_to_indices"],
            rng=rng,
        )

        shots: list[FewShotSample] = []
        for idx in selected_indices:
            sample = train_ds[idx]
            if "audio_path" not in sample:
                raise ValueError(f"Train sample at idx={idx} does not contain audio_path")
            target = sample.get("target")
            answer = FewShotSupportBuilder._format_answer_from_schema(
                target=target,
                response_schema=response_schema,
            )
            shots.append(FewShotSample(audio_path=str(sample["audio_path"]), answer=answer))
        return shots

    @staticmethod
    def dumps_answer(answer: dict[str, Any]) -> str:
        """Serialize answer json for few-shot model turn."""
        return json.dumps(answer, ensure_ascii=False)

    @staticmethod
    def _resolve_datamodule(support_data_cfg: Any) -> Any:
        """
        Resolve support_data_cfg into a datamodule instance.

        Root-cause handling:
        - Hydra instantiate(recursive=True) may already instantiate nested support_data_cfg.
        - In that case, calling hydra.utils.instantiate() again raises InstantiationException.
        """
        if isinstance(support_data_cfg, (DictConfig, dict)):
            return FewShotSupportBuilder._instantiate_datamodule_from_cfg(support_data_cfg)

        # Already-instantiated datamodule path (recursive instantiate)
        if hasattr(support_data_cfg, "setup"):
            return support_data_cfg

        raise ValueError(
            "Unsupported prompting.support_data_cfg type. "
            "Expected DictConfig/dict config or an instantiated datamodule object."
        )

    @staticmethod
    def _instantiate_datamodule_from_cfg(
        support_data_cfg: DictConfig | dict[str, Any]
    ) -> Any:
        """Instantiate support datamodule from config object."""
        if "_target_" not in support_data_cfg:
            raise ValueError(
                "prompting.support_data_cfg config must include `_target_` for datamodule."
            )
        return hydra.utils.instantiate(support_data_cfg)

    @staticmethod
    def _resolve_train_dataset(datamodule: Any) -> Any:
        """Resolve train dataset by standard priority."""
        for attr in ("ds_train", "train_dataset", "train_ds"):
            if hasattr(datamodule, attr):
                ds = getattr(datamodule, attr)
                if ds is not None:
                    return ds
        raise ValueError(
            "Could not resolve train dataset from datamodule. "
            "Expected one of: ds_train, train_dataset, train_ds."
        )

    @staticmethod
    def _scan_train_targets_once(train_ds: Any) -> dict[str, Any]:
        """Scan train metadata once to build candidate index pools."""
        all_indices = list(range(len(train_ds)))
        scanners = [
            ("hf_label_fast_path", FewShotSupportBuilder._scan_hf_label_fast_path),
            ("metadata_fast_path", FewShotSupportBuilder._scan_metadata_fast_path),
            ("geolocation_fast_path", FewShotSupportBuilder._scan_geolocation_fast_path),
        ]

        for scanner_name, scanner_fn in scanners:
            try:
                class_to_indices = scanner_fn(train_ds)
            except Exception as e:
                # Fall back to safer strategies when dataset-specific fast paths fail.
                log.warning(
                    "few-shot scanner '%s' failed, falling back: %s: %s",
                    scanner_name,
                    type(e).__name__,
                    e,
                )
                continue

            if class_to_indices is not None:
                return {"all_indices": all_indices, "class_to_indices": class_to_indices}

        class_to_indices = FewShotSupportBuilder._scan_targets_fallback(
            train_ds=train_ds, all_indices=all_indices
        )
        return {"all_indices": all_indices, "class_to_indices": class_to_indices}

    @staticmethod
    def _scan_hf_label_fast_path(train_ds: Any) -> Optional[dict[Any, list[int]]]:
        """Scan class targets from hf_ds + label_to_ids datasets."""
        if not (hasattr(train_ds, "hf_ds") and hasattr(train_ds, "label_to_ids")):
            return None

        class_to_indices: dict[Any, list[int]] = defaultdict(list)
        hf_ds = getattr(train_ds, "hf_ds")
        label_to_ids = getattr(train_ds, "label_to_ids")
        for idx, row in enumerate(hf_ds):
            target = label_to_ids[row["l1_label"]]
            class_to_indices[target].append(idx)
        return class_to_indices

    @staticmethod
    def _scan_metadata_fast_path(train_ds: Any) -> Optional[dict[Any, list[int]]]:
        """Scan class targets from in-memory metadata + target_key datasets."""
        if not (hasattr(train_ds, "metadata") and hasattr(train_ds, "target_key")):
            return None

        class_to_indices: dict[Any, list[int]] = defaultdict(list)
        metadata = getattr(train_ds, "metadata")
        target_key = getattr(train_ds, "target_key")
        for idx, (_, row) in enumerate(metadata.iterrows()):
            target = row[target_key]
            class_to_indices[FewShotSupportBuilder._to_class_key(target)].append(idx)
        return class_to_indices

    @staticmethod
    def _scan_geolocation_fast_path(train_ds: Any) -> Optional[dict[Any, list[int]]]:
        """Scan vaani-like geolocation datasets via raw rows only."""
        if not hasattr(train_ds, "ds"):
            return None

        raw_ds = getattr(train_ds, "ds")
        can_use_geo_fast_path = (
            hasattr(raw_ds, "column_names")
            and len(raw_ds) > 0
            and "latitude" in raw_ds.column_names
            and "longitude" in raw_ds.column_names
        )
        if not can_use_geo_fast_path:
            return None

        class_to_indices: dict[Any, list[int]] = defaultdict(list)
        for idx in range(len(raw_ds)):
            class_to_indices["__all__"].append(idx)
        return class_to_indices

    @staticmethod
    def _scan_targets_fallback(
        train_ds: Any, all_indices: list[int]
    ) -> dict[Any, list[int]]:
        """Fallback scan via __getitem__ across all train indices."""
        class_to_indices: dict[Any, list[int]] = defaultdict(list)
        for idx in all_indices:
            sample = train_ds[idx]
            target = sample.get("target")
            try:
                class_key = FewShotSupportBuilder._to_class_key(target)
            except ValueError:
                class_key = "__all__"
            class_to_indices[class_key].append(idx)
        return class_to_indices

    @staticmethod
    def _select_indices(
        policy: str,
        k: Optional[int],
        all_indices: list[int],
        class_to_indices: dict[Any, list[int]],
        rng: random.Random,
    ) -> list[int]:
        """Sample support indices by policy with deterministic RNG."""
        if policy == "all_class_coverage":
            if set(class_to_indices.keys()) == {"__all__"}:
                raise ValueError(
                    "all_class_coverage requires class-like targets, but class metadata "
                    "could not be extracted from train dataset"
                )
            selected: list[int] = []
            for class_key in sorted(class_to_indices.keys(), key=str):
                candidates = class_to_indices[class_key]
                if not candidates:
                    continue
                selected.append(rng.choice(candidates))
            if not selected:
                raise ValueError("all_class_coverage found no class candidates in train split")
            return selected

        if policy == "k_class_coverage":
            if k is None:
                raise ValueError("k_class_coverage requires prompt_config.few_shot.k")
            class_keys = sorted(class_to_indices.keys(), key=str)
            if class_keys == ["__all__"]:
                raise ValueError(
                    "k_class_coverage requires class-like targets, but class metadata "
                    "could not be extracted from train dataset"
                )
            if int(k) > len(class_keys):
                raise ValueError(
                    f"k_class_coverage requested k={k}, but only {len(class_keys)} classes available"
                )
            chosen_keys = rng.sample(class_keys, int(k))
            selected = [rng.choice(class_to_indices[class_key]) for class_key in chosen_keys]
            return selected

        if policy == "k_sample":
            if k is None:
                raise ValueError("k_sample requires prompt_config.few_shot.k")
            if int(k) > len(all_indices):
                raise ValueError(
                    f"k_sample requested k={k}, but only {len(all_indices)} train samples available"
                )
            return rng.sample(all_indices, int(k))

        raise ValueError(f"Unsupported few-shot policy: {policy}")

    @staticmethod
    def _format_answer_from_schema(
        target: Any, response_schema: Optional[dict[str, Any]]
    ) -> dict[str, Any]:
        """Format target as assistant answer json using response schema."""
        if not response_schema:
            raise ValueError("few-shot requires client_config.response_schema")

        properties = response_schema.get("properties", {})
        required = response_schema.get("required", [])
        required_set = set(required)

        if required_set == {"class_id"} and "class_id" in properties:
            return {"class_id": int(FewShotSupportBuilder._to_scalar(target))}

        if required_set == {"score"} and "score" in properties:
            score_type = str(properties["score"].get("type", "NUMBER")).upper()
            scalar = FewShotSupportBuilder._to_scalar(target)
            if score_type == "INTEGER":
                return {"score": int(scalar)}
            return {"score": float(scalar)}

        if required_set == {"lat", "lon"} and "lat" in properties and "lon" in properties:
            lat, lon = FewShotSupportBuilder._to_pair(target)
            return {
                "lat": FewShotSupportBuilder._radian_to_degree(float(lat)),
                "lon": FewShotSupportBuilder._radian_to_degree(float(lon)),
            }

        raise ValueError(
            "Unsupported response schema for few-shot formatting. "
            f"required={sorted(required_set)}"
        )

    @staticmethod
    def _to_scalar(value: Any) -> Any:
        """Convert scalar-like wrapper values to plain python scalars."""
        if hasattr(value, "item"):
            return value.item()
        return value

    @staticmethod
    def _to_pair(value: Any) -> tuple[float, float]:
        """Convert pair-like value to (x, y)."""
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("Expected target with length 2 for geolocation response schema")
        a = value[0].item() if hasattr(value[0], "item") else value[0]
        b = value[1].item() if hasattr(value[1], "item") else value[1]
        return float(a), float(b)

    @staticmethod
    def _to_class_key(value: Any) -> bool | int | float | str:
        """Convert class target to stable hashable key."""
        scalar = FewShotSupportBuilder._to_scalar(value)
        if isinstance(scalar, (bool, int, float, str)):
            return scalar
        raise ValueError(f"Unsupported class target type: {type(scalar).__name__}")

    @staticmethod
    def _radian_to_degree(value: float) -> float:
        """Convert radians to decimal degrees."""
        return math.degrees(value)
