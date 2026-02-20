"""Evaluate phone recognition output using panphon feature-based metrics.

The --prediction_file argument accepts a JSON file, a JSONL file, or a directory
containing JSONL files (produced by the distributed inference pipeline). When a
directory is provided, all *.jsonl files inside it are merged automatically,
removing the need for the intermediate jsonl2json conversion step.

Usage:
    # Load from a directory of JSONL files (no jsonl2json step needed)
    python -m src.metrics.phone_recognition \
        --prediction_file exp/runs/inf_doreco_powsm_ctc/8jobARR/ \
        --evaluation_name powsmctc \
        --output_file exp/runs/inf_doreco_powsm_ctc/8jobARR/inventory_results.csv \
        --gt_field target \
        --key_field utt_id \
        --language_field lang_sym

    # Load from a single JSON file
    python -m src.metrics.phone_recognition \
        --prediction_file exp/runs/inf_doreco_xlsr53/20251220_085642/transcription.withlang.json \
        --gt_field target \
        --key_field utt_id \
        --language_field lang_sym

    # Load from a single JSONL file
    python -m src.metrics.phone_recognition \
        --prediction_file exp/runs/inf_doreco_powsm_ctc/8jobARR/transcription.0.jsonl \
        --gt_field target \
        --key_field utt_id

    python -m src.metrics.phone_recognition \
        --prediction_file exp/runs/inf_doreco_lv60/20251220_085643/transcription.withlang.json \
        --gt_field target \
        --key_field utt_id \
        --language_field lang_sym \
        --noisy_pr # for noisy phone recognition
"""

import argparse
import string
import json
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Union
from tqdm import tqdm
import unicodedata
from collections import Counter
from itertools import chain, combinations

import panphon
import panphon.distance
from phone_inventory_metric import get_metrics as get_inventory_metrics
from phone_inventory_metric.common import setkeydict
from rich.console import Console
from rich.table import Table

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class PhoneRecognitionSummary:
    """Aggregate metrics for a phone recognition experiment."""

    PFER: float
    FER: float
    FED: float
    PER: float
    SUB: float
    INS: float
    DEL: float
    N: int  # number of utterances
    phones: int  # total number of reference phones
    inventory: setkeydict[float]


class PhoneRecognitionEvaluator:
    """
    Evaluates phone recognition output using panphon feature-based metrics.

      * PER (phone error rate, %)
      * FER (feature error rate, %)
      * FED (total feature edit distance)
      * PFER (phone feature error rate averaged per utterance)
      * per-utterance metrics

    Assumes `test_data` is a dict:
        { utt_id: {"prediction": str, "transcription": str, ...}, ... }
    """

    def __init__(self, normalize_ipa: bool = True):
        self.normalize_ipa = normalize_ipa
        self.dst = panphon.distance.Distance()

    @staticmethod
    def clean_text(s: str) -> str:
        """Normalize IPA text: remove spaces/punct, NFC->NFD, fix 'g'→'ɡ'."""
        s = s.replace(" ", "").translate(str.maketrans("", "", string.punctuation))
        s = unicodedata.normalize("NFD", s)
        return s.replace("g", "ɡ").strip()

    def _prepare(self, text: str) -> str:
        return self.clean_text(text) if self.normalize_ipa else text

    def _compute_sid_metrics(self, hyp: str, ref: str) -> Tuple[int, int, int]:
        """Calculates substitution, insertion, deletion rates on phones."""
        sub_errors = ins_errors = del_errors = 0
        # dp
        Hlen = len(hyp) + 1
        Rlen = len(ref) + 1
        D = [[0] * Rlen for _ in range(Hlen)]
        for hi in range(Hlen):
            D[hi][0] = hi
        for rj in range(Rlen):
            D[0][rj] = rj
        for hi in range(1, Hlen):
            for rj in range(1, Rlen):
                cost = 0 if hyp[hi - 1] == ref[rj - 1] else 1
                D[hi][rj] = min(
                    D[hi - 1][rj] + 1,
                    D[hi][rj - 1] + 1,
                    D[hi - 1][rj - 1] + cost,
                )
        # backtrack
        hi = Hlen - 1
        rj = Rlen - 1
        while hi > 0 or rj > 0:
            if (
                hi > 0
                and rj > 0
                and D[hi][rj] == D[hi - 1][rj - 1]
                and hyp[hi - 1] == ref[rj - 1]
            ):
                hi -= 1
                rj -= 1
            elif hi > 0 and rj > 0 and D[hi][rj] == D[hi - 1][rj - 1] + 1:
                sub_errors += 1
                hi -= 1
                rj -= 1
            elif rj > 0 and D[hi][rj] == D[hi][rj - 1] + 1:
                del_errors += 1
                rj -= 1
            else:
                ins_errors += 1
                hi -= 1
        return sub_errors, ins_errors, del_errors

    def _compute_utterance_metrics(
        self, hyp: str, ref: str
    ) -> Tuple[Dict[str, Union[int, float]]]:
        """
        Compute metrics for a single utterance.

        Returns:
            (metrics_dict, pfer, fed, per_errors, n_phones)
        """
        hyp = self._prepare(hyp)
        ref = self._prepare(ref)

        # Phone feature distances
        pfer = self.dst.hamming_feature_edit_distance(hyp, ref)
        fed = self.dst.feature_edit_distance(hyp, ref)

        # PER via min_edit_distance over IPA segments
        hyp_segs = self.dst.fm.ipa_segs(hyp)
        ref_segs = self.dst.fm.ipa_segs(ref)
        n_phones = len(ref_segs)

        per_errors = self.dst.min_edit_distance(
            lambda v: 1,  # deletion cost
            lambda v: 1,  # insertion cost
            lambda x, y: 0 if x == y else 1,  # substitution cost
            [[]],  # start
            hyp_segs,
            ref_segs,
        )
        sub_errors, ins_errors, del_errors = self._compute_sid_metrics(
            hyp_segs, ref_segs
        )

        metrics = {
            "pfer": float(pfer),
            "fed": float(fed),
            "per": float(per_errors / n_phones * 100) if n_phones > 0 else 0.0,
            "fer": float(fed / n_phones * 100) if n_phones > 0 else 0.0,
        }
        out = {
            "metrics": metrics,
            "pfer": pfer,
            "fed": fed,
            "per_errors": per_errors,
            "sub_errors": sub_errors,
            "ins_errors": ins_errors,
            "del_errors": del_errors,
            "n_phones": n_phones,
        }
        return out

    @classmethod
    def _get_phone_inventory_metrics(
        cls, test_data: dict[str, dict[str, Any]]
    ) -> setkeydict[float]:
        """
        Compute the phone inventory metrics on the dataset.

        The results are computed against a combination of different boolean options:
        - `featured`: if True, use a fuzzy notion of set membership based on
          phonetic feature similarity (provided by Panphon).
        - `exclusive`: if True, then phones in each inventory may match at most
          one other phone; if False, any phone matches its nearest neighbor in
          the other set (this only makes a difference for when `featured` is
          true.
        - `max`: if True, compute the optimal cutoff for the reference set in
          terms of F1-score (i.e., iteratively remove the least frequent phones
          from the predicted inventory).

        Returns
        -------
        setkeydict:
            A dict where the keys are tuples of strings that do not care about
            order.  Indexing with brackets works, but some methods (e.g.,
            `.get()`) may not work properly.


        """

        def get_inventory(key: str) -> list[str]:
            c = Counter()
            ft = panphon.FeatureTable()
            for _, sample in test_data.items():
                datum = sample.get(key, "")
                c.update(ft.ipa_segs(datum))
            # This will return phones in order of descending frequency.  For
            # the reference set, this is not taken into account, but for the
            # predictions, it used to calculate an upper bound onf the
            # F1-score.
            return [x[0] for x in c.most_common()]

        pred_inventory = get_inventory("prediction")
        ref_inventory = get_inventory("transcription")
        return get_inventory_metrics(ref_inventory, pred_inventory, search_max=True)

    def evaluate(
        self, test_data: Dict[str, Dict[str, Any]]
    ) -> Tuple[PhoneRecognitionSummary, Dict[str, Dict[str, float]]]:
        """
        Evaluate a full dataset.

        Args:
            test_data: mapping from utt_id -> {"prediction": ..., "transcription": ...}

        Returns:
            summary: PhoneRecognitionSummary (aggregate metrics)
            instance_metrics: per-utterance metrics, same keys as original script:
                             {utt_id: {"pfer":..., "fed":..., "per":..., "fer":...}}
        """
        if not test_data:
            empty_summary = PhoneRecognitionSummary(
                PFER=0.0,
                FER=0.0,
                FED=0.0,
                PER=0.0,
                N=0,
                phones=0,
                SUB=0.0,
                INS=0.0,
                DEL=0.0,
            )
            return empty_summary, {}

        instance_metrics: Dict[str, Dict[str, float]] = {}

        pfer_sum = 0.0
        fed_sum = 0.0
        per_err_sum = 0.0
        phones_sum = 0
        n_utts = 0
        sub_err_sum = 0
        ins_err_sum = 0
        del_err_sum = 0

        for utt_id, sample in tqdm(
            test_data.items(), total=len(test_data), desc="Evaluating", leave=False
        ):
            hyp = sample.get("prediction", "")
            ref = sample.get("transcription", "")

            out = self._compute_utterance_metrics(hyp, ref)

            instance_metrics[utt_id] = out["metrics"]
            pfer_sum += out["pfer"]
            fed_sum += out["fed"]
            per_err_sum += out["per_errors"]
            phones_sum += out["n_phones"]
            sub_err_sum += out["sub_errors"]
            ins_err_sum += out["ins_errors"]
            del_err_sum += out["del_errors"]
            n_utts += 1

        summary = PhoneRecognitionSummary(
            PFER=pfer_sum / n_utts if n_utts > 0 else 0.0,
            FER=(fed_sum / phones_sum * 100) if phones_sum > 0 else 0.0,
            FED=fed_sum,
            PER=(per_err_sum / phones_sum * 100) if phones_sum > 0 else 0.0,
            SUB=(sub_err_sum / phones_sum * 100) if phones_sum > 0 else 0.0,
            INS=(ins_err_sum / phones_sum * 100) if phones_sum > 0 else 0.0,
            DEL=(del_err_sum / phones_sum * 100) if phones_sum > 0 else 0.0,
            N=n_utts,
            phones=phones_sum,
            inventory=self._get_phone_inventory_metrics(test_data),
        )

        return summary, instance_metrics

    @staticmethod
    def pretty_print(summary: PhoneRecognitionSummary, **_kwargs: Any) -> None:
        """Rich summary table."""
        t = Table(title="Phone Recognition Results")
        t.add_column("Metric")
        t.add_column("Value", justify="right")
        t.add_row("Utterances (N)", f"{summary.N}")
        t.add_row("Total Phones", f"{summary.phones}")
        t.add_row("PFER (avg per utt)", f"{summary.PFER:.4f}")
        t.add_row("FER (%)", f"{summary.FER:.2f}")
        t.add_row("FED (total)", f"{summary.FED:.2f}")
        t.add_row("PER (%)", f"{summary.PER:.2f}")
        t.add_row("SUB (%)", f"{summary.SUB:.2f}")
        t.add_row("INS (%)", f"{summary.INS:.2f}")
        t.add_row("DEL (%)", f"{summary.DEL:.2f}")
        Console().print(t)

        PhoneRecognitionEvaluator.pretty_print_inventory_metrics(summary.inventory)

    @staticmethod
    def pretty_print_inventory_metrics(inventory_metrics: setkeydict[float]) -> None:
        t = Table(title="Phone Inventory Metrics")
        t.add_column("Exclusive\nMatch", justify="center")
        t.add_column("Featured", justify="center")
        t.add_column("Upper\nBound", justify="center")
        t.add_column("F1", justify="center")
        t.add_column("Precision", justify="center")
        t.add_column("Recall", justify="center")

        # powerset
        base_key_elements = ["exclusive", "max", "featured"]
        base_keys = chain.from_iterable(
            combinations(base_key_elements, n)
            for n in range(len(base_key_elements) + 1)
        )

        for base_key in base_keys:
            if "featured" not in base_key and "exclusive" in base_key:
                continue
            f1 = inventory_metrics[base_key + ("f1_score",)]
            precision = inventory_metrics[base_key + ("precision",)]
            recall = inventory_metrics[base_key + ("recall",)]
            t.add_row(
                # If we are not using features, the matches are implicitly exclusive.
                "x" if ("exclusive" in base_key or "featured" not in base_key) else "",
                "x" if "featured" in base_key else "",
                "x" if "max" in base_key else "",
                f"{f1:.3f}",
                f"{precision:.3f}",
                f"{recall:.3f}",
            )

        Console().print(t)

    def write_to_csv(
        self,
        summary: PhoneRecognitionSummary,
        evalname: str,
        output_file: str,
        language: str,
    ) -> None:
        """Append summary metrics to a CSV file."""
        import csv
        import os

        base_key_elements = ["exclusive", "max", "featured"]
        base_keys = chain.from_iterable(
            combinations(base_key_elements, n)
            for n in range(len(base_key_elements) + 1)
        )
        inv_headers = []
        inv_values = []
        for base_key in base_keys:
            if "featured" not in base_key and "exclusive" in base_key:
                continue
            label = "none" if not base_key else "_".join(base_key)
            prefix = f"inv_{label}"
            inv_headers.extend(
                [f"{prefix}_f1", f"{prefix}_precision", f"{prefix}_recall"]
            )
            inv_values.extend(
                [
                    f"{summary.inventory[base_key + ('f1_score',)]:.3f}",
                    f"{summary.inventory[base_key + ('precision',)]:.3f}",
                    f"{summary.inventory[base_key + ('recall',)]:.3f}",
                ]
            )

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        write_header = (
            not os.path.exists(output_file) or os.path.getsize(output_file) == 0
        )
        with open(output_file, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(
                    [
                        "eval_name",
                        "language",
                        "N",
                        "Total Phones",
                        "PFER",
                        "FER (%)",
                        "FED",
                        "PER (%)",
                        "SUB (%)",
                        "INS (%)",
                        "DEL (%)",
                    ]
                    + inv_headers
                )
            writer.writerow(
                [
                    evalname,
                    language,
                    summary.N,
                    summary.phones,
                    f"{summary.PFER:.4f}",
                    f"{summary.FER:.2f}",
                    f"{summary.FED:.2f}",
                    f"{summary.PER:.2f}",
                    f"{summary.SUB:.2f}",
                    f"{summary.INS:.2f}",
                    f"{summary.DEL:.2f}",
                ]
                + inv_values
            )


def _load_predictions_raw(pred_file: str) -> Dict[str, Any]:
    """Load prediction data from a JSON file, a JSONL file, or a directory of JSONL files.

    Supports three input formats:
      - A `.json` file: loaded directly as a single JSON object.
      - A `.jsonl` file: each line is a JSON object whose keys are merged.
      - A directory: all `*.jsonl` files in the directory are read and merged.

    Returns a flat dict: {key: {"pred": ..., "passthrough": ...}, ...}
    """
    from pathlib import Path

    path = Path(pred_file)

    if path.is_dir():
        jsonl_files = sorted(path.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in directory: {pred_file}")
        data: Dict[str, Any] = {}
        for p in jsonl_files:
            for line in p.open():
                if line.strip():
                    data.update(json.loads(line))
        return data

    if path.suffix == ".jsonl":
        data = {}
        with open(pred_file, "r") as f:
            for line in f:
                if line.strip():
                    data.update(json.loads(line))
        return data

    # Default: treat as a single JSON file
    with open(pred_file, "r") as f:
        return json.load(f)


def _load_predictions(
    pred_file: str, language_field: str = None
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Load predictions from a JSON file, JSONL file, or directory of JSONL files.

    The returned structure is:
    {'language': { utt_id: {"prediction": str, "transcription": str}, ... }}
    If language_field is None, 'language' is set to the string '"combined"'.
    """
    data = _load_predictions_raw(pred_file)

    all_languages = set()
    if language_field is not None:
        assert (
            language_field in next(iter(data.values()))["passthrough"]
        ), f"Language field '{language_field}' not found in prediction file."
        all_languages = {item["passthrough"][language_field] for item in data.values()}
    else:
        all_languages = {"combined"}

    all_languages = sorted(all_languages)
    print(f"Found {len(all_languages)} languages: {all_languages}")
    return_data = {}
    for lang in tqdm(all_languages, desc="Loading predictions"):
        D = {
            item["passthrough"][args.key_field]: {
                "prediction": item["pred"][0][args.pred_field],
                "transcription": (
                    item["passthrough"][args.gt_field]
                    if not args.noisy_pr
                    else "".join(
                        [
                            n
                            for n in item["passthrough"]["masked_phones"]
                            if n != "[NOISE]"
                        ]
                    )
                ),
            }
            for _, item in data.items()
            if item["passthrough"].get(language_field, "combined") == lang
        }
        return_data[lang] = D
    return return_data


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add phone recognition evaluation arguments to an argparse parser."""
    parser.add_argument(
        "--prediction_file",
        required=True,
        help="Path to prediction JSON file, JSONL file, or directory of JSONL files",
    )
    parser.add_argument(
        "--gt_field",
        type=str,
        default="masked_phones",
        help="Field name for ground truth transcription in the prediction file",
    )
    parser.add_argument(
        "--pred_field",
        type=str,
        default="processed_transcript",
        help="Field name for predicted transcription in the prediction file",
    )
    parser.add_argument(
        "--key_field",
        type=str,
        default="utt_id",
        help="Field name for utterance ID in the prediction file",
    )
    parser.add_argument(
        "--noisy_pr",
        action="store_true",
        help="Whether to evaluate noisy phone recognition",
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="File to write results to"
    )
    parser.add_argument(
        "--language_field",
        type=str,
        default=None,
        help="If provided, language field must exist in the prediction file and "
        "will be used to produce per language metrics.",
    )
    parser.add_argument("--evaluation_name", type=str, help="name for the evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    loaded_predictions = _load_predictions(args.prediction_file, args.language_field)
    print(
        f"Loaded predictions for {len(loaded_predictions)} languages containing {sum(len(v) for v in loaded_predictions.values())} utterances."
    )
    inventories = []
    langs_used = list(loaded_predictions.keys())
    for lang, preds in tqdm(loaded_predictions.items(), desc="Evaluating languages"):
        evaluator = PhoneRecognitionEvaluator(normalize_ipa=True)
        summary, _ = evaluator.evaluate(preds)
        inventories.append(summary.inventory)
        if args.output_file:
            assert args.evaluation_name is not None, "Please provide --evaluation_name"
            write_file = args.output_file
            evaluator.write_to_csv(summary, args.evaluation_name, write_file, lang)
            print(f"Appended results to {write_file}")

    all_keys = set().union(*[inv.keys() for inv in inventories])
    macro_inv_dict = {}
    for k in all_keys:
        vals = [inv[k] for inv in inventories]
        macro_inv_dict[k] = sum(vals) / len(vals)
    macro_inventory = setkeydict(list(macro_inv_dict.items()))
    Console().print(
        f"\nMacro-averaged Phone Inventory Metrics over {len(langs_used)} languages:"
    )
    PhoneRecognitionEvaluator.pretty_print_inventory_metrics(macro_inventory)
