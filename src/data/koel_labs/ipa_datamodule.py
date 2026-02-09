"""
Koel Labs LLC has released cleaned Huggingface Datasets for various IPA transcription corpora with a consistent format.
Most require gated read access which can be requested here: https://huggingface.co/collections/KoelLabs/processed-datasets.
The format is documented here: https://github.com/KoelLabs/ML/blob/main/scripts/data_loaders/huggingface.py

Usage:
    python -m src.data.koel_labs.ipa_datamodule --hf_repo KoelLabs/SpeechOcean --cache_dir exp/cache/kl_SpeechOcean
"""

import io
import argparse
from pathlib import Path
from typing import Optional, List, Dict
import torch
import torchaudio
import lightning as L
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset, DatasetDict, Audio

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------


class KoelIPADataset(Dataset):
    """
    Dataset wrapping a Koel Labs dataset with:

        audio: Audio() column
        ipa:   string IPA transcription  <-- TARGET
    """

    def __init__(
        self,
        hf_split,
        tokenizer,
        split: str,
        target_sr: int = 16000,
        max_duration_sec: Optional[float] = None,
    ):
        self.ds = hf_split
        self.tokenizer = tokenizer
        self.split = split
        self.target_sr = target_sr
        self.max_duration_sec = max_duration_sec

        log.info(f"KoelIPADataset split={split}: {len(self.ds)} examples")

    # --------------------------------------------------------------

    def __len__(self):
        return len(self.ds)

    # --------------------------------------------------------------

    def _resample(self, wav, sr):
        if sr == self.target_sr:
            return wav
        return torchaudio.functional.resample(wav, sr, self.target_sr)

    # --------------------------------------------------------------

    def __getitem__(self, idx):
        item = self.ds[idx]

        audio = item["audio"]
        if "array" in audio:
            wav = torch.tensor(audio["array"]).float()
            sr = audio["sampling_rate"]
        else:
            wav, sr = torchaudio.load(io.BytesIO(audio["bytes"]))

        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        wav = self._resample(wav, sr)
        sr = self.target_sr

        if self.max_duration_sec:
            max_len = int(sr * self.max_duration_sec)
            wav = wav[:, :max_len]

        wav = wav.mean(0)  # mono (T,)

        # -------------------------
        # IPA target
        # -------------------------
        ipa = item.get("ipa", "")
        phones = ipa.split() if " " in ipa else list(ipa)
        phone_ids = self.tokenizer.tokens2ids(phones)

        utt_id = item.get("id", f"{self.split}_{idx}")

        return {
            "utt_id": utt_id,
            "split": self.split,
            "speech": wav,
            "speech_length": wav.shape[-1],
            "text": ipa,
            "phones": ipa,
            "phone_id": torch.tensor(phone_ids, dtype=torch.long),
            "phone_length": len(phone_ids),
            "target": ipa,
        }


# ------------------------------------------------------------------
# Collate (mirrors UASpeech style)
# ------------------------------------------------------------------


def collate_fn(batch):
    B = len(batch)

    max_speech = max(x["speech_length"] for x in batch)
    max_phone = max(x["phone_length"] for x in batch)

    speech = torch.zeros(B, max_speech)
    speech_len = torch.zeros(B, dtype=torch.long)

    phone_id = torch.full((B, max_phone), -1, dtype=torch.long)
    phone_len = torch.zeros(B, dtype=torch.long)

    for i, x in enumerate(batch):
        speech[i, : x["speech_length"]] = x["speech"]
        speech_len[i] = x["speech_length"]

        if x["phone_length"] > 0:
            phone_id[i, : x["phone_length"]] = x["phone_id"]

        phone_len[i] = x["phone_length"]

    return {
        "utt_id": [x["utt_id"] for x in batch],
        "split": [x["split"] for x in batch],
        "speech": speech,
        "speech_length": speech_len,
        "phone_id": phone_id,
        "phone_length": phone_len,
        "target": [x["target"] for x in batch],
    }


# ------------------------------------------------------------------
# Lightning DataModule
# ------------------------------------------------------------------


class KoelIPADataModule(L.LightningDataModule):
    """
    Generic Koel Labs speech+IPA datamodule.

    Automatically detects available splits:
        train / validation / val / valid / test
    """

    def __init__(
        self,
        hf_repo: str,
        tokenizer,
        cache_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        target_sr: int = 16000,
        max_duration_sec: Optional[float] = None,
        predict_splits: Optional[List[str]] = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["tokenizer"])
        self.tokenizer = tokenizer

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.predict_splits = predict_splits or ["train", "validation", "test"]

        self.ds_train = None
        self.ds_val = None
        self.ds_test = None

    # --------------------------------------------------------------

    def prepare_data(self):
        # downloads only
        load_dataset(self.hparams.hf_repo, cache_dir=self.cache_dir) # type: ignore

    # --------------------------------------------------------------

    def _make_dataset(self, split_name, split_ds):
        return KoelIPADataset(
            split_ds,
            tokenizer=self.tokenizer,
            split=split_name,
            target_sr=self.hparams.target_sr,  # type: ignore
            max_duration_sec=self.hparams.max_duration_sec,  # type: ignore
        )

    # --------------------------------------------------------------

    def setup(self, stage: Optional[str] = None):
        ds_dict: DatasetDict = load_dataset(  # type: ignore
            self.hparams.hf_repo,  # type: ignore
            cache_dir=self.cache_dir,  # type: ignore
        )
        ds_dict = ds_dict.cast_column("audio", Audio(decode=False))

        splits: Dict[str, Dataset] = dict(ds_dict)  # type: ignore

        def pick(*names):
            for n in names:
                if n in splits:
                    return splits[n]
            return None

        train = pick("train")
        val = pick("validation", "val", "valid")
        test = pick("test")

        if train:
            self.ds_train = self._make_dataset("train", train)
        if val:
            self.ds_val = self._make_dataset("validation", val)
        if test:
            self.ds_test = self._make_dataset("test", test)

        log.info(
            f"Splits → train={len(train) if train else 0}, "  # type: ignore
            f"val={len(val) if val else 0}, "  # type: ignore
            f"test={len(test) if test else 0}"  # type: ignore
        )

    # --------------------------------------------------------------

    def _dl(self, ds, shuffle):
        if ds is None:
            return None
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,  # type: ignore
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,  # type: ignore
            pin_memory=self.hparams.pin_memory,  # type: ignore
            collate_fn=collate_fn,
        )

    # --------------------------------------------------------------

    def train_dataloader(self):
        return self._dl(self.ds_train, True)

    def val_dataloader(self):
        return self._dl(self.ds_val, False)

    def test_dataloader(self):
        return self._dl(self.ds_test, False)

    def predict_dataloader(self):
        datasets = []
        if "train" in self.predict_splits and self.ds_train:
            datasets.append(self.ds_train)
        if "validation" in self.predict_splits and self.ds_val:
            datasets.append(self.ds_val)
        if "test" in self.predict_splits and self.ds_test:
            datasets.append(self.ds_test)

        if not datasets:
            datasets = [self.ds_test]

        return self._dl(ConcatDataset(datasets), False)  # type: ignore


# ------------------------------------------------------------------
# CLI test
# ------------------------------------------------------------------


if __name__ == "__main__":
    from src.core.ipa_utils import IPATokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    args = parser.parse_args()

    dm = KoelIPADataModule(
        hf_repo=args.hf_repo,
        cache_dir=args.cache_dir,
        tokenizer=IPATokenizer(),
        batch_size=2,
    )

    dm.prepare_data()
    dm.setup()

    for batch in dm.train_dataloader():  # type: ignore
        print(batch)
        break
