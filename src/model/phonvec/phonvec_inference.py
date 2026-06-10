from typing import List, Optional, Sequence

import numpy as np
import torch
from phonological_posteriogram import PhoneModel


class PhonVecInference:
    """Transcribe a single utterance with a PhoneModel.

    ``PhoneModel.recognize`` runs encode -> segment -> recognize end to end and
    returns per-segment IPA labels; we concatenate them into a phone string.
    """

    def __init__(
        self,
        model: PhoneModel,
        *,
        lang: Optional[str] = None,
        phoible_id: Optional[int] = None,
        phoneme: bool = False,
        vocab: Optional[Sequence[str]] = None,
        dedup: bool = False,
    ):
        self.model = model
        self.lang = lang
        self.phoible_id = phoible_id
        self.phoneme = phoneme
        self.vocab = vocab
        self.dedup = dedup

    @torch.no_grad()
    def __call__(self, speech, *args, **kwargs) -> List[dict]:
        """Args:
            speech: (Length,) raw waveform at the model's sample rate.
        Returns:
            Single-element list with the predicted phone transcript.
        """
        if isinstance(speech, torch.Tensor):
            speech = speech.cpu().numpy()
        waveform = np.asarray(speech, dtype=np.float32)
        units = self.model.recognize(
            waveform,
            lang=self.lang,
            phoible_id=self.phoible_id,
            phoneme=self.phoneme,
            vocab=self.vocab,
            dedup=self.dedup,
        )
        labels = [str(u.label) for u in units if u.label is not None]
        return [
            {
                "processed_transcript": "".join(labels),
                "predicted_transcript": " ".join(labels),
            }
        ]
