from typing import Optional, Sequence

from phonological_posteriogram import PhoneModel

from src.model.phonvec.phonvec_inference import PhonVecInference
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def build_phonvec_inference(
    hf_repo: str,
    device: str = "cpu",
    *,
    lang: Optional[str] = None,
    phoible_id: Optional[int] = None,
    phoneme: bool = False,
    vocab: Optional[Sequence[str]] = None,
    dedup: bool = False,
):
    """Build a PhoneModel inference module.

    Args:
        hf_repo: HuggingFace repo id (or local path) for ``PhoneModel.from_pretrained``.
        device: torch device for the encoder forward.
        lang / phoible_id / phoneme / vocab: optional vocab constraints
            forwarded to ``PhoneModel.recognize``.
        dedup: merge consecutive segments sharing a label (default False).

    Returns:
        PhoneModel inference module.
    """
    model = PhoneModel.from_pretrained(hf_repo, device=device)
    log.info(f"PhoneModel loaded from {hf_repo}")
    return PhonVecInference(
        model,
        lang=lang,
        phoible_id=phoible_id,
        phoneme=phoneme,
        vocab=vocab,
        dedup=dedup,
    )
