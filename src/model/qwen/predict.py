"""
Qwen Direct Prompt Inference for PRiSM.

Uses the same distributed_inference contract and shared few-shot infrastructure
as Gemini DirectPromptInference. Returns a single prediction dict per sample
(classification, regression, geolocation). Audio is sent to vLLM as base64 data URIs.
"""

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import httpx
import soundfile as sf
from openai import OpenAI

from src.model.gemini.fewshot_support import FewShotSample, FewShotSupportBuilder


class QwenDirectPromptInference:
    """
    Inference wrapper for Qwen/vLLM direct-prompt tasks (classification, regression, geolocation).

    Implements the same contract as DirectPromptInference:
    - __call__(audio_path, **kwargs) -> dict | None
    - Uses FewShotSupportBuilder for few_shot mode (shared with Gemini).
    """

    def __init__(
        self,
        client_config: dict[str, Any],
        prompt_config: dict[str, Any],
        prompting: Optional[dict[str, Any]] = None,
        device: Optional[str] = None,
        cache_path: Optional[Union[str, Path]] = None,
        resume: bool = True,
        cache_key_field: str = "metadata_idx",
        error_log_path: Optional[Union[str, Path]] = None,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = client_config.get("base_url", "http://localhost:8000/v1")
        self.model_name = client_config.get(
            "model_name", "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        )
        self.api_key = client_config.get("api_key", "EMPTY")
        self.temperature = client_config.get("temperature", 0.0)
        self.max_tokens = client_config.get("max_tokens", 512)
        self.response_schema = client_config.get("response_schema")

        self.system_prompt = prompt_config.get("system_prompt", "")
        self.user_prompt = prompt_config.get("user_prompt", "")
        self.few_shot_cfg = prompt_config.get("few_shot", {})

        prompting = prompting or {}
        self.prompting_mode = str(prompting.get("mode", "zero_shot")).strip().lower()
        self.prompting_seed = int(prompting.get("seed", 42))
        self.support_data_cfg = prompting.get("support_data_cfg")
        self.shared_few_shots: Optional[list[FewShotSample]] = None

        self.cache_path = Path(cache_path) if cache_path else None
        self.error_log_path = Path(error_log_path) if error_log_path else None
        self.resume = resume
        self.cache_key_field = cache_key_field
        self._cache: dict[str, Any] = {}

        http_client = httpx.Client(timeout=timeout)
        self.client = OpenAI(
            base_url=self.base_url, api_key=self.api_key, http_client=http_client
        )

        if self.cache_path and self.resume:
            self._load_cache()

    def _load_cache(self) -> None:
        if self.cache_path is None or not self.cache_path.exists():
            return
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    k = rec.get("key")
                    pred = rec.get("pred")
                    if k is not None and pred is not None:
                        self._cache[str(k)] = pred
        except Exception:
            pass

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            try:
                f.flush()
                os.fsync(f.fileno())
            except OSError:
                pass

    def _get_cache_key(self, audio_path: Union[str, Path], kwargs: dict[str, Any]) -> str:
        if self.cache_key_field in kwargs:
            base_key = str(kwargs[self.cache_key_field])
        else:
            base_key = str(audio_path)

        signature = self.prompting_mode
        if self.prompting_mode == "few_shot":
            policy = str((self.few_shot_cfg or {}).get("policy", "")).strip().lower()
            k = (self.few_shot_cfg or {}).get("k")
            k_str = str(k) if k is not None else "none"
            signature = (
                f"{self.prompting_mode}"
                f"|policy={policy or 'none'}"
                f"|k={k_str}"
                f"|seed={self.prompting_seed}"
            )
        return f"{signature}::{base_key}"

    def _path_to_data_uri(self, path: Union[str, Path]) -> str:
        """Load audio from path and return data:audio/wav;base64,... URI."""
        audio, sr = sf.read(path)
        if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
            audio = audio.T
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:audio/wav;base64,{b64}"

    def _build_zero_shot_messages(self, audio_path: Union[str, Path]) -> list[dict[str, Any]]:
        data_uri = self._path_to_data_uri(audio_path)
        content = [
            {"type": "audio_url", "audio_url": {"url": data_uri}},
            {"type": "text", "text": self.user_prompt},
        ]
        return [{"role": "user", "content": content}]

    def _build_few_shot_messages(self, audio_path: Union[str, Path]) -> list[dict[str, Any]]:
        if self.shared_few_shots is None:
            raise RuntimeError("shared_few_shots is not initialized")

        messages: list[dict[str, Any]] = []
        for shot in self.shared_few_shots:
            shot_uri = self._path_to_data_uri(shot.audio_path)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": shot_uri}},
                    {"type": "text", "text": self.user_prompt},
                ],
            })
            messages.append({
                "role": "assistant",
                "content": FewShotSupportBuilder.dumps_answer(shot.answer),
            })
        query_uri = self._path_to_data_uri(audio_path)
        messages.append({
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": query_uri}},
                {"type": "text", "text": self.user_prompt},
            ],
        })
        return messages

    def _build_messages(self, audio_path: Union[str, Path]) -> list[dict[str, Any]]:
        if self.prompting_mode == "zero_shot":
            return self._build_zero_shot_messages(audio_path)
        if self.prompting_mode == "few_shot":
            return self._build_few_shot_messages(audio_path)
        raise ValueError(f"Unsupported prompting.mode: {self.prompting_mode}")

    def __call__(self, audio_path: Union[str, Path], **kwargs: Any) -> Any:
        cache_key = self._get_cache_key(audio_path, kwargs)

        if self.cache_path and self.resume and cache_key in self._cache:
            return self._cache[cache_key]

        if self.prompting_mode == "few_shot" and self.shared_few_shots is None:
            self.shared_few_shots = FewShotSupportBuilder.build(
                support_data_cfg=self.support_data_cfg,
                few_shot_cfg=self.few_shot_cfg,
                seed=self.prompting_seed,
                response_schema=self.response_schema,
            )

        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self._build_messages(audio_path))

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw_response = response.choices[0].message.content or ""
        except Exception as e:
            self._log_error(cache_key, audio_path, e)
            return None

        pred = self._parse_response(raw_response, cache_key, audio_path)
        if pred is not None and self.cache_path:
            self._cache[cache_key] = pred
            self._append_jsonl(self.cache_path, {"key": cache_key, "pred": pred})
        return pred

    def _parse_response(
        self, raw_response: str, cache_key: str, audio_path: Union[str, Path]
    ) -> Optional[dict[str, Any]]:
        try:
            parsed = json.loads(raw_response)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected dict, got {type(parsed).__name__}")
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            self._log_error(
                cache_key,
                audio_path,
                e,
                extra={"raw_response": raw_response[:500] if raw_response else ""},
            )
            return None

    def _log_error(
        self,
        cache_key: str,
        audio_path: Union[str, Path],
        error: Exception,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        if not self.error_log_path:
            return
        error_record = {
            "key": cache_key,
            "audio_path": str(audio_path),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "code": getattr(error, "code", None),
                "status": getattr(error, "status", None),
            },
        }
        if extra:
            error_record.update(extra)
        self._append_jsonl(self.error_log_path, error_record)
