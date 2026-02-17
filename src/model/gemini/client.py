"""
Gemini API client for multimodal processing.

This module provides a clean wrapper around the Gemini API for various tasks.
It handles file upload, inference, and cleanup with retry logic.
"""

import os
import random
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from google import genai
from google.genai import errors as genai_errors, types


@dataclass
class UploadedFile:
    """Wrapper for an uploaded file with its metadata."""

    file: Any  # The uploaded file object from Gemini
    original_path: Optional[Path] = None


class GeminiClient:
    """
    Gemini API client for multimodal content generation.

    This class handles low-level API communication including file upload,
    content generation, and resource cleanup. It is designed to be task-agnostic
    and can process any file type supported by Gemini (audio, image, video, PDF, etc.).
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        seed: int = 0,
        thinking_budget: int = 0,
        response_schema: Optional[dict] = None,
        retry_config: Optional[dict] = None,
    ) -> None:
        """
        Initialize the Gemini client.

        Args:
            model_name: Identifier of the Gemini model to use.
            api_key: API key for authentication. If None, falls back to
                     GEMINI_API_KEY environment variable.
            temperature: Sampling temperature for generation (default: 1.0).
            top_p: Top-p (nucleus) sampling parameter (default: 0.95).
            seed: Random seed for reproducibility (default: 0).
            thinking_budget: Thinking budget for reasoning models (default: 0).
                Set to 0 for non-thinking mode, higher values for more reasoning.
            response_schema: Optional schema for structured JSON output. If provided,
                enables JSON mode with the specified schema. Expected format:
                {
                    "type": "OBJECT",
                    "required": ["field_name"],
                    "properties": {
                        "field_name": {"type": "STRING"}
                    }
                }
            retry_config: Configuration for retry logic. Keys:
                - max_retries (int): Maximum retry attempts (default: 5)
                - initial_delay (float): Initial delay in seconds (default: 1.0)
                - backoff_factor (float): Multiplier for delay (default: 2.0)
        """
        # Resolve API key
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key is not configured. Please provide api_key or set "
                "the GEMINI_API_KEY environment variable."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.thinking_budget = thinking_budget
        self.response_schema = self._build_schema(response_schema) if response_schema else None
        self.client = genai.Client(api_key=key)

        # Retry configuration
        default_retry = {"max_retries": 5, "initial_delay": 1.0, "backoff_factor": 2.0}
        self.retry_config = {**default_retry, **(retry_config or {})}

    def _is_retryable_api_error(self, e: Exception) -> bool:
        """Return True if exception should be retried based on status code / type."""
        if isinstance(e, genai_errors.APIError):
            # Retry transient failures (quota/rate limiting and server errors)
            return e.code in {429, 500, 502, 503, 504}

        # Common transient/network-ish failures
        if isinstance(e, (TimeoutError, ConnectionError, OSError)):
            return True

        return False

    def _build_schema(self, schema_config: dict) -> types.Schema:
        """
        Build a Gemini Schema object from a dictionary configuration.

        Args:
            schema_config: Dictionary with schema definition.

        Returns:
            types.Schema object for structured output.
        """
        # Map string type names to Gemini Type enum
        type_mapping = {
            "STRING": types.Type.STRING,
            "NUMBER": types.Type.NUMBER,
            "INTEGER": types.Type.INTEGER,
            "BOOLEAN": types.Type.BOOLEAN,
            "ARRAY": types.Type.ARRAY,
            "OBJECT": types.Type.OBJECT,
        }

        schema_type = type_mapping.get(schema_config.get("type", "OBJECT").upper(), types.Type.OBJECT)

        # Build properties if present
        properties = None
        if "properties" in schema_config:
            properties = {}
            for prop_name, prop_config in schema_config["properties"].items():
                prop_type = type_mapping.get(prop_config.get("type", "STRING").upper(), types.Type.STRING)
                properties[prop_name] = types.Schema(type=prop_type)

        return types.Schema(
            type=schema_type,
            required=schema_config.get("required"),
            properties=properties,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        files: Optional[list[str | Path] | str | Path] = None,
        anonymize: bool = True,
    ) -> str:
        """
        Generate content from prompt and optional files.

        Args:
            prompt: User prompt for generation.
            system_prompt: Optional system instruction for the model.
            files: Optional file(s) to include. Accepts None (text-only),
                   single path (str/Path), or list of paths.
            anonymize: If True, anonymize filenames before upload.

        Returns:
            Model's response text.
        """
        messages = [
            {
                "role": "user",
                "text": prompt,
                "files": files,
            }
        ]
        return self.generate_from_messages(
            messages=messages,
            system_prompt=system_prompt,
            anonymize=anonymize,
        )

    def generate_from_messages(
        self,
        messages: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
        anonymize: bool = True,
    ) -> str:
        """
        Generate content from multi-turn messages.

        Args:
            messages: List of turn dictionaries with keys:
                - role: "user", "model"
                - text: turn text
                - files: optional file path or list of file paths (user turns only)
            system_prompt: Optional system instruction for the model.
            anonymize: If True, anonymize filenames before upload.

        Returns:
            Model's response text.
        """
        uploaded_files: list[UploadedFile] = []

        try:
            contents: list[types.Content] = []

            for turn in messages:
                role = str(turn.get("role", "user")).strip().lower()
                if role not in {"user", "model"}:
                    raise ValueError(
                        f"Invalid message role: {role}. Supported roles are user or model."
                    )

                parts: list[types.Part] = []
                text = str(turn.get("text", "")).strip()
                files = turn.get("files")

                if role == "user" and files is not None:
                    file_paths = [files] if isinstance(files, (str, Path)) else list(files)
                    for file_path in file_paths:
                        uploaded = self._upload_file(Path(file_path), anonymize=anonymize)
                        uploaded_files.append(uploaded)
                        parts.append(
                            types.Part(
                                file_data=types.FileData(
                                    file_uri=uploaded.file.uri,
                                    mime_type=uploaded.file.mime_type,
                                )
                            )
                        )

                if text:
                    parts.append(types.Part(text=text))
                if not parts:
                    raise ValueError("Each message must include text or at least one file")

                contents.append(types.Content(role=role, parts=parts))

            return self._generate_content_from_contents(
                contents=contents,
                system_prompt=system_prompt,
            )
        finally:
            for uploaded in uploaded_files:
                self._delete_file(uploaded)

    def _upload_file(self, path: Path, anonymize: bool = True) -> UploadedFile:
        """
        Upload a file to Gemini API.

        Args:
            path: Path to the file to upload.
            anonymize: If True, copy file to temp location with random name.

        Returns:
            UploadedFile object containing the uploaded file reference.

        Raises:
            RuntimeError: If upload fails after max retries.
        """
        temp_file: Optional[Path] = None
        upload_path = path

        # Anonymize filename to prevent information leakage
        if anonymize:
            suffix = path.suffix or ".bin"
            random_name = f"{uuid.uuid4().hex}{suffix}"
            temp_dir = tempfile.gettempdir()
            temp_file = Path(temp_dir) / random_name
            shutil.copy2(path, temp_file)
            upload_path = temp_file

        uploaded = None
        last_error: Optional[Exception] = None
        delay = self.retry_config["initial_delay"]

        for attempt in range(self.retry_config["max_retries"]):
            try:
                uploaded = self.client.files.upload(file=upload_path)
                break
            except Exception as e:
                last_error = e
                if attempt < self.retry_config["max_retries"] - 1:
                    time.sleep(delay)
                    delay *= self.retry_config["backoff_factor"]

        # Clean up temp file after upload attempt
        if temp_file and temp_file.exists():
            temp_file.unlink()

        if uploaded is None:
            raise RuntimeError(
                f"Failed to upload file after {self.retry_config['max_retries']} attempts"
            ) from last_error

        return UploadedFile(file=uploaded, original_path=path)

    def _delete_file(self, uploaded_file: UploadedFile) -> None:
        """
        Delete an uploaded file from Gemini.

        Args:
            uploaded_file: The UploadedFile object to delete.
        """
        try:
            self.client.files.delete(name=uploaded_file.file.name)
        except Exception:
            # Silently ignore deletion errors
            pass

    def _generate_content_from_contents(
        self, contents: list[types.Content], system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate content using Gemini API.

        Args:
            contents: Multi-turn contents payload.
            system_prompt: Optional system instruction.

        Returns:
            Model's response text.
        """
        # Build generation config
        config_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "candidate_count": 1,
            "response_modalities": ["TEXT"],
            "thinking_config": types.ThinkingConfig(thinking_budget=self.thinking_budget),
        }

        # Add structured output schema if configured
        if self.response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = self.response_schema

        # Add system instruction if provided
        if system_prompt:
            config_kwargs["system_instruction"] = types.Content(
                role="system", parts=[types.Part(text=system_prompt)]
            )

        config = types.GenerateContentConfig(**config_kwargs)

        last_error: Optional[Exception] = None
        delay: float = float(self.retry_config["initial_delay"])
        max_retries: int = int(self.retry_config["max_retries"])
        backoff: float = float(self.retry_config["backoff_factor"])

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                return (response.text or "").strip()
            except Exception as e:
                last_error = e
                should_retry = self._is_retryable_api_error(e)
                if not should_retry or attempt >= max_retries - 1:
                    raise

                jitter = random.uniform(0.0, min(0.25, delay * 0.1))
                time.sleep(delay + jitter)
                delay *= backoff

        raise RuntimeError(
            f"Failed to generate content after {max_retries} attempts"
        ) from last_error
