"""Runtime configuration, read from environment variables / a .env file.

Every stage talks to an OpenAI-compatible endpoint, so any provider that
speaks that protocol (Groq, OpenAI, Together, OpenRouter, a local Ollama or
vLLM server, ...) can be plugged in by changing base URL + model name.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv()  # also honour a .env in the current working directory
except ImportError:  # python-dotenv is optional
    pass

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class ModelEndpoint:
    """One model behind an OpenAI-compatible API."""

    model: str
    base_url: str | None
    api_key: str | None
    role: str = ""

    @property
    def label(self) -> str:
        host = (self.base_url or "api.openai.com").split("//")[-1].split("/")[0]
        return f"{self.model} @ {host}"


@dataclass(frozen=True)
class Settings:
    # Stage 1 — speech-to-text
    stt_backend: str = "api"  # "api" (OpenAI-compatible /audio/transcriptions) or "local" (faster-whisper)
    stt: ModelEndpoint = field(default_factory=lambda: ModelEndpoint("whisper-large-v3", GROQ_BASE_URL, None, "stt"))
    local_whisper_model: str = "large-v3"
    local_whisper_device: str = "auto"
    language: str = "en"

    # Stage 2 — transcript refinement LM
    refine: ModelEndpoint = field(
        default_factory=lambda: ModelEndpoint("openai/gpt-oss-20b", GROQ_BASE_URL, None, "refine")
    )
    # Stage 3 — meeting documentation LM
    document: ModelEndpoint = field(
        default_factory=lambda: ModelEndpoint("openai/gpt-oss-120b", GROQ_BASE_URL, None, "document")
    )
    document_reasoning_effort: str | None = "medium"  # only sent to reasoning models (gpt-oss, o-series)
    refine_reasoning_effort: str | None = "low"  # refinement only proposes small edits; deep reasoning is wasted tokens
    document_map_reasoning_effort: str | None = "low"  # per-part note extraction in map-reduce (the merge keeps document effort)
    llm_concurrency: int = 1  # parallel chunk requests; raise to 4 on a paid Groq plan (free tier is token-rate bound)

    # Limits & tuning
    max_upload_mb: float = 500.0
    max_duration_min: float = 180.0
    stt_chunk_seconds: float = 600.0  # chunk length for long recordings (split at silences)
    refine_chunk_words: int = 1400
    document_max_words: int = 3000  # longer transcripts go through map-reduce
    request_timeout: float = 180.0
    max_retries: int = 5
    evidence_match_threshold: float = 82.0

    output_dir: Path = Path("runs")


def _effort(name: str, default: str) -> str | None:
    value = _env(name, default)
    return None if value.lower() in ("none", "off", "") else value


def load_settings() -> Settings:
    """Build Settings from the environment."""
    default_key = _env("GROQ_API_KEY") or _env("OPENAI_API_KEY")
    default_base = _env("API_BASE_URL", GROQ_BASE_URL if _env("GROQ_API_KEY") or not _env("OPENAI_API_KEY") else None)

    def endpoint(prefix: str, default_model: str, role: str) -> ModelEndpoint:
        return ModelEndpoint(
            model=_env(f"{prefix}_MODEL", default_model),
            base_url=_env(f"{prefix}_BASE_URL", default_base),
            api_key=_env(f"{prefix}_API_KEY", default_key),
            role=role,
        )

    return Settings(
        stt_backend=_env("STT_BACKEND", "api").lower(),
        stt=endpoint("STT", "whisper-large-v3", "stt"),
        local_whisper_model=_env("LOCAL_WHISPER_MODEL", "large-v3"),
        local_whisper_device=_env("LOCAL_WHISPER_DEVICE", "auto"),
        language=_env("STT_LANGUAGE", "en"),
        refine=endpoint("REFINE", "openai/gpt-oss-20b", "refine"),
        document=endpoint("DOCUMENT", "openai/gpt-oss-120b", "document"),
        document_reasoning_effort=_effort("DOCUMENT_REASONING_EFFORT", "medium"),
        refine_reasoning_effort=_effort("REFINE_REASONING_EFFORT", "low"),
        document_map_reasoning_effort=_effort("DOCUMENT_MAP_REASONING_EFFORT", "low"),
        llm_concurrency=max(1, _env_int("LLM_CONCURRENCY", 1)),
        max_upload_mb=_env_float("MAX_UPLOAD_MB", 500.0),
        max_duration_min=_env_float("MAX_DURATION_MIN", 180.0),
        stt_chunk_seconds=_env_float("STT_CHUNK_SECONDS", 600.0),
        refine_chunk_words=_env_int("REFINE_CHUNK_WORDS", 1400),
        document_max_words=_env_int("DOCUMENT_MAX_WORDS", 3000),
        request_timeout=_env_float("REQUEST_TIMEOUT", 180.0),
        max_retries=_env_int("MAX_RETRIES", 5),
        evidence_match_threshold=_env_float("EVIDENCE_MATCH_THRESHOLD", 82.0),
        output_dir=Path(_env("OUTPUT_DIR", "runs")),
    )


def with_overrides(settings: Settings, **kwargs) -> Settings:
    return replace(settings, **kwargs)
