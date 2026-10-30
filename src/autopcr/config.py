from dataclasses import dataclass, asdict
from pathlib import Path

from autopcr._utils.llm import Provider
from autopcr.constants import PROVIDERS


@dataclass(frozen=True)
class AutoPCRConfig:
    working_dir: Path
    candidate_threshold: float = 0.85
    accept_threshold: float = 0.85
    span_extractor: str = "neural++"
    detect_abbreviations: bool = False
    retrieval_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    accept_threshold: float = 0.95
    candidate_threshold: float = 0.85
    num_candidates: int = 5
    linking_model: str = (
        "ttempvnn/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-Q4-K-M"
    )
    api_key: str | None = None
    use_finetuning_prompt: bool = False
    seed: int = 42
    use_cache: bool = False
    keep_longest_span_only: bool = False
    device: str | None = None
    provider: Provider = PROVIDERS["ollama"]

    def __post_init__(self):
        if not 0 <= self.candidate_threshold <= self.accept_threshold <= 1:
            raise ValueError("Need 0 <= candidate_threshold <= accept_threshold <= 1")
