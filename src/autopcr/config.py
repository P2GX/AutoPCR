from dataclasses import dataclass, asdict
from pathlib import Path


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
    linking_model: str = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    api_provider: str = "together"
    api_key: str | None = None
    use_finetuning_prompt: bool = False
    seed: int = 42
    use_cache: bool = False
    keep_longest_span_only: bool = False

    def __post_init__(self):
        if not 0 <= self.candidate_threshold <= self.accept_threshold <= 1:
            raise ValueError("Need 0 <= candidate_threshold <= accept_threshold <= 1")
