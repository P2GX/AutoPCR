from dataclasses import dataclass
from typing import Literal

ProviderBackend = Literal["openai", "ollama", "groq", "together", "vllm"]


@dataclass(frozen=True)
class Provider:
    name: ProviderBackend
    base_url: str | None
    concurrency: int
    local: bool = False
    api_key: str | None = None
