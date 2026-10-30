"""One tiny client for every OpenAI-compatible chat backend.

OpenAI, Groq, Together, vLLM and Ollama all speak the same
`/v1/chat/completions` protocol, so a single `openai.OpenAI` client
pointed at the right base URL covers all of them.

    from llm_client import prompt

    prompts = {
        "q1": {"system": "Answer briefly.", "user": "Capital of France?"},
        "q2": {"system": "Answer briefly.", "user": "2 + 2?"},
    }
    prompt(prompts, "llama3.1:8b", api_provider="ollama")
    # -> {"q1": "Paris.", "q2": "4"}

Base URLs can be overridden per provider with `<PROVIDER>_BASE_URL`
(e.g. `VLLM_BASE_URL=http://gpu-box:8000/v1`) or the `base_url` argument.
"""

from __future__ import annotations

import os
from collections.abc import Hashable, Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import TypedDict

import httpx
from openai import OpenAI

__all__ = ["Prompt", "PROVIDERS", "get_client", "prompt"]

from autopcr._utils.provider import Provider
from autopcr.constants import PROVIDERS


class Prompt(TypedDict):
    system: str
    user: str


def _resolve(
    api_provider: str, api_key: str | None, base_url: str | None
) -> tuple[Provider, str, str | None]:
    """Work out which provider, key and URL to use."""
    name = api_provider.lower()
    try:
        provider = PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown provider {api_provider!r}; choose one of {sorted(PROVIDERS)}"
        ) from None

    # Backwards compatibility: the old code passed the vLLM URL as `api_key`.
    if provider.local and api_key and api_key.startswith(("http://", "https://")):
        base_url, api_key = api_key, None

    base_url = base_url or os.getenv(f"{name.upper()}_BASE_URL") or provider.base_url
    api_key = api_key or os.getenv(f"{name.upper()}_API_KEY")

    if not api_key:
        if not provider.local:
            raise RuntimeError(
                f"No API key: pass api_key= or set {name.upper()}_API_KEY"
            )
        api_key = "EMPTY"  # vLLM/Ollama accept any non-empty string

    return provider, api_key, base_url


@lru_cache(maxsize=None)
def get_client(api_key: str | None, base_url: str | None, local: bool) -> OpenAI:
    """Build (once) and cache a client per (key, url) combination."""
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=3,
        timeout=httpx.Timeout(600.0 if local else 120.0, connect=10.0),
        # Local servers: ignore HTTP(S)_PROXY so localhost isn't routed through a proxy.
        http_client=httpx.Client(trust_env=False) if local else None,
    )


def prompt(
    pr: Mapping[Hashable, Prompt],
    model: str,
    provider: Provider,
    seed: int = 0,
    max_tokens: int = 32,
    *,
    temperature: float = 0.0,
    concurrency: int | None = None,
) -> dict[Hashable, str]:
    """Run every prompt in `pr` and return {id: response_text}, in input order."""
    # provider, key, url = _resolve(provider, api_key, base_url)
    provider, api_key, base_url = _resolve(
        provider.name, api_key=provider.api_key, base_url=provider.base_url
    )
    client = get_client(api_key, base_url, provider.local)

    def complete(p: Prompt) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": p["user"]},
            ],
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    workers = max(1, min(concurrency or provider.concurrency, len(pr)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(complete, pr.values())
        return dict(zip(pr.keys(), results))
