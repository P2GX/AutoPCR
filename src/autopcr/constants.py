from autopcr._utils.provider import Provider

PROVIDERS: dict[str, Provider] = {
    "openai": Provider(name="openai", base_url=None, concurrency=8),
    "groq": Provider(name="groq",  base_url="https://api.groq.com/openai/v1", concurrency=4),
    "together": Provider(name="together", base_url="https://api.together.xyz/v1", concurrency=8),
    "vllm": Provider(name="vllm", base_url="http://localhost:8000/v1", concurrency=16, local=True),
    "ollama": Provider(name="ollama", base_url="http://localhost:11434/v1", concurrency=4, local=True),
}
