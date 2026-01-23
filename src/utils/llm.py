import json
import subprocess
import os
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from together import Together
from concurrent.futures import ThreadPoolExecutor

load_dotenv(os.path.dirname(__file__)+'/../.env')


def prompt(pr, model, api_provider='openai', api_key=None, seed=0, max_tokens=32):
    if not api_key:
        api_key = os.getenv(f'{api_provider.upper()}_API_KEY')
    if not hasattr(prompt, 'client') or prompt.api_key != api_key:
        if api_provider == 'openai':
            prompt.client = OpenAI(api_key=api_key)
        elif api_provider == 'groq':
            prompt.client = Groq(api_key=api_key)
        elif api_provider == 'together':
            prompt.client = Together(api_key=api_key)
        elif api_provider == 'vllm':
            prompt.client = OpenAI(api_key='EMPTY', base_url=api_key, http_client=httpx.Client(trust_env=False))
        prompt.api_key = api_key
    client = prompt.client

    if api_provider == 'vllm':
        vllm_concurrency = 16
        def one(item):
            i, p = item
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': p['system']},
                    {'role': 'user',   'content': p['user']},
                ],
                temperature=0,
                seed=seed,
                max_tokens=max_tokens,
            )
            return i, r.choices[0].message.content
        with ThreadPoolExecutor(max_workers=vllm_concurrency) as pool:
            return dict(pool.map(one, pr.items()))

    responses = {}
    for i in pr:
        response = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': pr[i]['system']},
                    {'role': 'user', 'content': pr[i]['user']}
                ],
                temperature=0,
                seed=seed,
                max_tokens=max_tokens
            )
        responses[i] = response.choices[0].message.content
    return responses