import json
import subprocess
import os
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from together import Together

load_dotenv(os.path.dirname(__file__)+"/../.env")


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
        prompt.api_key = api_key
    client = prompt.client

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


def prompt_batch(pr_batch_path, model):
    responses = {}
    if model == 'llama':
        command = f'mamba run -n vllm python llama_infer.py --pr_batch_path {pr_batch_path}'
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return {"error": "An error occurred during the execution", "details": result.stderr}
        responses = json.load(open(pr_batch_path + '_res', 'r')) # json.loads(result.stdout)
    return responses