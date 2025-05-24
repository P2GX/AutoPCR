import json
import subprocess


def prompt(pr, model, client):
    responses = {}
    if model == 'gpt':
        responses = prompt_gpt(pr, client)
    return responses


def prompt_batch(pr_batch_path, model):
    responses = {}
    if model == 'llama':
        command = f'mamba run -n vllm python utils_llama_infer.py --pr_batch_path {pr_batch_path}'
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return {"error": "An error occurred during the execution", "details": result.stderr}
        responses = json.load(open(pr_batch_path + '_res', 'r')) # json.loads(result.stdout)
    return responses


def prompt_gpt(pr, client, model='gpt-4o-mini'):
    responses = {}
    for i in pr:
        response = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system',
                        'content': pr[i]['prompt']},
                    {'role': 'user',
                        'content': pr[i]['mention_to_ground']}
                ],
                temperature=0,
                seed=0
            )
        responses[i] = response.choices[0].message.content
    return responses
