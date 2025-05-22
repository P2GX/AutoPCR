import json
# import os
# os.environ['VLLM_CONFIGURE_LOGGING'] = '0'
from vllm import LLM, SamplingParams
import argparse
from tqdm import tqdm

def prompt_llama(pr_batch, model='meta-llama/Llama-3.1-8B-Instruct'):
    llm = LLM(model=model, seed=0)
    responses = {}
    for j, pr in tqdm(pr_batch.items()):
        conversations = [
            [{'role': 'system', 'content': pr_['prompt']},
            {'role': 'user', 'content': pr_['mention_to_ground']}]
            for pr_ in pr.values()]
        outputs = llm.chat(conversations, SamplingParams(max_tokens=4096, temperature=0))
        responses[j] = {i: [output.outputs[0].text, pr[i]['real_id']] for i, output in zip(pr, outputs)}      
    return responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My Python script that requires two string arguments.")
    parser.add_argument("--pr_batch_path", type=str, help="The first required string argument.")
    # parser.add_argument("model", type=str, help="The second required string argument.")
    args = parser.parse_args()
    
    pr_batch = json.load(open(args.pr_batch_path, 'r'))

    responses = prompt_llama(pr_batch)
    
    json.dump(responses, open(args.pr_batch_path + '_res', 'w'))
