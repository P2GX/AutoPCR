import time
import sys
import numpy as np
from transformers import AutoTokenizer, AutoModel
import faiss
import json
from tqdm import tqdm
import re
import yaml
from utils_llm import prompt, prompt_batch
import os
import torch

GROUNDING_PROMPT = (
    "As an expert clinician, your task is to accurately link the entity using the concepts listed below. "
    "Accuracy is paramount. "
    # "If the entity refers to multiplt concepts, return the last one. "
    "If the entity does not precisely refer to any of the concepts listed below, please return \"None\"; "
    "otherwise, return the corresponding concept ID in the following format:\n"
    # "reason: <one-sentence reasoning about entity linking>\n"
    "answer: <concept ID or None>\n"
    "confidence: <one of: HIGH, LOW, MEDIUM>\n"
)


def get_description(metadata, fields=None):
    if fields is None:
        if isinstance(metadata, dict):
            return yaml.safe_dump({k: v for k, v in metadata.items() if v}, sort_keys=False).strip()
        if isinstance(metadata, str):
            return metadata
    elif isinstance(fields, (list, set, tuple)):
        # if ('definition' in metadata and metadata['definition']) or ('synonyms' in metadata and 'synonyms' in metadata['synonyms']):
        #     fields = ("label", "definition", 'synonyms')
        return yaml.safe_dump({k: v for k, v in metadata.items() if k in fields and v}, sort_keys=False).strip()
    elif isinstance(fields, str) and isinstance(metadata, dict):
        return metadata[fields]


def gen_grounding_prompt2(mention, concepts, text, linking_fields=None, examples=None):
    system_prompt = GROUNDING_PROMPT
    if examples:
        system_prompt += f'Here are some examples:\n' + "\n".join(examples)+ '\n'

    if linking_fields is None:
        linking_fields = ("label", "definition", 'synonyms', 'UMLS synonyms')
    
    _candidates = []
    for i, concept in enumerate(concepts):
        _candidates.append(f'ID: {concept["id"]}')
        _candidates.append(get_description(concept, linking_fields) + "\n")

    system_prompt += 'Below are the concepts:\n\n' +  "\n".join(f'{c}' for c in _candidates)
    # system_prompt += '\nBelow are the text:\n\n' +  text
    # 'If the chosen concept is broader or narrower than the given entity, return \"None\".' (broader)
    mentions = f'\n\n[Entity to link]\n{get_description(mention, fields=["label"])}'
    # mentions += f'\ncontext: {text}'     
    return {"mention_to_ground": mentions, "prompt": system_prompt}


def gen_grounding_prompt2_llama(mention, concepts, text, linking_fields=None, examples=None):
    system_prompt = GROUNDING_PROMPT
    if examples:
        system_prompt += f'Here are some examples:\n' + "\n".join(examples)+ '\n'

    if linking_fields is None:
        linking_fields = ("label", "definition", 'synonyms', 'UMLS synonyms')
    
    _candidates = []
    real_id = {}
    for i, concept in enumerate(concepts):
        # _candidates.append(f'ID: {concept["id"]}')
        _candidates.append(f'ID: {i+1}')
        real_id[i+1] = concept["id"]
        _candidates.append(get_description(concept, linking_fields) + "\n")

    system_prompt += 'Below are the concepts:\n\n' +  "\n".join(f'{c}' for c in _candidates)
    # system_prompt += '\nBelow are the text:\n\n' +  text
    # 'If the chosen concept is broader or narrower than the given entity, return \"None\".' (broader)
    mentions = f'\n\n[Entity to link]\n{get_description(mention, fields=["label"])}'    
    return {"mention_to_ground": mentions, "prompt": system_prompt, "real_id": real_id}


class bioTag_SapBERT:
    def __init__(self, model_files, tau_1, tau_2, k, llm, client, seed, cand_path=''):
        self.model_type = 'sapbert'
        self.tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
        self.model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext').cuda()
        # self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
        # self.model = AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext').cuda()
        # self.tokenizer = AutoTokenizer.from_pretrained('../../sapbert/train/tmp/sapbert_umls-hpo')
        # self.model = AutoModel.from_pretrained('../../sapbert/train/tmp/sapbert_umls-hpo').cuda()
        self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss.read_index(model_files['index']))
        self.index_to_id = json.load(open(model_files['index_to_id'], 'r'))
        self.id_to_concept = json.load(open(model_files['id_to_concept'], 'r'))
        self.llm_range = (tau_2, tau_1)
        self.k = k
        self.llm = llm
        self.client = client
        self.seed = seed
        self.cand_path = cand_path
        self.counter = 0

    def encode(self, all_names, bs=128):
        all_embs = []
        for i in np.arange(0, len(all_names), bs):
            toks = self.tokenizer.batch_encode_plus(all_names[i:i+bs], 
                                            padding="max_length", 
                                            max_length=100, 
                                            truncation=True,
                                            return_tensors="pt")
            toks_cuda = {}
            for k,v in toks.items():
                toks_cuda[k] = v.cuda()
            cls_rep = self.model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
            all_embs.append(cls_rep.cpu().detach().numpy())
        if all_embs:
            all_embs = np.concatenate(all_embs, axis=0)
        return all_embs

    def get_cand(self, x, Top_N):
        if not x:
            return {}
        embs = self.encode(x).astype('float32')
        faiss.normalize_L2(embs)
        D, I = self.index.search(embs, Top_N)
        concepts = {}
        for i in range(len(x)):
            concepts[i] = [(self.id_to_concept[self.index_to_id[str(I[i][j])].replace(':', '_')], self.index_to_id[str(I[i][j])], float(D[i][j]), str(I[i][j])) for j in range(Top_N)]
        return concepts

    def predict(self, x, Top_N=3):
        concepts = self.get_cand(x, Top_N)
        lines = ''
        for i in range(len(x)):
            lines += '\t'.join([f'{concepts[i][j][1]}|{concepts[i][j][2]}' for j in range(Top_N)]) + '\n'
        return lines

    def predict_llm(self, x, texts, confidence=('HIGH')):
        self.counter += len(x)
        concepts = self.get_cand(x, self.k)
        pr = {}
        for i in range(len(x)):
            if ' ' not in x[i]:
                continue
            if self.llm_range[0] <= concepts[i][0][2] < self.llm_range[1]:
                # concepts_ = {}
                # for k in range(len(concepts[i])):
                #     if concepts[i][k][0]['id'] not in concepts_:
                #         concepts_[concepts[i][k][0]['id']] = concepts[i][k][0]
                #     if len(concepts_) == self.k:
                #         break
                # concepts_ = list(concepts_.values())
                concepts_ = list({concepts[i][k][0]['id']: concepts[i][k][0] for k in range(self.k)}.values())
                pr[i] = gen_grounding_prompt2({'label': x[i]}, concepts_, texts[i])
        # print(pr)
        responses = prompt(pr, self.llm, self.client, self.seed)
        lines = ''
        for i in range(len(x)):
            if i in responses:
                uri = re.findall('(?<=answer: )[^\n]+', responses[i])
                conf = re.findall('(?<=confidence: )[A-Z]+', responses[i])
                if uri and conf and conf[0] in confidence and 'none' not in uri[0].lower():
                    lines += f"{uri[0].strip().replace('_', ':')}|{-1.0}" + '\n'
                    continue
            lines += '\t'.join([f'{concepts[i][j][1]}|{concepts[i][j][2]}' for j in range(self.k)]) + '\n'
        return lines

    # TODO: for local inference
    def predict_llm_batch(self, x_batch, texts_batch, confidence=('HIGH')):
        if os.path.exists(self.cand_path):
            concepts_batch = json.load(open(self.cand_path, 'r'))
            concepts_batch = {int(i): {int(j): concepts[j] for j in concepts} for i, concepts in concepts_batch.items()}
        else:
            concepts_batch = {i: self.get_cand(x, self.k) for i, x in enumerate(x_batch)}
            json.dump(concepts_batch, open(self.cand_path, 'w'))

        if os.path.exists(self.cand_path + '_pr_res'):
            responses_batch = json.load(open(self.cand_path + '_pr_res', 'r'))
            responses_batch = {int(i): {int(j): responses[j] for j in responses} for i, responses in responses_batch.items()}
        else:
            pr_batch = {}
            for j, x in enumerate(x_batch):
                pr, concepts, texts = {}, concepts_batch[j], texts_batch[j]
                for i in range(len(x)):
                    if self.llm_range[0] <= concepts[i][0][2] < self.llm_range[1]:
                        concepts_ = list({concepts[i][k][0]['id']: concepts[i][k][0] for k in range(self.k)}.values())
                        pr[i] = gen_grounding_prompt2_llama({'label': x[i]}, concepts_, texts[i])
                pr_batch[j] = pr
            json.dump(pr_batch, open(self.cand_path + '_pr', 'w'))
            exit(0)
            responses_batch = prompt_batch(self.cand_path + '_pr', self.llm)

        lines_batch = []
        for j, x in enumerate(x_batch):
            concepts = concepts_batch[j]
            lines = ''
            for i in range(len(x)):
                if j in responses_batch and i in responses_batch[j]:
                    responses = responses_batch[j]
                    uri = re.findall('(?<=answer: )[^\n]+', responses[i][0])
                    if uri:
                        uri[0] = responses[i][1].get(uri[0].strip(), 'None')
                    conf = re.findall('(?<=confidence: )[A-Z]+', responses[i][0])
                    if uri and conf and conf[0] in confidence and 'none' not in uri[0].lower():
                        lines += f'''{uri[0].replace('_', ':')}|{-1.0}''' + '\n'
                        continue
                lines += '\t'.join([f'{concepts[i][k][1]}|{concepts[i][k][2]}' for k in range(self.k)]) + '\n'
            lines_batch.append(lines)
        return lines_batch


def print_gpu_usage():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(torch.device('cuda:0'))
        print(f'GPU memory free: {free / 1024**3} GB')
    else:
        print('CUDA is not available.')