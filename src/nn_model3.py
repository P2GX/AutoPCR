import time
import sys
import numpy as np
from transformers import AutoTokenizer, AutoModel
import faiss
import json
from tqdm import tqdm
import re
import yaml
import os
import torch
from nltk.tokenize import word_tokenize
import nltk

from utils import prompt, GROUNDING_PROMPT_NOCONF
from build_dict import lemmatizer, get_wordnet_pos


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
    system_prompt = GROUNDING_PROMPT_NOCONF
    if examples:
        system_prompt += f'Here are some examples:\n' + "\n".join(examples)+ '\n'

    if linking_fields is None:
        linking_fields = ("label", "definition", 'synonyms') # 'synonyms'
    
    _candidates = []
    for i, concept in enumerate(concepts):
        _candidates.append(f'ID: {concept["id"]}')
        _candidates.append(get_description(concept, linking_fields) + "\n")

    system_prompt += '\nBelow are the concepts:\n\n' +  "\n".join(f'{c}' for c in _candidates)
    # system_prompt += '\nBelow are the context:\n\n' +  text
    # 'If the chosen concept is broader or narrower than the given entity, return \"None\".' (broader)
    mentions = f'[Entity to link]\n{get_description(mention, fields=["label"])}'
    # mentions += f'\ncontext: {text}'
    # mentions += '\n\nBelow are the concepts:\n\n' +  "\n".join(f'{c}' for c in _candidates)     
    return {"user": mentions, "system": system_prompt}


class bioTag_SapBERT:
    def __init__(self, model_files, ccr, tau_1, tau_2, k, el, api_provider, api_key, seed, use_cache, cand_path=''):
        self.model_type = 'sapbert'
        # microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
        # ../../sapbert/train/tmp/sapbert_umls-hpo
        self.tokenizer = AutoTokenizer.from_pretrained(ccr)
        self.model = AutoModel.from_pretrained(ccr).cuda()
        self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss.read_index(model_files['index']))
        self.index_to_id = json.load(open(model_files['index_to_id'], 'r'))
        self.index_to_term = json.load(open(model_files['index_to_term'], 'r'))
        self.id_to_concept = json.load(open(model_files['id_to_concept'], 'r'))
        self.firstword_to_id = json.load(open(model_files['firstword_to_id'], 'r'))
        self.abbr = json.load(open(model_files['abbr'], 'r'))
        self.llm_range = (tau_2, tau_1)
        self.k = k
        self.use_cache = use_cache
        self.cache = {}
        self.el = el
        self.api_provider = api_provider
        self.api_key = api_key
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

    def get_concept(self, index):
        concept = self.id_to_concept[self.index_to_id[index].replace(':', '_')]
        return concept
    
    def get_concept_from_index(self, index):
        concept = self.id_to_concept[self.index_to_id[index].replace(':', '_')].copy()
        term = self.index_to_term[index]
        if concept['label'] == term:
            return concept
        synoyms = concept['synonyms'].split(' | ')
        assert term in synoyms
        new_synonyms = [concept['label']]
        for s in synoyms:
            if s != term:
                new_synonyms.append(s)
        concept['label'] = term
        concept['synonyms'] = ' | '.join(new_synonyms)
        return concept

    def get_cand(self, x, Top_N):
        if not x:
            return {}
        embs = self.encode(x).astype('float32')
        faiss.normalize_L2(embs)
        D, I = self.index.search(embs, Top_N)
        concepts = {}
        for i in range(len(x)):
            concepts[i] = [(self.get_concept(str(I[i][j])), self.index_to_id[str(I[i][j])], float(D[i][j]), str(I[i][j])) for j in range(Top_N)]
        return concepts

    def get_cand_nonoverlap(self, x, Top_N):
        """
        修改版：返回 Top_N 个【不同的】Concept ID。
        原理：搜多一点(Top_N * 10)，然后遍历去重。
        """
        if not x:
            return {}
        
        embs = self.encode(x).astype('float32')
        faiss.normalize_L2(embs)
        
        # 1. 扩大搜索范围 (Search Buffer)，防止前几个全是同一个ID
        # 比如你要 Top 3，我就搜 50 个回来挑
        Search_Buffer = max(100, Top_N * 10) 
        D, I = self.index.search(embs, Search_Buffer)
        # ^ D: Distances to Nearest Neighbors, Shape: (num_queries, k)
        # I: Indices of Nearest Neighbors, Shape: (num_queries, k)
        # I is always sorted by increasing distance (best match first)

        concepts = {}
        for i in range(len(x)):
            unique_candidates = []
            seen_ids = set() # 用来记录已经存入的 ID
            
            # 遍历 Buffer 里的结果
            for j in range(Search_Buffer):
                faiss_idx = str(I[i][j])
                # 获取 Concept ID
                c_id = self.index_to_id[faiss_idx]
                
                # 2. 关键去重逻辑：如果这个 ID 没见过，才加进去
                if c_id not in seen_ids:
                    seen_ids.add(c_id) # seen_ids do not follow best match first order
                    
                    # 组装结果
                    score = float(D[i][j])
                    concept_detail = self.get_concept(faiss_idx)
                    unique_candidates.append((concept_detail, c_id, score, faiss_idx)) # unique candidates follow best match first order
                
                # 3. 凑够了 Top_N 个就停
                if len(unique_candidates) >= Top_N:
                    break
            
            concepts[i] = unique_candidates # concepts follow best match first order
            
        return concepts # concepts follow best match first order

    def predict(self, x, Top_N=3):
        concepts = self.get_cand(x, Top_N)
        lines = ''
        for i in range(len(x)):
            lines += '\t'.join([f'{concepts[i][j][1]}|{concepts[i][j][2]}' for j in range(Top_N)]) + '\n'
        return lines

    def predict_llm(self, x, texts, confidence=('HIGH')):
        self.counter += len(x)
        concepts = self.get_cand(x, self.k)
        # for i, c in concepts.items():
        #     if x[i] == 'oral anomalies':
        #         print(x[i], [(c[j][0]['label'], c[j][0]['id'], c[j][2]) for j in range(self.k)])
        pr = {}
        for i in range(len(x)):
            if self.use_cache and x[i] in self.cache:
                concepts[i] = self.cache[x[i]]
                continue
            # if ' ' not in x[i]:
            #     continue
            if ' ' not in x[i]:
                # TODO: x[i] may be plural
                if id:=self.abbr.get(x[i].lower(), ''):
                    concepts[i] = [(self.id_to_concept[id.replace(':', '_')], id, 1.0)] + concepts[i]
                    continue
                tokens = word_tokenize(x[i].strip().lower().replace('-',' - ').replace('/',' / '))
                if len(tokens) == 1:
                    if tokens[0] in self.firstword_to_id:
                        # print(tokens)
                        id = self.firstword_to_id[tokens[0]][0]
                        concepts[i] = [(self.id_to_concept[id.replace(':', '_')], id, 1.0)] + concepts[i]
                        continue
                    token_pos = nltk.pos_tag(tokens)
                    lemmas = [lemmatizer.lemmatize(token[0], get_wordnet_pos(token[1])) for token in token_pos]
                    if lemmas[0] in self.firstword_to_id:
                        # print(lemmas)
                        id = self.firstword_to_id[lemmas[0]][0]
                        concepts[i] = [(self.id_to_concept[id.replace(':', '_')], id, 1.0)] + concepts[i]
                    continue
            if self.llm_range[0] <= concepts[i][0][2] < self.llm_range[1]:
                # concepts_ = {}
                # for k in range(len(concepts[i])):
                #     if concepts[i][k][0]['id'] not in concepts_:
                #         concepts_[concepts[i][k][0]['id']] = concepts[i][k][0]
                #     if len(concepts_) == self.k:
                #         break
                # concepts_ = list(concepts_.values())
                # key: (concepts[i][k][0]['id'], concepts[i][k][0]['label'])
                concepts_ = list({concepts[i][k][0]['id']: concepts[i][k][0] for k in range(self.k)}.values())
                pr[i] = gen_grounding_prompt2({'label': x[i]}, concepts_, texts[i])
        # print([pr[i] for i in pr if x[i] == 'thumb anomalies'])
        responses = prompt(pr, self.el, self.api_provider, self.api_key, self.seed) if self.el != 'none' else {}
        # lines = ''
        anns = []
        for i in range(len(x)):
            if i in responses:
                uri = re.findall('(?<=answer: )[^\n]+', responses[i])
                if uri and 'none' not in uri[0].lower():
                    # lines += f"{uri[0].strip().replace('_', ':')}|{-1.0}" + '\n'
                    # continue
                    id = uri[0].strip()
                    concepts[i] = [(self.id_to_concept.get(id, {}), id.replace('_', ':'), -1.0)] + concepts[i]
            # lines += '\t'.join([f'{concepts[i][j][1]}|{concepts[i][j][2]}' for j in range(self.k)]) + '\n'
            anns.append([(concepts[i][j][1], concepts[i][j][2]) for j in range(self.k)])
            if self.use_cache and x[i] not in self.cache:
                self.cache[x[i]] = concepts[i]
        # return lines
        return anns