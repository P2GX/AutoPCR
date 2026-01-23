# Unsloth env for training: /nfs/turbo/umms-drjieliu/usr/yctao/finetune/.venv_unsloth
# for this file use autopcr env: /nfs/turbo/umms-drjieliu/usr/yctao/miniforge3/envs/autopcr
# data format: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Meta_Synthetic_Data_Llama3_2_(3B).ipynb
import sys
import os
import json
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# # 1. Define the path to the directory containing the file
# module_path = '/nfs/turbo/umms-drjieliu/usr/yctao/BIOCR/AutoPCR/src'

# # 2. Add this path to sys.path if it's not already there
# if module_path not in sys.path:
#     sys.path.append(module_path)

# 3. Now you can import the module as if it were in the current folder
from nn_model3 import gen_grounding_prompt2
from build_dict import lemmatizer, get_wordnet_pos

def lemmatize(term):
    tokens = word_tokenize(term.strip().lower().replace('-',' - ').replace('/',' / '))
    token_pos = nltk.pos_tag(tokens)
    lemmas = tuple([lemmatizer.lemmatize(token[0], get_wordnet_pos(token[1])) for token in token_pos])
    return lemmas

def filter_concepts(concepts):
    '''
    input concepts: dict of all concepts
    '''
    concepts_copy = concepts.copy()
    for id, concept in concepts_copy.items():
        synonyms = []
        lemmatized_synonyms = set()
        lemmatized_synonyms.add(lemmatize(concept['label']))
        for syn in concept['synonyms'].split(' | '):
            lem_syn = lemmatize(syn)
            if lem_syn in lemmatized_synonyms:
                continue
            synonyms.append(syn)
            lemmatized_synonyms.add(lem_syn)
        concepts_copy[id]['synonyms'] = ' | '.join(synonyms)
    return concepts_copy

ontology_dict = '../dict/HPO'
vocabfiles = {'index': f'{ontology_dict}/main_lable.index',
              'index_to_id': f'{ontology_dict}/index_to_id_lable.json',
              'index_to_term': f'{ontology_dict}/index_to_term_lable.json',
              'id_to_concept': f'{ontology_dict}/id_to_concept.json',
              'firstword_to_id': f'{ontology_dict}/firstword_id_map.json',
              'abbr': f'{ontology_dict}/abbr.json'
              }

class bioTag_SapBERT:
    def __init__(self, model_files=vocabfiles, ccr='cambridgeltl/SapBERT-from-PubMedBERT-fulltext', k=3):
        self.tokenizer = AutoTokenizer.from_pretrained(ccr)
        self.model = AutoModel.from_pretrained(ccr).cuda()
        self.index = faiss.read_index(model_files['index'])
        # self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss.read_index(model_files['index']))
        self.index_to_id = json.load(open(model_files['index_to_id'], 'r'))
        self.index_to_term = json.load(open(model_files['index_to_term'], 'r'))
        self.id_to_concept = filter_concepts(json.load(open(model_files['id_to_concept'], 'r')))
        self.firstword_to_id = json.load(open(model_files['firstword_to_id'], 'r'))
        self.abbr = json.load(open(model_files['abbr'], 'r'))
        self.k = k

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

    def get_cand(self, x, Top_N): # D, I?
        if not x:
            return {}
        embs = self.encode(x).astype('float32')
        faiss.normalize_L2(embs)
        # D: Distances to Nearest Neighbors, Shape: (num_queries, k)
        # I: Indices of Nearest Neighbors, Shape: (num_queries, k)
        # I is always sorted by increasing distance (best match first)
        D, I = self.index.search(embs, Top_N)
        concepts = {}
        for i in range(len(x)):
            concepts[i] = [(self.get_concept(str(I[i][j])), self.index_to_id[str(I[i][j])], float(D[i][j]), str(I[i][j])) for j in range(Top_N)]
        return concepts # note concepts follows the order of D and I, so still best match first

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
        Search_Buffer = max(50, Top_N * 10) 
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

nn_model = bioTag_SapBERT()
concepts = json.load(open(vocabfiles['id_to_concept'], 'r'))

def extract_search_terms(concepts_dict):
    """
    只提取 Label 和 Synonyms (按 '|' 分隔)。
    """
    search_data = {}
    for concept_id, concept_info in concepts_dict.items():
        terms = []
        
        # # 1. 提取 Label
        # if concept_info.get('label'):
        #     terms.append(concept_info['label'])
        
        # 2. 提取 Synonyms
        synonyms_str = concept_info.get('synonyms', '').strip()
        if synonyms_str:
            # 简单粗暴：按 | 分割，去空格，去空值
            terms.extend([s.strip() for s in synonyms_str.split('|') if s.strip()])
        
        # 去重并保存 (只要有词就行)
        if terms:
            search_data[concept_id] = list(set(terms))

    return search_data

def generate_contrastive_examples_outside(tagger, concepts_dict, Top_K=3):
    """
    Top_K should be greater than or equal to 1
    {id1: {syn1: ([positive exps], [negative exps]), syn2:...}, id2:...}
    纯粹的对比学习样本生成：
    输入一个字典，对每个 Label/Synonym 找 TopK。
    """
    # 获取数据
    filterted_concept_dict = filter_concepts(concepts_dict)
    search_data = extract_search_terms(filterted_concept_dict)
    output_dict = {}
    
    for target_id, search_terms in tqdm(search_data.items()):
        # 统一 ID 格式
        target_concept_id = target_id.replace(':', '_')
        
        # 批量检索当前 ID 下的所有词
        retrieved_results = tagger.get_cand_nonoverlap(search_terms, Top_K + 1) # best match first order
        
        output_dict[target_id] = {}
        
        for i, term in enumerate(search_terms):
            # 获取当前词检索到的 TopK 结果
            retrieved_list = retrieved_results.get(i, []) # best match first order
            # 提取检索到的 ID 列表
            retrieved_ids = [res[1].replace(':', '_') for res in retrieved_list] # best match first order
            
            positive_exp_ids = []
            negative_exp_ids = []
            
            # --- 核心逻辑 ---
            # we want: for both positive and negative examples, keep K candidates
            # check for synonyms: get rid of the synonyms in the ids
            if target_concept_id in retrieved_ids[:Top_K]:
                # 情况A: 本身在检索结果里
                # 正例: 本身 (1)
                positive_exp_ids = retrieved_ids[:Top_K]
                # 负例: 检索结果里除了本身以外的 (K-1)
                negative_exp_ids = [rid for rid in retrieved_ids if rid != target_concept_id]
            else:
                # 情况B: 本身不在检索结果里
                # 正例: 强行把本身加进来 (1)
                positive_exp_ids = retrieved_ids[:Top_K-1] + [target_concept_id]
                # 负例: 检索出来的全是错的，都作为负例
                negative_exp_ids = retrieved_ids[:Top_K]
            
            # so we have totally K examples
            assert len(positive_exp_ids) == Top_K
            assert len(negative_exp_ids) == Top_K
            
            # --- 转回 Concept 对象 ---
            final_pos = [tagger.id_to_concept[pid].copy() for pid in positive_exp_ids if pid in tagger.id_to_concept]
            final_neg = [tagger.id_to_concept[nid].copy() for nid in negative_exp_ids if nid in tagger.id_to_concept]

            if target_concept_id in positive_exp_ids:
                target_concept = final_pos[positive_exp_ids.index(target_concept_id)]
                synonyms = [s for s in target_concept['synonyms'].split(' | ') if s != term]
                final_pos[positive_exp_ids.index(target_concept_id)]['synonyms'] = ' | '.join(synonyms)

            output_dict[target_id][term] = (final_pos, final_neg)
            
    return output_dict

K = 3
# contrastive_data = generate_contrastive_examples_outside(nn_model, concepts_dict={'HP_0000003': concepts['HP_0000003']}, Top_K=K)
contrastive_data = generate_contrastive_examples_outside(nn_model, concepts_dict=concepts, Top_K=K)

# with open('data/contrastive_data.json', 'w') as f:
#     json.dump(contrastive_data, f)
# contrastive_data = json.load(open('data/contrastive_data.json', 'r'))

pos_prompts, neg_prompts = [], []
for id in tqdm(contrastive_data):
    for syn in contrastive_data[id]:
        pos_prompt = gen_grounding_prompt2({'label': syn}, contrastive_data[id][syn][0], '')
        neg_prompt = gen_grounding_prompt2({'label': syn}, contrastive_data[id][syn][1], '')
        pos_prompt = {'messages': [
            {'content': pos_prompt['system'], 'role': 'system'},
            {'content': pos_prompt['user'], 'role': 'user'},
            {'content': f'answer: {id}', 'role': 'assistant'},
        ]}        
        neg_prompt = {'messages': [
            {'content': neg_prompt['system'], 'role': 'system'},
            {'content': neg_prompt['user'], 'role': 'user'},
            {'content': 'answer: None', 'role': 'assistant'},
        ]}
        pos_prompts.append(pos_prompt)
        neg_prompts.append(neg_prompt)

with open('data/pos_prompts_noumls.json', 'w') as f:
    json.dump(pos_prompts, f)
with open('data/neg_prompts_noumls.json', 'w') as f:
    json.dump(neg_prompts, f)