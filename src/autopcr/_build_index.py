from pathlib import Path

import numpy as np
import requests
import torch
from transformers import AutoTokenizer, AutoModel
import os
import json
import faiss

from autopcr._working_files import (
    OBO_JSON_FILE,
    INDEX_TO_ID_FILE,
    INDEX_TO_TERM_FILE,
    TERM_TO_MAIN_LABEL_FILE,
    MAIN_INDEX_FILE,
    LABLE_VOCAB_FILE,
    INDEX_TO_ID_LABEL_FILE,
    INDEX_TO_TERM_LABEL_FILE,
    MAIN_LABEL_INDEX_FILE,
    ID_TO_CONCEPT_FILE,
)


class IndexBuilder:
    def __init__(
        self, hf_model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    ):
        self._tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        self._model = AutoModel.from_pretrained(hf_model_name)

    def build_index(self, working_dir: Path):
        with open(working_dir / OBO_JSON_FILE, "r") as f:
            hpo_concepts = json.load(f)

        # 🔹 3. 计算 embedding 并存入 Faiss
        term_to_main_label = {}  # 存储同义词 → 主要术语的映射
        term_to_id = {}

        all_terms = []
        for main_label, info in hpo_concepts.items():
            terms = [info["name"][0].lower()] + [
                t[0].strip().lower() for t in info["synonym"]
            ]  # 术语 + 同义词
            all_terms += terms
            term_to_main_label.update(
                {syn: info["name"][0].lower() for syn in terms}
            )  # 记录同义词 → 主术语
            term_to_id.update({syn: main_label for syn in terms})

        index_to_id = {i: term_to_id[all_terms[i]] for i in range(len(all_terms))}
        index_to_term = {i: all_terms[i] for i in range(len(all_terms))}
        with open(working_dir / INDEX_TO_ID_FILE, "w") as f:
            json.dump(index_to_id, f)
        with open(working_dir / INDEX_TO_TERM_FILE, "w") as f:
            json.dump(index_to_term, f)
        with open(working_dir / TERM_TO_MAIN_LABEL_FILE, "w") as f:
            json.dump(term_to_main_label, f)

        # 🔹 4. 转换为 numpy 数组，并存入 Faiss
        ontology_embeddings = np.array(
            self._get_embs(all_terms, self._tokenizer, self._model)
        ).astype("float32")
        faiss.normalize_L2(ontology_embeddings)

        faiss_index = faiss.IndexFlatIP(ontology_embeddings.shape[1])
        faiss_index.add(ontology_embeddings)
        faiss.write_index(faiss_index, str(working_dir / MAIN_INDEX_FILE))

        # 🔹 5. Make index focus on lable.vocab
        with open(working_dir / LABLE_VOCAB_FILE, "r") as f:
            ids_filtered = {line.strip() for line in f if line.strip()}

        indices_filtered = [
            index for index, id_ in index_to_id.items() if id_ in ids_filtered
        ]

        index_to_id_filtered = {
            i: index_to_id[index] for i, index in enumerate(indices_filtered)
        }
        index_to_term_filtered = {
            i: index_to_term[index] for i, index in enumerate(indices_filtered)
        }
        with open(working_dir / INDEX_TO_ID_LABEL_FILE, "w") as f:
            json.dump(index_to_id_filtered, f)

        with open(working_dir / INDEX_TO_TERM_LABEL_FILE, "w") as f:
            json.dump(index_to_term_filtered, f)

        embs = np.array([faiss_index.reconstruct(i) for i in range(faiss_index.ntotal)])
        embs_filtered = np.array([embs[int(i)] for i in indices_filtered])

        faiss_index_filtered = faiss.IndexFlatIP(faiss_index.d)

        faiss_index_filtered.add(embs_filtered)
        faiss.write_index(
            faiss_index_filtered, str(working_dir / MAIN_LABEL_INDEX_FILE)
        )

        def list2str(some_list, delimiter=" | "):
            return delimiter.join([str(i) for i in some_list])

        concepts = {}
        for id, concept in hpo_concepts.items():
            id = id.replace(":", "_")
            concepts[id] = {
                "id": id,
                "label": concept["name"][0],
                "definition": concept["def"],
                "synonyms": list2str([s[0] for s in concept["synonym"]]),
                "xref": concept.get("xref", []),
                "is_a": list2str([c.replace(":", "_") for c in concept["is_a"]]),
            }
        with open(working_dir / ID_TO_CONCEPT_FILE, "w") as f:
            json.dump(concepts, f)

        def get_umls_name(s):
            apikey = os.getenv("UMLS_API_KEY")
            version = "current"

            base_uri = "https://uts-ws.nlm.nih.gov"
            path = "/search/" + version
            query = {"string": s, "apiKey": apikey}
            output = requests.get(base_uri + path, params=query)
            output.encoding = "utf-8"

            output_json = output.json()

            names = []
            if len(output_json["result"]) == 0:
                pass
            else:
                names = [
                    r["name"]
                    for r in output_json["result"]["results"]
                    if r["rootSource"] != "HPO"
                ]
            return names

        if str(working_dir) == "../dict/HPO":
            umls_syns = {}
            if os.path.exists("../data/HPO_UMLS-synonyms.json"):
                with open("../data/HPO_UMLS-synonyms.json", "r") as f:
                    umls_syns = json.load(f)
            concepts = {
                id: concept
                | {
                    "UMLS synonyms": umls_syns[id]["UMLS synonyms"]
                    if id in umls_syns
                    else list2str(get_umls_name(concept["label"]))
                }
                for id, concept in concepts.items()
            }
            # 去除UMLS重复的id
            for id, concept in concepts.items():
                ori_syns = [concept["label"].lower()] + [
                    s.lower() for s in concept["synonyms"].split(" | ")
                ]
                new_umls_syns = [
                    c
                    for c in concept["UMLS synonyms"].split(" | ")
                    if c.lower() not in ori_syns
                ]
                concepts[id]["UMLS synonyms"] = list2str(new_umls_syns)
            with open(working_dir / ID_TO_CONCEPT_FILE, "w") as f:
                json.dump(concepts, f)

    @staticmethod
    def _get_embs(all_names, tokenizer, model, bs=128):
        device = next(model.parameters()).device
        model.eval()
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(all_names), bs):
                toks = tokenizer(
                    all_names[i : i + bs],
                    padding=True,
                    max_length=100,
                    truncation=True,
                    return_tensors="pt",
                )
                toks = {k: v.to(device) for k, v in toks.items()}
                cls_rep = model(**toks)[0][:, 0, :]  # CLS representation
                all_embs.append(cls_rep.cpu().numpy())
        return np.concatenate(all_embs, axis=0)
