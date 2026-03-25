# AutoPCR: Automated Phenotype Concept Recognition by Prompting


## 🚀 Installation (Linux)
```bash
conda env create -f environment.yml
```


## 📁 Data Preparation
The four datasets used in the experiments are available at [BIOC-GS](https://github.com/Ian-Campbell-Lab/Clinical-Genetics-Training-Data/blob/main/Physical-Exam/Clinical-Genetics-Physical-Exam-Data-Dev.tsv), [GSC-2024](https://github.com/tudorgroza/code-for-papers/tree/main/gsc-2024), [ID-68](https://github.com/EclipseCN/PhenoBERT/tree/main/phenobert/data/ID-68), and [NCBI](https://github.com/ncbi-nlp/PhenoTagger/blob/master/data/corpus.zip).
These datasets should be formatted according to NCBI and processed by the function src.evaluate.GSCplus_corpus_gold, following PhenoTagger's pipeline. The processed datasets are provided in data/corpus.zip.

The two ontologies used are available at [HPO](https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2024-02-08/hp.obo) and [MEDIC](https://ctdbase.org/reports/CTD_diseases.obo.gz), and also provided in ontology.zip.

Please unzip the files if you need to use them.
```bash
unzip data/corpus.zip -d data/
unzip ontology.zip
```

## 🔁 Usage

### Activate environment
```bash
conda activate autopcr
cd src
```

### Build the ontology dictionary
Parameters:
- --input, -i, help='input the ontology .obo file', default='../ontology/hp_20240208.obo'
- --output, -o, help='the output path of dictionary', default='../dict/HPO'
- --rootnode, -r, help='input the root node of the ontology', nargs='+', default=['HP:0000118']

Examples:
```bash
python build_dict.py -i ../ontology/hp_20240208.obo  -o ../dict/HPO   -r HP:0000118
python build_dict.py -i ../ontology/CTD_diseases.obo -o ../dict/MEDIC -r MESH:C
```

### Build the ontology index using SapBERT
Parameters:
- --ontology_dict, help='folder of the ontology dictionary', type=str, required=True
- --ccr, help='candidate concept retrieval model', type=str, default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext'

Examples:
```bash
python build_index.py --ontology_dict ../dict/HPO
python build_index.py --ontology_dict ../dict/MEDIC
```

### Run experiments
Parameters:
- --ontology_dict, help='folder of the ontology dictionary', type=str, required=True
- --corpus, -c, help='input corpus dataset', type=str, required=True
- --output, -o, help='output prediction file', type=str, required=True
- --doc_id, help='specific docs to test', nargs='*', type=str
- --ee, help='entity extraction method: (1) "rule": PhenoTagger-style rule-based, (2) "neural": BioNER, (3) "neural+": BioNER + benepar, (4) "neural++": BioNER + benepar + coordination decomposition', choices=['rule', 'neural', 'neural+', 'neural++'], type=str, default='neural++'
- --abbr_recog, help='whether to identify abbreviations', action='store_true'
- --ccr, help='candidate concept retrieval model', type=str, default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
- --tau_1, help='high-confidence threshold for candidate concept retrieval', type=float, default=0.95
- --tau_2, help='low-confidence threshold for candidate concept retrieval', type=float, default=0.85
- --k, help='number of retrieved candidate concepts', type=int, default=5

- --el, help='entity linking llm backend; "none" to disable', type=str, default='Qwen/Qwen3-Next-80B-A3B-Instruct'
- --api_provider, help='api provider for llm inference', choices=['openai', 'groq', 'together', 'vllm'], type=str, default='together'
- --api_key, help='api key for api provider, api url if api provider is vllm, or loaded from .env (e.g., TOGETHER_API_KEY) if not provided', type=str
- --use_finetuning_prompt, help='simplified prompt for finetuned llm', action='store_true'
- --seed, help='seed for llm', type=int, default=0
- --use_cache, help='whether to cache results', action='store_true'

- --only_longest, help='whether the output only keeps the longest nested entity', action='store_true'

Examples:
```bash
python HPO_evaluation.py --ontology_dict ../dict/HPO   -c BIOC-GS  -o ../results/bioc-gs.tsv  --only_longest
python HPO_evaluation.py --ontology_dict ../dict/HPO   -c GSC-2024 -o ../results/gsc-2024.tsv
python HPO_evaluation.py --ontology_dict ../dict/HPO   -c ID-68    -o ../results/id-68.tsv    --only_longest
python HPO_evaluation.py --ontology_dict ../dict/MEDIC -c NCBI     -o ../results/ncbi.tsv     --only_longest --abbr_recog
```

### (Optional) Fine-tune the entity linker
First, build the training data:
```bash
python build_training_dataset.py
```
Then, install [unsloth](https://unsloth.ai/docs/get-started/install) in a new environment and use it to fine-tune the entity linker:
```bash
python unsloth_finetune_30b.py
```
To use the fine-tuned model, you may want to serve it using vllm and additionally set "--api_provider vllm --api_key <Your_API_URL> --use_finetuning_prompt" in the last step. A fine-tuned model is also provided [here](https://huggingface.co/yctao/AutoPCR_entitylinker).

### Pre-computed results
We provide the pre-computed results of AutoPCR in the `results/submission/` folder and those of baselines in the [Google Drive](https://drive.google.com/drive/folders/18UMDsRJi5ELhb-FNAEsUjLy5-ZwzFhP_?usp=sharing).


## 📚 Acknowledgements
This project is adapted in part from the following repositories:
- [PhenoTagger](https://github.com/ncbi-nlp/PhenoTagger)
- [PhenoBERT](https://github.com/EclipseCN/PhenoBERT)
- [REAL](https://github.com/dash-ka/REAL-BioCR)

We thank the authors for open-sourcing their code.
