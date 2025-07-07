# AutoPCR: Automated Phenotype Concept Recognition by Prompting


## 🚀 Installation
```bash
conda env create -f environment.yml
```


## 📁 Data Preparation
The four datasets used in the experiments are available at [BIOC-GS](https://github.com/Ian-Campbell-Lab/Clinical-Genetics-Training-Data/blob/main/Physical-Exam/Clinical-Genetics-Physical-Exam-Data-Dev.tsv), [GSC-2024](https://github.com/tudorgroza/code-for-papers/tree/main/gsc-2024), [ID-68](https://github.com/EclipseCN/PhenoBERT/tree/main/phenobert/data/ID-68), and [NCBI](https://github.com/ncbi-nlp/PhenoTagger/blob/master/data/corpus.zip).
These datasets should be formated according to NCBI and processed by the function src.evaluate.GSCplus_corpus_gold, following PhenoTagger's pipeline. The processed datasets are provided in data/corpus.zip.

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
- --input, -i, help="ontology .obo file"
- --output, -o, help="output folder of the dictionary"
- --rootnode, -r, help="root node of the ontology"

Examples:
```bash
python build_dict.py -i ../ontology/hp_20240208.obo -o ../dict/HPO/ -r HP:0000118
python build_dict.py -i ../ontology/CTD_diseases.obo -o ../dict/MEDIC/ -r MESH:C
```

### Build the ontology index using SapBERT
Parameters:
- --ontology_dict, help="folder of the ontology dictionary"

Examples:
```bash
python build_index.py --ontology_dict ../dict/HPO
python build_index.py --ontology_dict ../dict/MEDIC
```

### Run experiments
Paramaters:
- --ontology_dict, help="folder of the ontology dictionary"
- --corpus, -c, help="input corpus dataset"
- --output, -o, help="output prediction file"
- --ee, help="entity extraction method", choices=['rule', 'neural']
- --tau_1, help="high-confidence threshold for candidate concept retrieval", default=0.95
- --tau_2, help="low-confidence threshold for candidate concept retrieval", default=0.85
- --k, help="number of retrieved candidate concepts", default=5
- --llm, help="which llm (OpenAI or Groq models) to use for entity linking, or skip for no entity linking"
- --api_key, help="OpenAI or Groq api key, or load from .env"
- --seed, help="seed for llm", default=0
- --longest, help="whether the output only keeps the longest nested entity"

Examples:
```bash
python HPO_evaluation.py --ontology_dict ../dict/HPO   -c BIOC-GS  -o ../results/bioc-gs.tsv  --ee rule   --llm gpt-4o-mini
python HPO_evaluation.py --ontology_dict ../dict/HPO   -c GSC-2024 -o ../results/gsc-2024.tsv --ee neural --llm gpt-4o-mini
python HPO_evaluation.py --ontology_dict ../dict/HPO   -c ID-68    -o ../results/id-68.tsv    --ee neural --llm gpt-4o-mini --longest
python HPO_evaluation.py --ontology_dict ../dict/MEDIC -c NCBI     -o ../results/ncbi.tsv     --ee neural --llm gpt-4o-mini
```

## 📚 Acknowledgements
This project is adapted in part from the following repositories:
- [PhenoTagger](https://github.com/ncbi-nlp/PhenoTagger)
- [PhenoBERT](https://github.com/EclipseCN/PhenoBERT)
- [REAL](https://github.com/dash-ka/REAL-BioCR)

We thank the authors for open-sourcing their code.
