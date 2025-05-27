# AutoPCR: Automated Phenotype Concept Recognition by Prompting


## 🚀 Installation
```
conda env create -f environment.yml
```


## 📁 Data Preparation
The four datasets used in the experiments are available at [BIOC-GS](https://github.com/Ian-Campbell-Lab/Clinical-Genetics-Training-Data/blob/main/Physical-Exam/Clinical-Genetics-Physical-Exam-Data-Dev.tsv), [GSC-2024](https://github.com/tudorgroza/code-for-papers/tree/main/gsc-2024), [ID-68](https://github.com/EclipseCN/PhenoBERT/tree/main/phenobert/data/ID-68), and [NCBI](https://github.com/ncbi-nlp/PhenoTagger/blob/master/data/corpus.zip).
These datasets should be formated according to NCBI and processed by the function src.evaluate.GSCplus_corpus_gold, following PhenoTagger's pipeline. The processed datasets are provided in data/corpus.zip.

The two ontologies used are available at [HPO](https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2024-02-08/hp.obo) and [MEDIC](https://ctdbase.org/reports/CTD_diseases.obo.gz), and also provided in ontology.zip.

Please unzip the files if you need to use them.
```bash
unzip data/corpus.zip
unzip ontology.zip
```

## 🔁 Usage

### Activate environment
```bash
conda activate tagger
cd src
```

### Build the ontology dictionary
```bash
python build_dict.py --input ../ontology/hp_20240208.obo --output ../dict/HPO/ --rootnode HP:0000118
python build_dict.py --input ../ontology/CTD_diseases.obo --output ../dict/MEDIC/ --rootnode MESH:C
```

### Build the ontology index using SapBERT
```bash
python build_index.py --ontology_dict ../dict/HPO
python build_index.py --ontology_dict ../dict/MEDIC
```

### Run experiments
```bash
python HPO_evaluation.py --ontology_dict ../dict/HPO -c BIOC-GS -o ../results/bioc-gs.tsv --llm gpt -ee rule
python HPO_evaluation.py --ontology_dict ../dict/HPO -c GSC-2024 -o ../results/gsc-2024.tsv --llm gpt -ee neural 
python HPO_evaluation.py --ontology_dict ../dict/HPO -c ID-68 -o ../results/id-68.tsv --llm gpt -ee neural --longest
python HPO_evaluation.py --ontology_dict ../dict/MEDIC -c NCBI -o ../results/ncbi.tsv --llm gpt -ee neural
```

## 📚 Acknowledgements
This project is adapted in part from the following repositories:
- [PhenoTagger](https://github.com/ncbi-nlp/PhenoTagger)
- [PhenoBERT](https://github.com/EclipseCN/PhenoBERT)
- [REAL](https://github.com/dash-ka/REAL-BioCR)

We thank the authors for open-sourcing their code.