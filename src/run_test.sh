python HPO_evaluation.py --ontology_dict ../dict/HPO -c BIOC-GS -o ../results/bioc-gs.tsv --llm gpt --ee rule
python HPO_evaluation.py --ontology_dict ../dict/HPO -c GSC-2024 -o ../results/gsc-2024.tsv --llm gpt --ee neural 
python HPO_evaluation.py --ontology_dict ../dict/HPO -c ID-68 -o ../results/id-68.tsv --llm gpt --ee neural --longest
python HPO_evaluation.py --ontology_dict ../dict/MEDIC -c NCBI -o ../results/ncbi.tsv --llm gpt --ee neural