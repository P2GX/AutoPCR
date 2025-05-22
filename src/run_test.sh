# mamba activate tagger

# GSC24
# python HPO_evaluation.py -m sapbert -c gsc24 -o ../results/gsc24_sapbert_95.tsv --ts 0.95
# python HPO_evaluation.py -m sapbert -c gsc24 -o ../results/gsc24_sapbert_95_llama.tsv --ts 0.95 --llm llama
# python HPO_evaluation.py -m sapbert -c gsc24 -o ../results/gsc24_sapbert_8595_gpt_test.tsv --ts 0.95 --llm gpt

# BIO-GS
# python HPO_evaluation.py -m sapbert -c biogs -o ../results/biogs_sapbert.tsv --ts 0.95 --llm gpt

# ID-68
# python HPO_evaluation.py -m sapbert -c id68 -o ../results/id68_sapbert_95_gpt85_tt25.tsv --ts 0.95 --llm gpt
