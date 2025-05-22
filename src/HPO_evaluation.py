import argparse
from dic_ner import dic_ont
from tagging_text import bioTag, bioTag_batch
import os
import time
import json
import tensorflow as tf
import logging
from tqdm import tqdm
import nltk
from nn_model2 import bioTag_SapBERT
from evaluate import GSCplus_corpus_hponew as GSCplus_corpus
from evaluate import JAX_corpus_hponew as JAX_corpus

'''
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
session = tf.Session(config=config) 
'''
def run_gsc_test(files,biotag_dic,nn_model,ts,batch=False):
    fin_test=open(files['testfile'],'r',encoding='utf-8')
    all_test=fin_test.read().strip().split('\n\n')
    fin_test.close()
    test_out=open(files['outfile'],'w',encoding='utf-8')

    if batch:
        lines_batch = [doc_test.split('\n') for doc_test in all_test]
        # lines_batch = [lines for lines in lines_batch if lines[0] == '3185841']
        test_result_batch = bioTag_batch([lines[1] for lines in lines_batch],biotag_dic,nn_model,onlyLongest=False,abbrRecog=False,Threshold=ts)
        for lines, test_result in zip(lines_batch, test_result_batch):
            pmid = lines[0]
            test_out.write(pmid+'\n'+lines[1]+'\n')
            for ele in test_result:
                test_out.write(ele[0]+'\t'+ele[1]+'\t'+lines[1][int(ele[0]):int(ele[1])]+'\t'+ele[2]+'\t'+ele[3]+'\n')
            test_out.write('\n')
        test_out.close()
    else:
        for doc_test in tqdm(all_test):
            lines=doc_test.split('\n')
            pmid = lines[0]
            # if pmid != '10712204': # '15DG2492':
            #     continue
            # test_result = []
            # for line_sub in nltk.sent_tokenize(lines[1]):
            #     test_result += bioTag(line_sub,biotag_dic,nn_model,onlyLongest=False,abbrRecog=False,Threshold=ts)
            test_result = bioTag(lines[1],biotag_dic,nn_model,onlyLongest=False,abbrRecog=False,Threshold=ts)
            test_out.write(pmid+'\n'+lines[1]+'\n')
            for ele in test_result:
                test_out.write(ele[0]+'\t'+ele[1]+'\t'+lines[1][int(ele[0]):int(ele[1])]+'\t'+ele[2]+'\t'+ele[3]+'\n')
            test_out.write('\n')
        test_out.close()
    GSCplus_corpus(files['outfile'],files['testfile'],subtree=True)


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ontology', '-onto', choices=['HPO', 'MEDIC'], default='HPO')
    parser.add_argument('--corpus', '-c', choices=['BIOC-GS', 'GSC-2024', 'ID-68', 'NCBI'], default='GSC-2024')
    parser.add_argument('--output', '-o', help="the output prediction file", default='../results/GSC-2024.tsv')
    parser.add_argument('--ts', help="threshold for model", type=float, default=0.95)
    parser.add_argument('--llm', type=str, default='')
    args = parser.parse_args()

    dict_str = f'dict_{args.ontology.lower()}'
    ontfiles={'dic_file':f'../{dict_str}/noabb_lemma.dic',
                'word_hpo_file':f'../{dict_str}/word_id_map.json',
                'hpo_word_file':f'../{dict_str}/id_word_map.json'}
    biotag_dic=dic_ont(ontfiles)

    vocabfiles={'index': f'../{dict_str}/{args.ontology.lower()}_lable.index',
                'index_to_id': f'../{dict_str}/index_to_id_lable.json',
                'id_to_concept': f'../{dict_str}/id_to_concept.json'}

    output_dir, name = os.path.split(args.output)
    base_name, ext = os.path.splitext(name)
    cand_path = os.path.join(output_dir, base_name + '.json')

    nn_model=bioTag_SapBERT(vocabfiles, llm=args.llm, cand_path=cand_path)
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.output+'_log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    files={'testfile': f'../data/corpus/{args.corpus}/{args.corpus}_test_gold.tsv',
           'outfile': args.output}
    
    start_time=time.time()
    run_gsc_test(files,biotag_dic,nn_model,args.ts,batch=(args.llm=='llama'))
    logging.info(f'{args.corpus} done: {time.time()-start_time}')