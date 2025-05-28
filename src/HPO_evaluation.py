import argparse
from dic_ner import dic_ont
from tagging_text import bioTag, bioTag_batch
import os
import time
import logging
from tqdm import tqdm
from nn_model2 import bioTag_SapBERT
from evaluate import GSCplus_corpus
from dotenv import load_dotenv
from utils_pbert_ee import process_text2phrases_pb

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
    GSCplus_corpus(files['outfile'],files['testfile'],files['ontology_dict'],subtree=True)


def run_gsc_test_ner(files,nn_model,ts,disjoint):
    fin_test=open(files['testfile'],'r',encoding='utf-8')
    all_test=fin_test.read().strip().split('\n\n')
    fin_test.close()
    test_out=open(files['outfile'],'w',encoding='utf-8')

    phrasefile = '/'.join(testfile.split('/')[:-1]) + '/phrases.json'
    if os.path.exists(phrasefile):
        all_phrases = json.load(phrasefile, 'r')
    else:
        all_phrases = {}
        for doc_test in tqdm(all_test):
            lines=doc_test.split('\n')
            text, phrases_list = process_text2phrases_pb(lines[1])
            all_phrases[lines[0]] = {'text_new': text}
            all_phrases[lines[0]]['phrases'] = [{'start': int(p.start_loc), 'end': int(p.end_loc), 'phrase': text[int(p.start_loc):int(p.end_loc)], 'phrase_nostopword': p.toString()} for p in phrases_list]
        json.dump(all_phrases, open(phrasefile, 'w'))

    for doc_test in tqdm(all_test):
        lines=doc_test.split('\n')
        pmid = lines[0]
        annotations = nn_model.predict_llm([phrase['phrase'] for phrase in all_phrases[pmid]['phrases']], lines[1] * len(all_phrases[pmid]['phrases']))
        test_out.write(pmid+'\n'+lines[1]+'\n')
        
        anns = []
        for phrase, ele in zip(all_phrases[pmid]['phrases'], annotations.split('\n')):
            start, end = phrase['start'], phrase['end']
            uri, score = ele.split('\t')[0].split('|')
            score = min(float(score), 1.0)
            if abs(score) > ts:
                anns.append((str(start), str(end), phrase['phrase'], uri, str(score)))
        anns = list({(ann[0], ann[1]): ann for ann in anns}.values()) # remove duplicates in phrases
        
        anns_new = []
        for i in range(len(anns)):
            overlap = False
            for j in range(len(anns)): # outer / inner
                if i != j and int(anns[j][0]) <= int(anns[i][0]) and int(anns[i][1]) <= int(anns[j][1]) and int(anns[i][1]) - int(anns[i][0]) < int(anns[j][1]) - int(anns[j][0]) and anns[i][3] == anns[j][3]:
                # if i != j and int(anns[i][0]) <= int(anns[j][0]) and int(anns[j][1]) <= int(anns[i][1]) and int(anns[j][1]) - int(anns[j][0]) < int(anns[i][1]) - int(anns[i][0]) and anns[i][3] == anns[j][3]:
                    overlap = True
                    break
            if not overlap:
                anns_new.append(anns[i])
        
        for ann in anns_new:
            test_out.write(ann[0]+'\t'+ann[1]+'\t'+ann[2]+'\t'+ann[3]+'\t'+ann[4]+'\n')
        test_out.write('\n')
    test_out.close()

    # only keep longest overlapped entities
    if disjoint:
        all_lines = []
        for line in open(files['outfile'], 'r').read().strip().split('\n\n'):
            line = line.split('\n')
            anns = [ann.split('\t') for ann in line[2:]]
            anns_new = []
            for i in range(len(anns)):
                overlap = False
                for j in range(len(anns)):
                    if i != j and int(anns[j][0]) <= int(anns[i][0]) and int(anns[i][1]) <= int(anns[j][1]) and int(anns[i][1]) - int(anns[i][0]) < int(anns[j][1]) - int(anns[j][0]):
                        overlap = True
                        break
                if not overlap:
                    anns_new.append(anns[i])
            line_new = '\n'.join(line[:2] + ['\t'.join(ann) for ann in anns_new])
            all_lines.append(line_new)
        all_lines = '\n\n'.join(all_lines)
        open(files['outfile'], 'w').write(all_lines)

    GSCplus_corpus(files['outfile'],files['testfile'],files['ontology_dict'],subtree=True)


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--ontology_dict', help="folder of the ontology dictionary", type=str, required=True)
    parser.add_argument('--corpus', '-c', help="input corpus dataset", type=str, required=True)
    parser.add_argument('--output', '-o', help="output prediction file", type=str, required=True)
    parser.add_argument('--tau_1', help="high-confidence threshold for model", type=float, default=0.95)
    parser.add_argument('--tau_2', help="low-confidence threshold for model", type=float, default=0.85)
    parser.add_argument('--llm', help="which llm to use for entity linking, or skip for no entity linking", choices=['gpt', 'llama'], type=str)
    parser.add_argument('--api_key', help="api key for OpenAI API, or load from .env", type=str)

    parser.add_argument('--ee', help="entity extraction method", choices=['rule', 'neural'], type=str, required=True)
    parser.add_argument('--longest', help="whether the output only keeps the longest nested entity", action='store_true')
    
    args = parser.parse_args()

    ontfiles={'dic_file':f'{args.ontology_dict}/noabb_lemma.dic',
                'word_hpo_file':f'{args.ontology_dict}/word_id_map.json',
                'hpo_word_file':f'{args.ontology_dict}/id_word_map.json'}
    biotag_dic=dic_ont(ontfiles)

    vocabfiles={'index': f'{args.ontology_dict}/main_lable.index',
                'index_to_id': f'{args.ontology_dict}/index_to_id_lable.json',
                'id_to_concept': f'{args.ontology_dict}/id_to_concept.json'}

    # output_dir, name = os.path.split(args.output)
    # base_name, ext = os.path.splitext(name)
    # cand_path = os.path.join(output_dir, base_name + '.json')

    load_dotenv()
    api_key = args.api_key if args.api_key else os.getenv('OPENAI_API_KEY')

    nn_model=bioTag_SapBERT(vocabfiles, args.tau_1, args.tau_2, args.llm, api_key)
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.output+'_log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    files={'testfile': f'../data/corpus/{args.corpus}/{args.corpus}_test_gold.tsv',
           'outfile': args.output,
           'ontology_dict': args.ontology_dict}
    
    start_time=time.time()
    if args.ee == 'rule':
        run_gsc_test(files,biotag_dic,nn_model,args.tau_1,batch=(args.llm=='llama'))
    elif args.ee == 'neural':
        if args.longest:
            run_gsc_test_ner(files,nn_model,args.tau_1,disjoint=True)
        else:
            run_gsc_test_ner(files,nn_model,args.tau_1,disjoint=False)
    logging.info(f'{args.corpus} done: {time.time()-start_time}')