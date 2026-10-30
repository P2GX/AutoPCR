from autopcr._ssplit_tokenzier import ssplit_token_pos_lemma
from autopcr._ml_ner import ml_tagging, ml_tagging_batch, ml_tagging_allngram
from autopcr._combine_result import combine_ml_dict
from autopcr._restore_index import restore_index_nest_fn
from autopcr._post_processing import combine_overlap
from autopcr._abbre_resolution import postprocess_abbr


# hybrid method
def bioTag(
    text, biotag_dic, ml_model, onlyLongest=False, abbrRecog=False, Threshold=0.95
):

    #    startTime=time.time()
    ssplit_token = ssplit_token_pos_lemma(text)
    # print(ssplit_token)
    #    print('ssplit token:',time.time()-startTime)

    #    startTime=time.time()
    dict_tsv = biotag_dic.matching(ssplit_token)
    #    print('dict tsv:\n',dict_tsv)
    #    print('dict ner:',time.time()-startTime)

    #    startTime=time.time()
    ml_tsv = ml_tagging(ssplit_token, ml_model, Threshold)
    # print('ml_tsv:\n',ml_tsv)
    #    print('ml ner:',time.time()-startTime)

    #    startTime=time.time()
    combine_tsv = combine_ml_dict(dict_tsv, ml_tsv)
    # combine_tsv=combine_ml_dict_fn(ml_tsv,dict_tsv)
    # print('combine:\n',combine_tsv)

    final_result = restore_index_nest_fn(text, combine_tsv)
    #    print('final ner:',time.time()-startTime)
    if onlyLongest:
        final_result = combine_overlap(final_result)
    if abbrRecog:
        final_result = postprocess_abbr(final_result, text)
    #    print('final result:')
    #    print(final_result)

    return final_result


# hybrid method
def bioTag_batch(
    text_batch, biotag_dic, ml_model, onlyLongest=False, abbrRecog=False, Threshold=0.95
):
    ssplit_token_batch = [ssplit_token_pos_lemma(text) for text in text_batch]
    ml_tsv_batch = ml_tagging_batch(
        ssplit_token_batch, ml_model, Threshold, text_batch=text_batch
    )
    final_result_batch = []
    for i in range(len(text_batch)):
        dict_tsv = biotag_dic.matching(ssplit_token_batch[i])
        combine_tsv = combine_ml_dict(dict_tsv, ml_tsv_batch[i])
        final_result = restore_index_nest_fn(text_batch[i], combine_tsv)
        if onlyLongest:
            final_result = combine_overlap(final_result)
        if abbrRecog:
            final_result = postprocess_abbr(final_result, text_batch[i])
        final_result_batch.append(final_result)
    return final_result_batch


# only machine learning-based method
def bioTag_ml(text, ml_model, onlyLongest=False, abbrRecog=False, Threshold=0.95):

    #    startTime=time.time()
    ssplit_token = ssplit_token_pos_lemma(text)
    #    print(ssplit_token)
    #    print('ssplit token:',time.time()-startTime)

    #    startTime=time.time()
    ml_tsv = ml_tagging_allngram(ssplit_token, ml_model, Threshold)
    #    print('ml_tsv:\n',ml_tsv)
    #    print('ml ner:',time.time()-startTime)

    final_result = restore_index_nest_fn(text, ml_tsv)
    #    print('final ner:',time.time()-startTime)
    if onlyLongest:
        final_result = combine_overlap(final_result)

    if abbrRecog:
        final_result = postprocess_abbr(final_result, text)

    return final_result


# only dict method
def bioTag_dic(text, biotag_dic, onlyLongest=False, abbrRecog=False):

    #    startTime=time.time()
    ssplit_token = ssplit_token_pos_lemma(text)
    #    print(ssplit_token)
    #    print('ssplit token:',time.time()-startTime)

    #    startTime=time.time()
    dict_tsv = biotag_dic.matching(ssplit_token)
    #    print('dict tsv:\n',dict_tsv)
    #    print('dict ner:',time.time()-startTime)

    final_result = restore_index_nest_fn(text, dict_tsv)
    #    print('final ner:',time.time()-startTime)
    if onlyLongest:
        final_result = combine_overlap(final_result)

    if abbrRecog:
        final_result = postprocess_abbr(final_result, text)

    return final_result
