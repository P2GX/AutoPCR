import spacy
import benepar
from scispacy.abbreviation import AbbreviationDetector
from collections import defaultdict
from .utils import *

SPACY_MODEL = 'en_core_web_trf'
BENEPAR_MODEL = 'benepar_en3_large'
SCISPACY_MODEL = 'en_core_sci_scibert'

try:
    clinical_ner_model = spacy.load(SPACY_MODEL)
except OSError:
    spacy.cli.download(SPACY_MODEL)
    clinical_ner_model = spacy.load(SPACY_MODEL)

try:
    clinical_ner_model.add_pipe('benepar', config={'model': BENEPAR_MODEL})
except LookupError:
    benepar.download(BENEPAR_MODEL)
    clinical_ner_model.add_pipe('benepar', config={'model': BENEPAR_MODEL})

try:
    abbr_model = spacy.load(SCISPACY_MODEL)
    abbr_model.add_pipe('abbreviation_detector')
except OSError:
    os.system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz')
    abbr_model = spacy.load(SCISPACY_MODEL)
    abbr_model.add_pipe('abbreviation_detector')


# def generate_spans(sent, tags=['NP', 'UCP', 'NML', 'NX', 'NN', 'NNS', 'NNP', 'NNPS']):
#     return [s for s in sent._.constituents if (s._.labels[0] if s._.labels else s.root.tag_) in tags]
def generate_spans(sent, tags=['NP', 'UCP', 'NML', 'NX', 'NN', 'NNS', 'NNP', 'NNPS']):
    def get_tag(s):
        if not s:
            return ''
        return s._.labels[0] if s._.labels else s.root.tag_
    def skip_tokens(s, tokens, left):
        if left:
            return s if s[0].text not in tokens else s[1:]
        else:
            return s if s[-1].text not in tokens else s[:-1]
    spans = []
    for s in sent._.constituents:
        if get_tag(s) in tags and sent.doc[s.start-1:s.start].text != '-' and sent.doc[s.end:s.end+1].text != '-':
            if not (s:=skip_tokens(s, [',', '.'], left=True)):
                continue
            if not (s:=skip_tokens(s, [',', '.'], left=False)):
                continue
            if not (s:=skip_tokens(s, ['A', 'a', 'An', 'an', 'the', 'The'], left=True)): # s[1:] if s[0].text in ['A', 'a', 'An', 'an', 'the', 'The'] else s):
                continue
            if len(s) == 1 and s.root.tag_ == 'PRP':
                continue
            spans.append(s)
            # check if all modifier before
            # if s.root.i > s.start:
            #     all_modifier = True
            #     for j in range(s.root.i-s.start):
            #         if s[j].pos_ not in ['ADJ', 'ADV', 'NOUN', 'PROPN']:
            #             all_modifier = False
            #             break
            #     if all_modifier:
            #         for j in range(s.root.i-s.start):
            #             spans.append(s[j+1:])
            # remove word headed at root
            # for j in range(s.root.i-s.start):
            #     if s[j].is_punct:
            #         continue
            #     if s[j].head == s.root:
            #         if s_:=skip_punct(s[j+1:]):
            #             spans.append(s_)
            #     else:
            #         break
    return spans


def get_phrases_by_root(sent):
    root2phrases = defaultdict(list)
    for s in sent._.constituents:
        label = s._.labels[0] if s._.labels else ''
        if label and not label.endswith('S'):
            root2phrases[s.root].append(s)
    for root, phrases in root2phrases.items():
        phrases.sort(key=len)
    return dict(root2phrases)


def get_conjunct_groups(sent):
    groups = []
    for token in sent:
        if token.dep_ == 'conj':
            continue
        conjuncts = list(token.conjuncts)
        if conjuncts:
            conjuncts = [token] + conjuncts
            groups.append(conjuncts)
    # add tokens(-) besides /
    for token in sent:
        if token.text == '/' and sent.start < token.i < sent.end-1:
            if sent.start+1 < token.i < sent.end-2 and sent.doc[token.i-1].text == '-' == sent.doc[token.i+2].text:
                groups.append([sent.doc[token.i-2], sent.doc[token.i+1]])
            else:
                groups.append([sent.doc[token.i-1], sent.doc[token.i+1]])
    return groups


def get_conjunct_phrase_groups(root2phrases, groups):
    phrase_groups = []
    for group in groups:
        phrase_group = []
        for i, token in enumerate(group):
            phrase_curr = token.doc[token.i:token.i+1]
            for phrase in root2phrases.get(token, []):
                overlap = False
                for conjunct in group[:i] + group[i+1:]:
                    if phrase.start <= conjunct.i < phrase.end:
                        overlap = True
                        break
                if overlap:
                    break
                phrase_curr = phrase
            phrase_group.append(phrase_curr)
        phrase_groups.append(phrase_group)
    return phrase_groups


# def split_spans_on_conjunctions(spans, phrase_groups, conjunct_groups):
#     spans_conjunct = []
#     for phrase_group in phrase_groups:
#         left = min(phrase.start for phrase in phrase_group)
#         right = max(phrase.end for phrase in phrase_group)
#         left_char = min(phrase.start_char for phrase in phrase_group)
#         right_char = max(phrase.end_char for phrase in phrase_group)
#         for s in spans:
#             if s.start <= left and s.end >= right:
#                 if s.start == left and s.end == right:
#                     for phrase in phrase_group:
#                         spans_conjunct.append((phrase.start_char, phrase.end_char, phrase))
#                 else:
#                     for phrase in phrase_group:
#                         s_conjunct = list(s[:left-s.start]) + list(phrase) + list(s[right-s.start:])
#                         spans_conjunct.append((s.start_char, s.end_char, s_conjunct))
#                         # if s.start < left and s.end > right:
#                         #     s_conjunct = list(s[:left-s.start]) + list(phrase)
#                         #     spans_conjunct.append((s.start_char, right_char, s_conjunct))
#                         #     s_conjunct = list(phrase) + list(s[right-s.start:])
#                         #     spans_conjunct.append((left_char, s.end_char, s_conjunct))
#     return spans_conjunct


def split_spans_on_conjunctions(spans, phrase_groups, conjunct_groups):
    def skip_bracket(s):
        if s and s[0].tag_ in ['-LRB-'] and len([t for t in s[1:] if t.tag_ in ['-RRB-']]) == 0:
            return s[1:]
        if s and s[-1].tag_ in ['-RRB-'] and len([t for t in s[:-1] if t.tag_ in ['-LRB-']]) == 0:
            return s[:-1]
        return s
    spans_conjunct = []
    for phrase_group, conjunct_group in zip(phrase_groups, conjunct_groups):
        for s in spans:
            phrases_cover = [p for p in phrase_group if s.start <= p.start and p.end <= s.end]
            conjunct_cover = [p for p in conjunct_group if s.start <= p.i < s.end]
            if len(phrases_cover) >= 2 and len(phrases_cover) == len(conjunct_cover):
                left = min(p.start for p in phrases_cover)
                right = max(p.end for p in phrases_cover)
                left_char = min(p.start_char for p in phrases_cover)
                right_char = max(p.end_char for p in phrases_cover)
                # if s.start == left and s.end == right:
                #     for phrase in phrases_cover:
                #         spans_conjunct.append((phrase.start_char, phrase.end_char, phrase))
                # else:
                if s.start < left or s.end > right:
                    for phrase in phrases_cover:
                        s_conjunct = list(s[:left-s.start]) + list(phrase) + list(s[right-s.start:])
                        spans_conjunct.append((s.start_char, s.end_char, s_conjunct))
                        if s.start < left and s.end > right:
                            s_left = skip_bracket(s[:left-s.start])
                            s_conjunct = list(s_left) + list(phrase)
                            spans_conjunct.append((s_left.start_char, right_char, s_conjunct))
                            s_right = skip_bracket(s[right-s.start:])
                            s_conjunct = list(phrase) + list(s_right)
                            spans_conjunct.append((left_char, s_right.end_char, s_conjunct))
    return spans_conjunct


def generate_spans_conjunct(spans, sent):
    root2phrases = get_phrases_by_root(sent)
    # print(root2phrases)
    conjunct_groups = get_conjunct_groups(sent)
    # print(conjunct_groups)
    conjunct_phrase_groups = get_conjunct_phrase_groups(root2phrases, conjunct_groups)
    # print(conjunct_phrase_groups)
    spans_conjunct = split_spans_on_conjunctions(spans, conjunct_phrase_groups, conjunct_groups)
    # print(spans_conjunct)
    return spans_conjunct


def iterate_spans(spans):
    new_spans = []
    for s in spans:
        # check if all modifier before
        if s.root.i > s.start:
            all_modifier = True
            for j in range(s.root.i-s.start):
                if s[j].pos_ not in ['ADJ', 'ADV', 'NOUN', 'PROPN']:
                    all_modifier = False
                    break
            if all_modifier:
                for j in range(s.root.i-s.start):
                    new_spans.append(s[j+1:])
    return new_spans


def process_text2phrases(text, clinical_ner_model):
    """
    用于从文本中提取Clinical Text Segments
    :param text:自由文本
    :param clinical_ner_model: Stanza提供的预训练NER模型
    :return: List[PhraseItem]
    """
    # # 将文本处理成正常的小写形式
    # text = strip_accents(text.lower())
    # text = re.sub("[-_\"\'\\\\\t‘’]", " ", text)
    # # 对于换行符替换为句号，后续作为分割词
    # text = re.sub("(?<=[\w])[\r\n]", ".", text)

    all_spans = []
    all_spans_conjunct = []

    doc = clinical_ner_model(text)
    for sent in doc.sents:
        spans = generate_spans(sent)
        all_spans += spans
        spans_conjunct = generate_spans_conjunct(spans, sent)
        all_spans_conjunct += spans_conjunct

    # 对phrase_item进行简化，去除常用词以及替换数字
    all_spans = [(s.start_char, s.end_char, s.text, PhraseItem._toSimpleString(s)) for s in all_spans]
    all_spans_conjunct = [(s[0], s[1], PhraseItem._toString(s[2]), PhraseItem._toSimpleString(s[2])) for s in all_spans_conjunct if len(s[2]) <= 10]

    return all_spans, all_spans_conjunct


def process_text2phrases_benepar(text):
    # text = re.sub("(?<=[A-Z])-(?=[\d])", "", text)
    # text = ' '.join(text.split()) # https://github.com/nikitakit/self-attentive-parser/issues/75
    text, index_map = normalize_with_char_map(text)
    phrases_list = process_text2phrases(text, clinical_ner_model)
    return text, phrases_list, index_map


def process_text2phrases_abbr(text):
    # 很多abbr在truth里没标记
    text, index_map = normalize_with_char_map(text)
    phrases_list = [(abbr.start_char, abbr.end_char, abbr.text, abbr._.long_form.text, PhraseItem._toSimpleString(abbr._.long_form)) for abbr in abbr_model(text)._.abbreviations]
    return text, phrases_list, index_map