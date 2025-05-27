import nltk
import re
import os
import unicodedata
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import stanza

project_path = os.path.dirname(__file__)+"/"

# file path definition
stopwords_file_path = project_path + "../data/stopwords.txt"
num2word_file_path = project_path + "../data/NUM.txt"
wnl = WordNetLemmatizer()

clinical_ner_model = stanza.Pipeline('en', package='mimic', processors={'ner': 'i2b2'}, verbose=False)


class WordItem:
    """
    英文单词的包装类
    """
    # 用于lemmatize
    lemma_dict = {}
    def __init__(self, text, start, end):
        self.text = text.lower()
        self.start = start
        self.end = end


def getNum2Word(file_path):
    Num2Word = {}
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            inter = line.strip().split("\t")
            Num2Word[inter[0]] = inter[1]
    return Num2Word


class PhraseItem:
    """
    英文短语的包装类，包含了文本信息和起止点信息
    """

    Num2Word = getNum2Word(num2word_file_path)
    StopWords = stopwords.words("english")

    def __init__(self, word_items):
        self.word_items = word_items
        self.simple_items = []
        self.simplify()
        self.locs_set = set([i.start for i in word_items])
        self.start_loc = self.word_items[0].start
        self.end_loc = self.word_items[-1].end
        self.no_flag = False

    def simplify(self):
        """
        对phrase_item进行简化，去除常用词以及替换数字
        :return:
        """
        for word_item in self.word_items:
            if word_item.text in PhraseItem.Num2Word:
                self.simple_items.append(WordItem(PhraseItem.Num2Word[word_item.text], word_item.start, word_item.end))
            elif word_item.text in PhraseItem.StopWords or isNum(word_item.text):
                continue
            else:
                self.simple_items.append(word_item)

    def toString(self):
        return " ".join([i.text for i in self.word_items])

    def toSimpleString(self):
        return " ".join([i.text for i in self.simple_items])

    def include(self, phrase_item):
        if self.locs_set.issubset(phrase_item.locs_set) or self.locs_set.issuperset(phrase_item.locs_set):
            return True
        return False

    def issubset(self, phrase_item):
        if self.locs_set.issubset(phrase_item.locs_set):
            return True
        return False

    def set_no_flag(self):
        self.no_flag = True

    def __len__(self):
        return len(self.word_items)


class SpanTokenizer:
    """
    基于NLTK工具包，自定义一个详细版本的Tokenizer
    """

    def __init__(self):
        self.tokenizer_big = PunktSentenceTokenizer()
        self.tokenizer_small = TreebankWordTokenizer()

    def tokenize(self, text):
        result = []
        sentences_span = self.tokenizer_big.span_tokenize(text)
        for start, end in sentences_span:
            sentence = text[start:end]
            tokens_span = self.tokenizer_small.span_tokenize(sentence)
            for token_start, token_end in tokens_span:
                result.append([start + token_start, start + token_end])
        return result


def strip_accents(s):
    """
    去除口音化，字面意思
    :param s:
    :return:
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def isNum(strings):
    """
    判断给定字符串是否为数字
    :param strings:
    :return:
    """
    try:
        float(strings)
        return True
    except ValueError:
        return False


def getStopWords():
    """
    返回stopwords_file_path给定的stop words
    :return:
    """
    stopwords = set()
    with open(stopwords_file_path, encoding="utf-8") as file:
        for line in file:
            stopwords.add(line.strip())
    return stopwords


def getSpliters():
    """
    用于分割短句的分割词
    :return:
    """
    spliters = set([word for word, pos in nltk.pos_tag(stopwords.words('english')) if pos in {'CC', 'WP', 'TO', 'WDT'}]+[',', '.', ':', ';', '(', ')', '[', ']', '/'])
    return spliters


def process_text2phrases(text, clinical_ner_model):
    """
    用于从文本中提取Clinical Text Segments
    :param text:自由文本
    :param clinical_ner_model: Stanza提供的预训练NER模型
    :return: List[PhraseItem]
    """
    tokenizer = SpanTokenizer()
    spliters = getSpliters()
    stopwords = getStopWords()
    # 将文本处理成正常的小写形式
    text = strip_accents(text.lower())
    text = re.sub("[-_\"\'\\\\\t‘’]", " ", text)
    # 对于换行符替换为句号，后续作为分割词
    text = re.sub("(?<=[\w])[\r\n]", ".", text)

    clinical_docs = clinical_ner_model(text)
    sub_sentences = []

    for sent_c in clinical_docs.sentences:
        clinical_tokens = sent_c.tokens

        # Stanza
        flag = False
        curSentence = []
        tmp = set()
        for i in range(len(clinical_tokens)):
            wi = WordItem(clinical_tokens[i].text, clinical_tokens[i].start_char, clinical_tokens[i].end_char)
            if "PROBLEM" in clinical_tokens[i].ner and wi.text not in {',', '.', ':', ';', '(', ')', '[', ']'}:
                curSentence.append(wi)
            else:
                if len(curSentence) > 0:
                    phrase_item = PhraseItem(curSentence)
                    sub_sentences.append(phrase_item)
                    tmp.update(phrase_item.locs_set)
                    flag = True
                curSentence = []

        if len(curSentence) > 0:
            phrase_item = PhraseItem(curSentence)
            sub_sentences.append(phrase_item)
            tmp.update(phrase_item.locs_set)
            flag = True

        # phrase segmentation
        # 只有Stanza标记的句子才作补充
        if not flag:
            continue
        curSentence = []
        for i in range(len(clinical_tokens)):
            wi = WordItem(clinical_tokens[i].text, clinical_tokens[i].start_char, clinical_tokens[i].end_char)
            # 用于后续lemma比对
            text_lemma = wnl.lemmatize(clinical_tokens[i].text)
            if clinical_tokens[i].text not in WordItem.lemma_dict:
                WordItem.lemma_dict[clinical_tokens[i].text] = text_lemma
            if clinical_tokens[i].text in spliters:
                if len(curSentence) > 0:
                    phrase_item = PhraseItem(curSentence)
                    # 只有不与Stanza重叠的部分才加入
                    if len(phrase_item.locs_set & tmp) == 0:
                        sub_sentences.append(phrase_item)
                curSentence = []
            else:
                curSentence.append(wi)

        if len(curSentence) > 0:
            phrase_item = PhraseItem(curSentence)
            if len(phrase_item.locs_set & tmp) == 0:
                sub_sentences.append(phrase_item)

    # 否定检测
    for phrase_item in sub_sentences:
        flag = False
        for token in phrase_item.word_items:
            if token.text.lower() in {"no", "not", "none", "negative", "non", "never", "few", "lower", "fewer", "less",
                                      "normal"}:
                flag = True
                break
        if flag:
            phrase_item.set_no_flag()

    # 省略恢复
    sub_sentences_ = []
    for idx, pi in enumerate(sub_sentences):
        # 将含有and, or, / 的短句用tokenize进行拆分
        sub_locs = [[i + pi.start_loc, j + pi.start_loc] for i, j in
                    tokenizer.tokenize(text[pi.start_loc:pi.end_loc])]
        sub_phrases = []
        curr_phrase = []
        # 把以and, or, / 分割的短语提出
        for loc in sub_locs:
            wi = WordItem(text[loc[0]:loc[1]], loc[0], loc[1])
            if wi.text in {"and", "or", "/"}:
                if len(curr_phrase) > 0:
                    sub_phrases.append(PhraseItem(curr_phrase))
                    sub_phrases[-1].no_flag = pi.no_flag
                curr_phrase = []
            else:
                curr_phrase.append(wi)
        if len(curr_phrase) > 0:
            sub_phrases.append(PhraseItem(curr_phrase))
            sub_phrases[-1].no_flag = pi.no_flag

        # 首先把原始分割的短语都加进去
        for item in sub_phrases:
            sub_sentences_.append(item)

        # 只考虑A+B形式的恢复
        if len(sub_phrases) == 2:
            if len(sub_phrases[0]) >= 1 and len(sub_phrases[1]) == 1:
                tmp = sub_phrases[0].word_items[:-1][:]
                tmp.extend(sub_phrases[1].word_items)
                sub_sentences_.append(PhraseItem(tmp))
                sub_sentences_[-1].no_flag = pi.no_flag
            elif len(sub_phrases[0]) == 1 and len(sub_phrases[1]) >= 1:
                tmp = sub_phrases[0].word_items[:]
                tmp.extend(sub_phrases[1].word_items[1:])
                sub_sentences_.append(PhraseItem(tmp))
                sub_sentences_[-1].no_flag = pi.no_flag

    sub_sentences = sub_sentences_

    # print([i.toSimpleString() for i in sub_sentences])

    # 穷举短语 删除纯数字
    phrases_list = []
    for pi in sub_sentences:
        tmp = pi.toSimpleString()
        if isNum(tmp) or len(tmp) <= 1:
            continue
        for i in range(len(pi.simple_items)):
            for j in range(10):
                if i + j == len(pi.simple_items):
                    break
                if len(pi.simple_items[i:i + j + 1]) == 1:
                    tmp_str = pi.simple_items[i:i + j + 1][0].text
                    if tmp_str in stopwords or isNum(tmp_str):
                        continue
                phrases_list.append(PhraseItem(pi.simple_items[i:i + j + 1]))
                phrases_list[-1].no_flag = pi.no_flag

    # print(len(phrases_list))
    # print([i.toString() for i in phrases_list])
    return phrases_list


def process_text2phrases_pb(text):
    text = re.sub("(?<=[A-Z])-(?=[\d])", "", text)
    phrases_list = process_text2phrases(text, clinical_ner_model)
    return text, phrases_list