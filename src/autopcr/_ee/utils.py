import nltk
import os
import unicodedata
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re

from autopcr._stopwords import stopwords

project_path = os.path.dirname(__file__) + "/../"

wnl = WordNetLemmatizer()


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


def get_num_2_word() -> dict[str, str]:
    return {
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "10": "ten",
        "11": "eleven",
        "12": "twelve",
        "13": "thirteen",
        "14": "fourteen",
        "15": "fifteen",
        "16": "sixteen",
        "17": "seventeen",
        "18": "eighteen",
        "19": "nineteen",
        "20": "twenty",
        "30": "thirty",
        "40": "forty",
        "50": "fifty",
        "60": "sixty",
        "70": "seventy",
        "80": "eighty",
        "90": "ninety",
        "100": "hundred",
        "1st": "first",
        "2nd": "second",
        "3rd": "third",
        "4th": "fourth",
        "5th": "fifth",
        "6th": "sixth",
        "7th": "seventh",
        "8th": "eighth",
        "9th": "ninth",
        "10th": "tenth",
    }


def toString(word_items):
    return " ".join([i.text for i in word_items])


class PhraseItem:
    """
    英文短语的包装类，包含了文本信息和起止点信息
    """

    def __init__(self, word_items):
        self.word_items = word_items
        self._num_2_word = get_num_2_word()
        self._stop_words = stopwords
        self.simple_items = []
        self.simplify()
        self.locs_set = set([i.start for i in word_items])
        self.start_loc = self.word_items[0].start
        self.end_loc = self.word_items[-1].end
        self.no_flag = False

    def _simplify(self, word_items):
        simple_items = []
        for word_item in word_items:
            if word_item.text in self._num_2_word:
                simple_items.append(
                    WordItem(
                        self._num_2_word[word_item.text],
                        word_item.start,
                        word_item.end,
                    )
                )
            elif word_item.text in self._stop_words or isNum(word_item.text):
                continue
            else:
                simple_items.append(word_item)
        return simple_items

    def simplify(self):
        """
        对phrase_item进行简化，去除常用词以及替换数字
        :return:
        """
        self.simple_items = self._simplify(word_items=self.word_items)

    @staticmethod
    def _toString(items):
        return " ".join(i.text for i in items)

    @staticmethod
    def _toSimpleString(word_items):
        simple_items = []
        for word_item in word_items:
            if word_item.text in get_num_2_word():
                simple_items.append(get_num_2_word()[word_item.text])
            elif word_item.text in stopwords or isNum(word_item.text):
                continue
            else:
                simple_items.append(word_item.text)
        return " ".join(simple_items)

    def toString(self):
        return PhraseItem._toString(self.word_items)

    def toSimpleString(self):
        return PhraseItem._toString(self.simple_items)

    def include(self, phrase_item):
        if self.locs_set.issubset(phrase_item.locs_set) or self.locs_set.issuperset(
            phrase_item.locs_set
        ):
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
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


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


def get_spliters():
    """
    用于分割短句的分割词
    :return:
    """
    spliters = set(
        [
            word
            for word, pos in nltk.pos_tag(stopwords)
            if pos in {"CC", "WP", "TO", "WDT"}
        ]
        + [",", ".", ":", ";", "(", ")", "[", "]", "/"]
    )
    return spliters


import unicodedata


def normalize_with_char_map(text):
    text = strip_accents(text)
    text = re.sub("[\"'\\\\\t‘’]", " ", text)
    text = re.sub("(?<=[\w])[\r\n]", ".", text)

    n = len(text)
    out_chars = []
    index_map = []

    i = 0
    while i < n:
        ch = text[i]

        # Rule 0: drop invisible format/control chars (zero-width space, BOM, soft hyphen, ...)
        if not ch.isspace() and unicodedata.category(ch) in ("Cf", "Cc", "Co", "Cn"):
            i += 1
            continue

        # Rule 1: drop '-' in [A-Z]-[0-9]
        if (
            ch == "-"
            and 0 < i < n - 1
            and text[i - 1].isupper()
            and text[i + 1].isdigit()
        ):
            i += 1
            continue

        # Rule 2: collapse whitespace; skip it entirely at the start of the text
        if ch.isspace():
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if out_chars:
                out_chars.append(" ")
                index_map.append(i)
            i = j
            continue

        out_chars.append(ch)
        index_map.append(i)
        i += 1

    # strip a trailing space (and its map entry)
    if out_chars and out_chars[-1] == " ":
        out_chars.pop()
        index_map.pop()

    index_map.append(n)
    return "".join(out_chars), index_map


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R") or treebank_tag == "IN":
        return wordnet.ADV
    else:
        return wordnet.NOUN
