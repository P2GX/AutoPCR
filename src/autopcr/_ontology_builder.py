from pathlib import Path

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

import json
import copy

from autopcr._working_files import (
    NOABB_LEMMA_FILE,
    LABLE_VOCAB_FILE,
    OBO_JSON_FILE,
    ABBR_JSON_FILE,
    FIRST_WORD_ID_MAP_FILE,
    WORD_ID_MAP_FILE,
    ID_WORD_MAP_FILE,
    ALT_HPO_ID_FILE,
)


class OntologyBuilder:
    def __init__(
        self,
        hpo_file_path: Path,
        root_node: str = "HP:0000118",
        lemmatizer=WordNetLemmatizer(),
    ):
        self._hpo_file_path = hpo_file_path
        self._root_node = root_node
        self._lemmatizer = lemmatizer

    def build(self, working_dir: Path):
        clean_nodes = []

        hpo_obo = self._build_dict(working_dir, clean_nodes)

        self._word_hpo_map(hpo_obo, working_dir, clean_nodes)
        self._hpo_word_map(hpo_obo, working_dir)
        self._alt_hpo(hpo_obo, working_dir)

    def _build_dict(self, out_path: Path, clean_nodes: list) -> dict:
        out_path.mkdir(parents=True, exist_ok=True)

        with open(self._hpo_file_path, "r", encoding="utf-8") as fin:
            all_obo = fin.read().strip().split("\n\n")

        all_nodes = self._get_all_child(all_obo, self._root_node, clean_nodes)

        with open(out_path / LABLE_VOCAB_FILE, "w", encoding="utf-8") as fout_nodes:
            for ele in all_nodes:
                fout_nodes.write(ele + "\n")
            fout_nodes.write("HP:None\n")  # neg label

        hpo_dict = {}
        hpo_obo = {}
        abbr = {}

        for block in all_obo:
            lines = block.split("\n")
            if lines[0] != "[Term]":
                continue

            hpoid = None
            first_name = []
            synonym_list = []
            alt_id_list = []
            def_string = ""
            is_obsolete = ""
            replace_id = ""
            xref_list = []
            isa_list = []

            for line in lines:
                if line.startswith("id: "):
                    hpoid = line[len("id: ") :]
                elif line.startswith("name:"):
                    term = line[len("name: ") :]
                    tokens, lemmas = self._tokenize_and_lemmatize(term)
                    first_name_ori = " ".join(tokens)
                    first_name_lemma = " ".join(lemmas)
                    first_name = [first_name_ori, first_name_lemma]
                    if hpoid in all_nodes:
                        hpo_dict[first_name_ori] = len(tokens)
                        hpo_dict[first_name_lemma] = len(lemmas)
                elif line.startswith("alt_id: "):
                    alt_id = line[len("alt_id: ") :]
                    if alt_id not in alt_id_list:
                        alt_id_list.append(alt_id)
                elif line.startswith('def: "'):
                    def_string = line[len('def: "') : line.find('" ')]
                elif line.startswith("synonym:"):
                    term = line[len('synonym: "') : line.find('" ')]
                    if term.isupper() and len(term) < 10:
                        if hpoid in all_nodes:
                            abbr[term.lower()] = hpoid
                    else:
                        tokens, lemmas = self._tokenize_and_lemmatize(term)
                        synonym_ori = " ".join(tokens)
                        synonym_lemma = " ".join(lemmas)
                        pair = [synonym_ori, synonym_lemma]
                        if pair not in synonym_list:
                            synonym_list.append(pair)
                        if hpoid in all_nodes:
                            hpo_dict[synonym_ori] = len(tokens)
                            hpo_dict[synonym_lemma] = len(lemmas)
                elif line.startswith("xref: "):
                    xref_id = line[len("xref: ") :]
                    if xref_id not in xref_list:
                        xref_list.append(xref_id)
                elif line.startswith("is_a: "):
                    isa_id = line[len("is_a: ") :].split(" ! ")[0]
                    if isa_id not in isa_list:
                        isa_list.append(isa_id)
                elif line.startswith("is_obsolete: "):
                    is_obsolete = line[len("is_obsolete: ") :]
                elif line.startswith("replaced_by: "):
                    replace_id = line[len("replaced_by: ") :]

            if first_name:
                hpo_obo[hpoid] = {
                    "name": first_name,
                    "alt_id": alt_id_list,
                    "def": def_string,
                    "synonym": synonym_list,
                    "xref": xref_list,
                    "is_a": isa_list,
                    "is_obsolete": is_obsolete,
                    "replace_id": replace_id,
                }

        sorted_terms = sorted(hpo_dict.items(), key=lambda kv: (kv[1], kv[0]))
        with open(out_path / NOABB_LEMMA_FILE, "w", encoding="utf-8") as fout_dic:
            for term, _ in sorted_terms:
                fout_dic.write(term + "\n")

        with open(out_path / OBO_JSON_FILE, "w", encoding="utf-8") as fout_obo:
            json.dump(hpo_obo, fout_obo, indent=2)

        with open(out_path / ABBR_JSON_FILE, "w", encoding="utf-8") as fout_abbr:
            json.dump(abbr, fout_abbr)

        return hpo_obo

    def _tokenize_and_lemmatize(self, term: str) -> tuple[list, list]:
        tokens = word_tokenize(
            term.strip().lower().replace("-", " - ").replace("/", " / ")
        )
        lemmas = [
            self._lemmatizer.lemmatize(tok, self._get_wordnet_pos(pos))
            for tok, pos in nltk.pos_tag(tokens)
        ]
        return tokens, lemmas

    @staticmethod
    def _get_wordnet_pos(treebank_tag):
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

    def _get_all_child(self, all_obo, root_node, clean_nodes: list):
        tree_obo = {}
        temp_id = []
        for i in range(0, len(all_obo)):
            lines = all_obo[i].split("\n")
            if lines[0] != "[Term]":
                continue
            is_flag = 0
            for line in lines:
                if line.startswith("id: ") > 0:  # [0:len('id: HP:')]=='id: HP:':
                    hpoid = line[len("id: ") :]
                    temp_id.append(hpoid)
                elif line.startswith("is_a: ") > 0:  # [0:len('is_a: ')]=='is_a: ':
                    is_flag = 1
                    if line.find(" ! ") >= 0:
                        father_hpoid = line[len("is_a: ") : line.find(" ! ")]
                    else:
                        father_hpoid = line[len("is_a: ") :]
            if is_flag == 0:
                # print('is_a none:',lines)
                continue
            if father_hpoid not in tree_obo.keys():
                tree_obo[father_hpoid] = [hpoid]
            else:
                if hpoid not in tree_obo[father_hpoid]:
                    tree_obo[father_hpoid].append(hpoid)

        if root_node == "None":
            clean_nodes = temp_id
        else:
            self._get_child(tree_obo, root_node, clean_nodes)
        return clean_nodes

    def _get_child(self, tree_obo, hpoid, clean_nodes: list):
        father_hpoid = hpoid
        if father_hpoid not in tree_obo.keys():
            return 1
        else:
            for ele in tree_obo[father_hpoid]:
                if ele not in clean_nodes:
                    clean_nodes.append(ele)
                    self._get_child(tree_obo, ele, clean_nodes)

        return 0

    @staticmethod
    def _word_hpo_map(hpo_obo: dict, working_dir: Path, clean_nodes: list):
        for first in [False, True]:
            file_path = working_dir / (
                FIRST_WORD_ID_MAP_FILE if first else WORD_ID_MAP_FILE
            )
            with open(file_path, "w", encoding="utf-8") as fout:
                word_hpoid = {}

                for hpoid in hpo_obo.keys():
                    if first and hpoid not in clean_nodes:
                        continue

                    first_name_ori = hpo_obo[hpoid]["name"][0]
                    first_name = hpo_obo[hpoid]["name"][1]
                    if first_name != first_name_ori:
                        if first_name not in word_hpoid.keys():
                            word_hpoid[first_name] = [hpoid]
                        else:
                            if word_hpoid[first_name][0] != hpoid:
                                word_hpoid[first_name] = [hpoid]

                        if first_name_ori not in word_hpoid.keys():
                            word_hpoid[first_name_ori] = [hpoid]
                        else:
                            if word_hpoid[first_name_ori][0] != hpoid:
                                word_hpoid[first_name_ori] = [hpoid]
                    else:
                        if first_name not in word_hpoid.keys():
                            word_hpoid[first_name] = [hpoid]
                        else:
                            if word_hpoid[first_name][0] != hpoid:
                                word_hpoid[first_name] = [hpoid]
                    if first:
                        continue

                    for synonym in hpo_obo[hpoid]["synonym"]:
                        word = synonym[1]
                        word_ori = synonym[0]
                        if word_ori == word:
                            if word not in word_hpoid.keys():
                                word_hpoid[word] = [hpoid]
                            else:
                                if word_hpoid[word][0] != hpoid:
                                    old_hpoid = word_hpoid[word][0]
                                    new_hpoid = hpoid
                                    if hpo_obo[old_hpoid]["name"][1] != word:
                                        word_hpoid[word].append(new_hpoid)

                        else:
                            if word not in word_hpoid.keys():
                                word_hpoid[word] = [hpoid]
                            else:
                                if word_hpoid[word][0] != hpoid:
                                    old_hpoid = word_hpoid[word][0]
                                    new_hpoid = hpoid
                                    if hpo_obo[old_hpoid]["name"][1] != word:
                                        word_hpoid[word].append(new_hpoid)

                            if word_ori not in word_hpoid.keys():
                                word_hpoid[word_ori] = [hpoid]
                            else:
                                if word_hpoid[word_ori][0] != hpoid:
                                    old_hpoid = word_hpoid[word_ori][0]
                                    new_hpoid = hpoid
                                    if hpo_obo[old_hpoid]["name"][0] != word_ori:
                                        word_hpoid[word_ori].append(new_hpoid)

                json.dump(word_hpoid, fout, indent=2)

    @staticmethod
    def _hpo_word_map(hpo_obo: dict, out_path: Path):
        with open(out_path / ID_WORD_MAP_FILE, "w", encoding="utf-8") as fout:
            hpoid_word = {}

            for hpoid in hpo_obo.keys():
                first_name = hpo_obo[hpoid]["name"][1]
                first_name_ori = hpo_obo[hpoid]["name"][0]
                if first_name_ori == first_name:
                    if hpoid not in hpoid_word.keys():
                        hpoid_word[hpoid] = [first_name]
                    else:
                        if first_name not in hpoid_word[hpoid]:
                            hpoid_word[hpoid].append(first_name)
                else:
                    if hpoid not in hpoid_word.keys():
                        hpoid_word[hpoid] = [first_name, first_name_ori]
                    else:
                        if first_name not in hpoid_word[hpoid]:
                            hpoid_word[hpoid].append(first_name)
                        if first_name_ori not in hpoid_word[hpoid]:
                            hpoid_word[hpoid].append(first_name_ori)

                for synonym in hpo_obo[hpoid]["synonym"]:
                    word = synonym[1]
                    word_ori = synonym[0]
                    if word == word_ori:
                        if word not in hpoid_word[hpoid]:
                            hpoid_word[hpoid].append(word)
                    else:
                        if word not in hpoid_word[hpoid]:
                            hpoid_word[hpoid].append(word)
                        if word_ori not in hpoid_word[hpoid]:
                            hpoid_word[hpoid].append(word_ori)

                for alt_id in hpo_obo[hpoid]["alt_id"]:
                    if alt_id not in hpoid_word.keys():
                        hpoid_word[alt_id] = hpoid_word[hpoid]
                    else:
                        temp = copy.deepcopy(hpoid_word[alt_id])
                        for ele in hpoid_word[hpoid]:
                            if ele not in temp:
                                temp.append(ele)
                        hpoid_word[alt_id] = copy.deepcopy(temp)
                        hpoid_word[hpoid] = copy.deepcopy(temp)

            json.dump(hpoid_word, fout, indent=2)

    @staticmethod
    def _alt_hpo(hpo_obo: dict, out_path: Path):
        with open(out_path / ALT_HPO_ID_FILE, "w", encoding="utf-8") as fout:
            alt_hpoid = {}
            for hpoid in hpo_obo.keys():
                if hpo_obo[hpoid]["is_obsolete"] == "":
                    alt_hpoid[hpoid] = hpoid
                    for ele in hpo_obo[hpoid]["alt_id"]:
                        if ele not in alt_hpoid.keys():
                            alt_hpoid[ele] = hpoid
                        else:
                            pass
                            # print('alt_id:',ele,'old:',alt_hpoid[ele])
            json.dump(alt_hpoid, fout, indent=2)
