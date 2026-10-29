import os
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
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        fin = open(self._hpo_file_path, "r", encoding="utf-8")
        fout_dic = open(out_path / NOABB_LEMMA_FILE, "w", encoding="utf-8")
        fout_nodes = open(out_path / LABLE_VOCAB_FILE, "w", encoding="utf-8")

        hpo_dict = {}
        all_obo = fin.read().strip().split("\n\n")
        fin.close()

        all_nodes = self._get_all_child(
            all_obo, self._root_node, clean_nodes
        )  # get all nodes
        # print(len(all_nodes))
        for ele in all_nodes:
            fout_nodes.write(ele + "\n")
        fout_nodes.write("HP:None" + "\n")  # neg label
        fout_nodes.close()

        fout_obo = open(out_path / OBO_JSON_FILE, "w", encoding="utf-8")

        hpo_obo = {}
        abbr = {}
        for i in range(0, len(all_obo)):
            print(
                "Obo File Processing:{0}%".format(round(i * 100 / len(all_obo))),
                end="\r",
            )
            lines = all_obo[i].split("\n")
            if lines[0] != "[Term]":
                continue
            first_name = []
            synonym_list = []
            alt_id_list = []
            def_string = ""
            is_obsolete = ""
            replace_id = ""
            xref_list = []
            isa_list = []
            for line in lines:
                if line.startswith("id: ") > 0:  # find('id: HP:')==0:
                    hpoid = line[len("id: ") :]
                elif line.find("name:") == 0:
                    term = line[len("name: ") :]
                    tokens = word_tokenize(
                        term.strip().lower().replace("-", " - ").replace("/", " / ")
                    )
                    token_pos = nltk.pos_tag(tokens)
                    lemmas = [
                        self._lemmatizer.lemmatize(
                            token[0], self._get_wordnet_pos(token[1])
                        )
                        for token in token_pos
                    ]
                    first_name_ori = " ".join(tokens)
                    first_name_lemma = " ".join(lemmas)
                    first_name = [first_name_ori, first_name_lemma]

                    if hpoid in all_nodes:  # input dict
                        hpo_dict[first_name_ori] = len(tokens)
                        hpo_dict[first_name_lemma] = len(lemmas)

                elif line.find("alt_id: ") == 0:
                    alt_id = line[len("alt_id: ") :]
                    if alt_id not in alt_id_list:
                        alt_id_list.append(alt_id)
                elif line.find('def: "') == 0:
                    eid = line.find('" ')
                    def_string = line[len('def: "') : eid]
                elif line.find("synonym:") == 0:
                    eid = line.find('" ')
                    term = line[len('synonym: "') : eid]
                    if term.isupper() and len(term) < 10:
                        # pass
                        # print('synonym name:',term)
                        if hpoid in all_nodes:
                            abbr[term.lower()] = hpoid
                    else:
                        tokens = word_tokenize(
                            term.strip().lower().replace("-", " - ").replace("/", " / ")
                        )
                        token_pos = nltk.pos_tag(tokens)
                        lemmas = [
                            self._lemmatizer.lemmatize(
                                token[0], self._get_wordnet_pos(token[1])
                            )
                            for token in token_pos
                        ]
                        synonym_ori = " ".join(tokens)
                        synonym_lemma = " ".join(lemmas)
                        temp_list = [synonym_ori, synonym_lemma]
                        if temp_list not in synonym_list:
                            synonym_list.append(temp_list)

                        if hpoid in all_nodes:  # input dict
                            hpo_dict[synonym_ori] = len(tokens)
                            hpo_dict[synonym_lemma] = len(lemmas)

                elif line.find("xref: ") == 0:
                    xref_id = line[len("xref: ") :]
                    if xref_id not in xref_list:
                        xref_list.append(xref_id)
                elif line.find("is_a: ") == 0:
                    if line.find(" ! ") >= 0:
                        isa_id = line[len("is_a: ") : line.find(" ! ")]
                    else:
                        isa_id = line[len("is_a: ") :]
                    if isa_id not in isa_list:
                        isa_list.append(isa_id)
                elif line.find("is_obsolete: ") == 0:
                    is_obsolete = line[len("is_obsolete: ") :]
                elif line.find("replaced_by: ") == 0:
                    replace_id = line[len("replaced_by: ") :]
            if first_name != []:
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

        sort_hpo_dict = sorted(
            hpo_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=False
        )
        for ele in sort_hpo_dict:
            fout_dic.write(ele[0] + "\n")
        fout_dic.close()

        json.dump(hpo_obo, fout_obo, indent=2)
        fout_obo.close()
        json.dump(abbr, open(out_path / ABBR_JSON_FILE, "w", encoding="utf-8"))
        return hpo_obo

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
            fout = open(
                file_path,
                "w",
                encoding="utf-8",
            )

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
            fout.close()

    @staticmethod
    def _hpo_word_map(hpo_obo: dict, out_path: Path):
        fout = open(out_path / ID_WORD_MAP_FILE, "w", encoding="utf-8")
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
        fout.close()

    @staticmethod
    def _alt_hpo(hpo_obo: dict, out_path: Path):
        fout = open(out_path / ALT_HPO_ID_FILE, "w", encoding="utf-8")
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
