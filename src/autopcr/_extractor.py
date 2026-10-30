from pathlib import Path

from autopcr.config import AutoPCRConfig
from autopcr.hit import Hit
from autopcr._dic_ner import dic_ont
from autopcr._tagging_text import bioTag
from autopcr._ee import (
    process_text2phrases_pbert,
    process_text2phrases_benepar,
    process_text2phrases_abbr,
)
from autopcr._working_files import (
    NOABB_LEMMA_FILE,
    WORD_ID_MAP_FILE,
    ID_WORD_MAP_FILE,
    MAIN_LABEL_INDEX_FILE,
    INDEX_TO_ID_LABEL_FILE,
    INDEX_TO_TERM_LABEL_FILE,
    ID_TO_CONCEPT_FILE,
    FIRST_WORD_ID_MAP_FILE,
    ABBR_JSON_FILE,
)

"""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
"""


class Extractor:
    def __init__(
        self,
        extractor_config: AutoPCRConfig,
    ):
        self.working_dir = extractor_config.working_dir
        self.ee = extractor_config.span_extractor
        self.abbr_recog = extractor_config.detect_abbreviations
        self.ccr = extractor_config.retrieval_model
        self.tau_1 = extractor_config.accept_threshold
        self.only_longest = extractor_config.keep_longest_span_only

        ontfiles = {
            "dic_file": self.working_dir / NOABB_LEMMA_FILE,
            "word_hpo_file": self.working_dir / WORD_ID_MAP_FILE,
            "hpo_word_file": self.working_dir / ID_WORD_MAP_FILE,
        }
        self.biotag_dic = dic_ont(ontfiles)

        vocabfiles = {
            "index": self.working_dir / MAIN_LABEL_INDEX_FILE,
            "index_to_id": self.working_dir / INDEX_TO_ID_LABEL_FILE,
            "index_to_term": self.working_dir / INDEX_TO_TERM_LABEL_FILE,
            "id_to_concept": self.working_dir / ID_TO_CONCEPT_FILE,
            "firstword_to_id": self.working_dir / FIRST_WORD_ID_MAP_FILE,
            "abbr": self.working_dir / ABBR_JSON_FILE,
        }

        if extractor_config.use_finetuning_prompt:
            from autopcr._nn_model3 import bioTag_SapBERT
        else:
            from autopcr._nn_model2 import bioTag_SapBERT

        self.nn_model = bioTag_SapBERT(
            vocabfiles,
            ccr=extractor_config.retrieval_model,
            tau_1=extractor_config.accept_threshold,
            tau_2=extractor_config.candidate_threshold,
            k=extractor_config.num_candidates,
            el=extractor_config.linking_model,
            api_key=extractor_config.api_key,
            seed=extractor_config.seed,
            use_cache=extractor_config.use_cache,
            device=extractor_config.device,
            provider=extractor_config.provider,
        )

        self.files = None

    def extract(self, text) -> list[Hit]:
        """Returns a list of dicts: start, end, text, concept_id, score."""
        if self.ee == "rule":
            anns = []
            for ele in bioTag(
                text,
                self.biotag_dic,
                self.nn_model,
                onlyLongest=False,
                abbrRecog=self.abbr_recog,
                Threshold=self.tau_1,
            ):
                anns.append(
                    (ele[0], ele[1], text[int(ele[0]) : int(ele[1])], ele[2], ele[3])
                )
            if self.only_longest:
                anns = self._keep_longest_rule(anns)
        else:
            phrases = self._extract_phrases(text)
            _, anns = self._link(text, phrases)

        return [
            Hit(
                start=int(a[0]),
                end=int(a[1]),
                text=a[2],
                hpo_id=a[3],
                score=float(a[4]),
            )
            for a in anns
        ]

    def _extract_phrases(self, text):
        """Same phrase sources as the corpus run, for one text and without the json cache."""
        sources = []
        if self.abbr_recog:
            text_new, phrases_list, index_map = process_text2phrases_abbr(text)
            sources.append(
                [
                    {
                        "start": index_map[p[0]],
                        "end": index_map[p[1]],
                        "phrase": p[2],
                        "phrase_exp": p[3],
                        "phrase_nostopword": p[4],
                    }
                    for p in phrases_list
                ]
            )

        text_new, phrases_list, index_map = process_text2phrases_pbert(text)
        sources.append(
            [
                {
                    "start": index_map[int(p.start_loc)],
                    "end": index_map[int(p.end_loc)],
                    "phrase": text_new[int(p.start_loc) : int(p.end_loc)],
                    "phrase_nostopword": p.toSimpleString(),
                }
                for p in phrases_list
            ]
        )

        if self.ee == "neural+" or self.ee == "neural++":
            text_new, phrases_list, index_map = process_text2phrases_benepar(text)
            sources.append(
                [
                    {
                        "start": index_map[p[0]],
                        "end": index_map[p[1]],
                        "phrase": p[2],
                        "phrase_nostopword": p[3],
                    }
                    for p in phrases_list[0]
                ]
            )
            if self.ee == "neural++":
                sources.append(
                    [
                        {
                            "start": index_map[p[0]],
                            "end": index_map[p[1]],
                            "phrase": p[2],
                            "phrase_nostopword": p[3],
                        }
                        for p in phrases_list[1]
                    ]
                )

        # merge, dropping duplicates (same as remove_dup + add_phrases)
        phrases, seen = [], set()
        for source in sources:
            for p in source:
                key = (p["start"], p["end"], p["phrase"])
                if key not in seen:
                    seen.add(key)
                    phrases.append(p)
        return phrases

    def _link(self, text, phrases):
        """Links phrases to concepts and filters them. Returns (raw anns, final anns)."""
        ts, disjoint = self.tau_1, self.only_longest

        annotations = self.nn_model.predict_llm(
            [phrase.get("phrase_exp", phrase["phrase"]) for phrase in phrases],
            text * len(phrases),
        )

        anns = []
        for phrase, ele in zip(phrases, annotations):  # annotations.split('\n')
            start, end = phrase["start"], phrase["end"]
            uri, score = ele[0]  # ele.split('\t')[0].split('|')
            score = min(float(score), 1.0)
            if abs(score) >= ts:
                anns.append((str(start), str(end), phrase["phrase"], uri, str(score)))
        anns = list(
            {(ann[0], ann[1], ann[3]): ann for ann in anns}.values()
        )  # remove duplicates in phrases
        anns_raw = anns

        # intersect condition
        anns_new = []
        for i in range(len(anns)):
            overlap = False
            for j in range(len(anns)):
                if i != j:
                    len_i, len_j = (
                        int(anns[i][1]) - int(anns[i][0]),
                        int(anns[j][1]) - int(anns[j][0]),
                    )
                    cond = (
                        max(int(anns[i][0]), int(anns[j][0]))
                        < min(int(anns[i][1]), int(anns[j][1]))
                        and anns[i][3] == anns[j][3]
                        and (
                            float(anns[i][4]) < float(anns[j][4])
                            or (
                                float(anns[i][4]) == float(anns[j][4])
                                and (
                                    len_i > len_j
                                    or (
                                        len_i == len_j
                                        and int(anns[i][0]) > int(anns[j][0])
                                    )
                                )
                            )
                        )
                    )
                    if cond:
                        overlap = True
                        break
            if not overlap:
                anns_new.append(anns[i])

        # only keep longest nested entities
        if disjoint:

            def is_word_subsequence(short, long):
                it = iter(long.lower().split())
                return all(any(w == x for x in it) for w in short.lower().split())

            anns = anns_new
            anns_new = []
            for i in range(len(anns)):
                overlap = False
                for j in range(len(anns)):
                    if (
                        i != j
                        and int(anns[j][0]) <= int(anns[i][0])
                        and int(anns[i][1]) <= int(anns[j][1])
                        and int(anns[i][1]) - int(anns[i][0])
                        < int(anns[j][1]) - int(anns[j][0])
                        and anns[i][2].lower() in anns[j][2].lower()
                    ):
                        overlap = True
                        break
                    if (
                        i != j
                        and int(anns[j][0]) == int(anns[i][0])
                        and int(anns[i][1]) == int(anns[j][1])
                        and is_word_subsequence(anns[i][2], anns[j][2])
                    ):
                        overlap = True
                        break
                if not overlap:
                    anns_new.append(anns[i])

        return anns_raw, anns_new

    @staticmethod
    def _keep_longest_rule(anns):
        anns_new = []
        for i in range(len(anns)):
            overlap = False
            for j in range(len(anns)):
                if (
                    i != j
                    and int(anns[j][0]) <= int(anns[i][0])
                    and int(anns[i][1]) <= int(anns[j][1])
                    and int(anns[i][1]) - int(anns[i][0])
                    < int(anns[j][1]) - int(anns[j][0])
                ):
                    overlap = True
                    break
            if not overlap:
                anns_new.append(anns[i])
        return anns_new
