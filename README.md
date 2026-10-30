# AutoPCR
[![DOI](https://img.shields.io/badge/DOI-10.1093%2Fbioinformatics%2Fbtag304-blue)](https://doi.org/10.1093/bioinformatics/btag304)
[![arXiv](https://img.shields.io/badge/arXiv-2507.19315-b31b1b.svg)](https://arxiv.org/abs/2507.19315)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Automated Phenotype Concept Recognition by Prompting, packaged as a Python library.**

AutoPCR finds phenotype mentions in free text (clinical notes, case reports, abstracts) and links each one to a concept in the [Human Phenotype Ontology (HPO)](https://hpo.jax.org/). It needs no ontology-specific training: you point it at an `.obo` file, it builds a dense index once, and a large language model makes the final call on ambiguous matches.

This package wraps the method published by Tao *et al.* in *Bioinformatics* ([10.1093/bioinformatics/btag304](https://doi.org/10.1093/bioinformatics/btag304)). The research code and experiment scripts behind that paper are at [yctao7/AutoPCR](https://github.com/yctao7/AutoPCR). This repository turns them into an installable package with a small API you can import.

```python
from pathlib import Path
from autopcr.autopcr import AutoPCR
from autopcr.config import AutoPCRConfig

pcr = AutoPCR(AutoPCRConfig(working_dir=Path("autopcr_wd")))
pcr.initialize(Path("hp.obo"))

pcr.extract("The patient showed poor eye contact and reported ongoing hallucinations.")
# [{'start': 21, 'end': 37, 'text': 'poor eye contact', 'hpo_id': 'HP:0000817', 'score': ...},
#  {'start': 61, 'end': 75, 'text': 'hallucinations',   'hpo_id': 'HP:0000738', 'score': ...}]
```

*(The output above is illustrative. Exact IDs and scores depend on your HPO release and LLM.)*

---

## How it works

AutoPCR runs every input text through three stages:

```
 text ──► 1. Entity extraction ──► 2. Candidate retrieval ──► 3. Entity linking ──► hits
          BioNER + syntax            SapBERT + FAISS             LLM picks among
          (spans that could be       (top-k nearest HPO          the top-k candidates
           phenotypes)                terms per span)             (only when unsure)
```

1. **Entity extraction.** A clinical NER model (Stanza, i2b2/MIMIC) proposes spans. Constituency parsing (spaCy + benepar) adds noun phrases and splits coordinated phrases. For example, *"short and broad fingers"* becomes *"short fingers"* and *"broad fingers"*. Abbreviation resolution is optional.
2. **Candidate concept retrieval.** Each span is embedded with [SapBERT](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) and searched against a FAISS index of all HPO labels and synonyms.
   - Similarity ≥ `accept_threshold` (τ₁, default 0.95): the top hit is accepted directly.
   - Similarity ≥ `candidate_threshold` (τ₂, default 0.85): the top `num_candidates` (k = 5) concepts go to stage 3.
   - Anything lower is dropped.
3. **Entity linking.** An LLM receives the sentence, the span and the k candidates, and chooses one of them or none. Spans accepted in stage 2 never reach the LLM, so only the hard cases cost LLM calls.

In the paper, this setup had the best average mention-level F1 across four benchmarks (BIOC-GS, GSC-2024, ID-68 and NCBI). It also transferred to a new ontology (MEDIC) without retraining. See the paper for full results.

---

## Installation

Requires **Python ≥ 3.11**.

```bash
pip install git+https://github.com/SmartMonkey-git/AutoPCR.git
```

To add optional abbreviation detection through scispaCy:

```bash
pip install "autopcr[abbreviations] @ git+https://github.com/SmartMonkey-git/AutoPCR.git"
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
```

### Models downloaded on first use

The first import or run fetches these, so expect a few GB of downloads and a slow first start:

| Model | Used for | Source |
|---|---|---|
| `en_core_web_trf` | Parsing and tokenisation | spaCy (downloaded automatically) |
| `benepar_en3_large` | Constituency parsing | benepar (downloaded automatically) |
| Stanza `mimic` + `i2b2` NER | Clinical entity extraction | Stanza |
| `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | Span and concept embeddings | Hugging Face |

A CUDA GPU is used automatically if one is available. Otherwise everything runs on CPU.

### Getting the ontology

Download an HPO release in OBO format:

```bash
curl -LO https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp.obo
```


To skip the LLM and use retrieval only, set `linking_model="none"`.

---

## Usage

### 1. Initialise once

```python
from pathlib import Path
from autopcr.autopcr import AutoPCR
from autopcr.config import AutoPCRConfig

config = AutoPCRConfig(working_dir=Path("autopcr_wd"))
pcr = AutoPCR(config)
pcr.initialize(Path("hp.obo"))     # builds the dictionary and SapBERT index
```

`initialize()` parses the ontology below `root_node`, which defaults to `HP:0000118` (*Phenotypic abnormality*). It writes the lookup tables and FAISS indices into `working_dir`. Later calls reuse those files. Pass `overwrite=True` to rebuild them, for example after you update `hp.obo`.

### 2. Extract

```python
hits = pcr.extract(
    "During the interview the patient displayed marked autistic behaviors, "
    "including poor eye contact and rigid, repetitive speech."
)

for h in hits:
    print(f"{h['hpo_id']}\t{h['text']!r}\t[{h['start']}:{h['end']}]\t{h['score']:.2f}")
```

Each result is a `Hit` (a `TypedDict`):

| Field | Type | Meaning |
|---|---|---|
| `start`, `end` | `int` | Character offsets into the input text |
| `text` | `str` | The matched span (`text[start:end]`) |
| `hpo_id` | `str` | The linked concept ID, e.g. `HP:0000817` |
| `score` | `float` | Confidence of the link |

Calling `extract()` before `initialize()` raises `autopcr.autopcr.UninitializedError`.

---

## Configuration

All options live on the frozen dataclass `AutoPCRConfig`:

| Option | Default | Description |
|---|---|---|
| `working_dir` | **required** | Where ontology tables and indices are stored |
| `span_extractor` | `"neural++"` | `"rule"`: PhenoTagger-style dictionary · `"neural"`: BioNER · `"neural+"`: + benepar noun phrases · `"neural++"`: + coordination splitting |
| `detect_abbreviations` | `False` | Resolve abbreviations with scispaCy (needs the `abbreviations` extra) |
| `retrieval_model` | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | Embedding model for candidate retrieval |
| `accept_threshold` | `0.95` | τ₁: link directly at or above this similarity |
| `candidate_threshold` | `0.85` | τ₂: send to the LLM at or above this similarity |
| `num_candidates` | `5` | k: candidates shown to the LLM |
| `linking_model` | `ttempvnn/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-Q4-K-M` | LLM model name on the chosen provider, or `"none"` |
| `provider` | `PROVIDERS["ollama"]` | LLM backend (see [Setting up an LLM](#setting-up-an-llm)) |
| `use_finetuning_prompt` | `False` | Use the short prompt for the fine-tuned linker |
| `seed` | `42` | Sampling seed passed to the LLM (temperature is fixed at 0) |
| `use_cache` | `False` | Cache LLM decisions per span within a session |
| `keep_longest_span_only` | `False` | Drop nested matches and keep only the longest |
| `device` | `None` | Torch device (`"cuda"`, `"cpu"`, `"mps"` …); auto-detects if unset |

`0 ≤ candidate_threshold ≤ accept_threshold ≤ 1` is checked on construction.

---

## macOS notes

PyTorch, FAISS and scikit-learn wheels on macOS each bundle their own OpenMP runtime, which can cause hangs or segfaults. On import, AutoPCR sets `OMP_NUM_THREADS=1` and `TOKENIZERS_PARALLELISM=false` and limits OpenMP to one thread. For this to work, **import `autopcr` before `torch`, `faiss` or `sklearn`**.

---



## Citation

If you use AutoPCR in your work, please cite the original paper:

> Tao Y, Huang Y, Wang Y, Luo X, Liu J. **AutoPCR: Automated Phenotype Concept Recognition by Prompting.** *Bioinformatics*, 2026. doi:[10.1093/bioinformatics/btag304](https://doi.org/10.1093/bioinformatics/btag304)

```bibtex
@article{tao2026autopcr,
  title   = {AutoPCR: Automated Phenotype Concept Recognition by Prompting},
  author  = {Tao, Yicheng and Huang, Yuanhao and Wang, Yiqun and Luo, Xin and Liu, Jie},
  journal = {Bioinformatics},
  year    = {2026},
  doi     = {10.1093/bioinformatics/btag304}
}
```

Preprint: [arXiv:2507.19315](https://arxiv.org/abs/2507.19315)

## Acknowledgements

The algorithm and most of the underlying code are by **Yicheng Tao** and colleagues at the University of Michigan ([yctao7/AutoPCR](https://github.com/yctao7/AutoPCR)). Their code in turn adapts parts of [PhenoTagger](https://github.com/ncbi-nlp/PhenoTagger), [PhenoBERT](https://github.com/EclipseCN/PhenoBERT) and [REAL](https://github.com/dash-ka/REAL-BioCR). Thanks to all of them for open-sourcing their work.

## License

[MIT](LICENSE)