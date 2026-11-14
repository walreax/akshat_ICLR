"""
build_concept_spans.py
======================
Concept units + character spans for the 600-prompt eval set, for the text-grounded
SAE-feature-selection experiment (src/text_grounded_sd3.py).

Same unit definition as extract_concepts.py (spaCy noun-chunk heads, typed object/global via
WordNet, ranked by tf x idf) but (a) idf is computed over the 600 eval-set prompts themselves
(the dev/test corpora used by extract_concepts.py do not cover this set), and (b) every
occurrence of each selected head is recorded as a character span in the prompt, so the GPU
script can map it onto the text encoders' tokens.

Selection (fixed up front, not tuned): the K=8 most salient *object*-typed heads per prompt.

Usage:
    python src/build_concept_spans.py            # -> data/concept_spans_evalset.json
"""
import json
import math
from collections import Counter

import pandas as pd
import spacy
from nltk.corpus import wordnet as wn

K = 8
OBJECT_LEXNAMES = {"noun.animal", "noun.artifact", "noun.body", "noun.food", "noun.object",
                   "noun.person", "noun.plant", "noun.shape", "noun.group"}

nlp = spacy.load("en_core_web_sm", disable=["ner"])
nlp.max_length = 200_000


def concept_type(tok) -> str:
    if tok.pos_ == "PROPN":
        return "object"
    syns = wn.synsets(tok.lemma_, pos=wn.NOUN)
    if not syns:
        return "object" if tok.pos_ == "NOUN" else "global"
    return "object" if syns[0].lexname() in OBJECT_LEXNAMES else "global"


def analyse(text: str):
    doc = nlp(text)
    heads = {}
    for chunk in doc.noun_chunks:
        r = chunk.root
        if r.pos_ not in ("NOUN", "PROPN") or r.is_stop or not r.is_alpha or len(r.lemma_) < 3:
            continue
        heads.setdefault(r.lemma_.lower(), r.pos_)
    spans = {}
    tf = Counter()
    for tok in doc:
        lem = tok.lemma_.lower()
        if lem in heads and tok.pos_ in ("NOUN", "PROPN"):
            tf[lem] += 1
            spans.setdefault(lem, []).append([tok.idx, tok.idx + len(tok.text), tok.text])
    units = []
    for lem in heads:
        tok_for_type = next(t for t in doc if t.lemma_.lower() == lem and t.pos_ in ("NOUN", "PROPN"))
        units.append({"head": lem, "type": concept_type(tok_for_type), "tf": tf[lem], "spans": spans.get(lem, [])})
    return units


def main():
    df = pd.read_csv("data/prompts_600.csv")
    analysed = {}
    df_count = Counter()
    for _, row in df.iterrows():
        units = analyse(str(row["content"]))
        analysed[row["id"]] = units
        df_count.update({u["head"] for u in units})
    n_docs = len(df)
    idf = {h: math.log((1 + n_docs) / (1 + c)) + 1 for h, c in df_count.items()}

    out = {}
    n_obj = 0
    for _, row in df.iterrows():
        units = [u for u in analysed[row["id"]] if u["type"] == "object" and u["spans"]]
        for u in units:
            u["salience"] = u["tf"] * idf.get(u["head"], 1.0)
        units.sort(key=lambda u: -u["salience"])
        sel = units[:K]
        n_obj += len(sel)
        out[str(row["name_of_work"])] = {"id": row["id"], "units": sel}
    with open("data/concept_spans_evalset.json", "w") as f:
        json.dump(out, f, indent=1)
    none = sum(1 for v in out.values() if not v["units"])
    print(f"{len(out)} prompts, {n_obj / len(out):.1f} selected object units/prompt, "
          f"{none} prompts with no object unit -> data/concept_spans_evalset.json")


if __name__ == "__main__":
    main()
