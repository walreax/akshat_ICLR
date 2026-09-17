"""
extract_concepts.py
====================
Step 01 of the Concept Fidelity Score (writeups/metric_spec.html): turn each
prompt into typed concept units, with no model and no human data involved.

For every prompt: spaCy noun chunks -> (head lemma, attribute lemmas), typed
`object` (a thing with a spatial footprint: person/animal/artifact/plant/
food/body/shape/group, or any proper noun -- a character) or `global`
(feeling, weather, time, state, place ... things that are *supposed* to be
diffuse). Units are ranked by tf x idf salience over the whole prompt
corpus (train 4200 + test 1800) and the top K=8 are marked `selected`; the
rest are kept in the file but unselected, so K can be audited on the dev
split without re-running this.

Usage:
    python extract_concepts.py
Writes data/concepts_test.json and data/concepts_calib.json, keyed by the
same concept_id the extractor writes to concept_metrics.csv (name_of_work).
"""

import json
import math
import re
from collections import Counter, defaultdict

import pandas as pd
import spacy
from nltk.corpus import wordnet as wn

K = 8
OBJECT_LEXNAMES = {"noun.animal", "noun.artifact", "noun.body", "noun.food", "noun.object",
                   "noun.person", "noun.plant", "noun.shape", "noun.group"}
CORPUS_FILES = ["train_set_4200.csv", r"C:\Users\Akshat Jha\Downloads\test_set_1800.csv"]
TARGETS = {"test": "data/test_set_600.csv", "calib": "data/calib_set_600.csv"}

nlp = spacy.load("en_core_web_sm", disable=["ner"])
nlp.max_length = 200_000


def concept_type(head_token) -> str:
    if head_token.pos_ == "PROPN":
        return "object"
    syns = wn.synsets(head_token.lemma_, pos=wn.NOUN)
    if not syns:
        return "object" if head_token.pos_ == "NOUN" else "global"
    return "object" if syns[0].lexname() in OBJECT_LEXNAMES else "global"


def units_for(text: str):
    doc = nlp(text)
    seen = {}
    for chunk in doc.noun_chunks:
        head = chunk.root
        if head.pos_ not in ("NOUN", "PROPN") or head.is_stop or not head.is_alpha or len(head.lemma_) < 3:
            continue
        lemma = head.lemma_.lower()
        attrs = [t.lemma_.lower() for t in head.children if t.dep_ == "amod" and t.is_alpha and not t.is_stop]
        u = seen.setdefault(lemma, {"head": lemma, "type": concept_type(head), "attrs": [], "tf": 0, "surface": chunk.text})
        u["tf"] += 1
        for a in attrs:
            if a not in u["attrs"]:
                u["attrs"].append(a)
    return list(seen.values())


def main():
    print("building idf over the full prompt corpus ...")
    df_count = Counter()
    n_docs = 0
    for f in CORPUS_FILES:
        for text in pd.read_csv(f)["content"].astype(str):
            n_docs += 1
            heads = {u["head"] for u in units_for(text)}
            df_count.update(heads)
    idf = {h: math.log((1 + n_docs) / (1 + c)) + 1 for h, c in df_count.items()}
    print(f"  {n_docs} prompts, {len(idf)} distinct heads")

    for name, path in TARGETS.items():
        df = pd.read_csv(path)
        out = {}
        n_units = n_sel = n_obj = 0
        for _, row in df.iterrows():
            units = units_for(str(row["content"]))
            for u in units:
                u["salience"] = u["tf"] * idf.get(u["head"], 1.0)
            units.sort(key=lambda u: -u["salience"])
            for i, u in enumerate(units):
                u["selected"] = i < K
            out[str(row["name_of_work"])] = {"id": row["id"], "source": row["source"],
                                             "split": row.get("split", ""), "units": units}
            n_units += len(units); n_sel += min(K, len(units)); n_obj += sum(u["type"] == "object" for u in units[:K])
        with open(f"data/concepts_{name}.json", "w") as f:
            json.dump(out, f, indent=1)
        print(f"{name}: {len(out)} prompts, {n_units/len(out):.1f} units/prompt, "
              f"{n_sel/len(out):.1f} selected, {n_obj/max(n_sel,1):.0%} of selected are object-typed "
              f"-> data/concepts_{name}.json")

    # concept recurrence in the calibration set -- what the contrastive mapping can actually use
    cal = json.load(open("data/concepts_calib.json"))
    heads = Counter(u["head"] for p in cal.values() for u in p["units"] if u["selected"])
    print(f"\ncalib: {len(heads)} distinct selected heads; with >=5 prompts: {sum(c >= 5 for c in heads.values())}, "
          f">=10: {sum(c >= 10 for c in heads.values())}")
    print("most common:", heads.most_common(15))


if __name__ == "__main__":
    main()
