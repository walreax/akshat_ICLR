from sd_activation_extractor import load_concept_dataset

for f in ["data/test_set_600.csv", "data/calib_set_600.csv"]:
    d = load_concept_dataset(f)
    ids = [x["concept_id"] for x in d]
    print(f, "->", len(d), "entries,", len(set(ids)), "unique ids; first:", repr(ids[0]))
