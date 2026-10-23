from intent_nlp.data import get_splits, subsample


def test_sample_splits_are_disjoint_and_cover_all_labels():
    s = get_splits("sample")
    assert len(s.labels) == 8
    for df in (s.train, s.val, s.test):
        assert set(df["label"]) == set(s.labels)  # stratified: every class everywhere
    train, val, test = set(s.train["text"]), set(s.val["text"]), set(s.test["text"])
    assert not (train & val) and not (train & test) and not (val & test)


def test_splits_are_deterministic():
    a, b = get_splits("sample", seed=7), get_splits("sample", seed=7)
    assert a.train["text"].tolist() == b.train["text"].tolist()


def test_subsample_is_stratified_and_reduces_size():
    s = get_splits("sample")
    half = subsample(s.train, 0.5, seed=1)
    assert len(half) < len(s.train)
    assert set(half["label"]) == set(s.labels)
    assert subsample(s.train, 1.0) is s.train


def test_fraction_only_changes_train_split():
    full, small = get_splits("sample", fraction=1.0), get_splits("sample", fraction=0.5)
    assert len(small.train) < len(full.train)
    assert small.val["text"].tolist() == full.val["text"].tolist()
    assert small.test["text"].tolist() == full.test["text"].tolist()


def test_label_mappings_are_consistent():
    s = get_splits("sample")
    assert all(s.id2label[s.label2id[label]] == label for label in s.labels)
