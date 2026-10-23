"""Data loading and splitting for Banking77 intent classification.

Banking77 (Casanueva et al., 2020) has 13,083 customer-service queries labelled with
77 fine-grained banking intents. The official train/test CSVs are downloaded from the
PolyAI GitHub repository on first use and cached under ``data/raw``.

Evaluation protocol used throughout this project
------------------------------------------------
* ``train`` (official, 10,003 rows) is split into train/validation (stratified, 10% val).
  Validation is used for model selection only (C for the baseline, best epoch for the
  transformer).
* ``test`` (official, 3,080 rows) is touched exactly once per run, for the final numbers.
* ``fraction < 1`` subsamples the *training* portion only (stratified), so validation and
  test stay identical across data-efficiency experiments.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

_BASE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
)
BANKING77_URLS = {"train": f"{_BASE_URL}/train.csv", "test": f"{_BASE_URL}/test.csv"}
DEFAULT_CACHE_DIR = Path("data/raw")
# Tiny hand-written fixture (8 intents) used by the tests and for quick smoke runs.
SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "sample" / "smoke_test.csv"


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    labels: list  # sorted list of intent names; index in this list == class id

    @property
    def label2id(self) -> dict:
        return {label: i for i, label in enumerate(self.labels)}

    @property
    def id2label(self) -> dict:
        return dict(enumerate(self.labels))


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as fh:
        fh.write(response.read())
    tmp.replace(dest)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"category": "label"})[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    return df[df["text"] != ""].reset_index(drop=True)


def load_banking77(cache_dir: Path = DEFAULT_CACHE_DIR):
    """Return the official (train_df, test_df); downloads the CSVs on first call."""
    cache_dir = Path(cache_dir)
    frames = {}
    for split, url in BANKING77_URLS.items():
        path = cache_dir / f"banking77_{split}.csv"
        if not path.exists():
            print(f"Downloading Banking77 {split} split ...")
            _download(url, path)
        frames[split] = _clean(pd.read_csv(path))
    return frames["train"], frames["test"]


def load_sample() -> pd.DataFrame:
    return _clean(pd.read_csv(SAMPLE_PATH))


def _stratified_holdout(df: pd.DataFrame, fraction: float, seed: int):
    """Split off a stratified hold-out of ~``fraction`` rows (at least one per class)."""
    n_hold = max(int(round(len(df) * fraction)), df["label"].nunique())
    rest, hold = train_test_split(
        df, test_size=n_hold, stratify=df["label"], random_state=seed
    )
    return rest.reset_index(drop=True), hold.reset_index(drop=True)


def subsample(df: pd.DataFrame, fraction: float, seed: int = 42) -> pd.DataFrame:
    """Stratified subsample of ``df`` keeping ~``fraction`` of the rows."""
    if fraction >= 1.0:
        return df
    n_keep = max(int(round(len(df) * fraction)), df["label"].nunique())
    kept, _ = train_test_split(
        df, train_size=n_keep, stratify=df["label"], random_state=seed
    )
    return kept.reset_index(drop=True)


def get_splits(
    dataset: str = "banking77",
    fraction: float = 1.0,
    val_size: float = 0.1,
    seed: int = 42,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Splits:
    """Build train/val/test splits. ``dataset`` is ``banking77`` or ``sample``."""
    if dataset == "banking77":
        train_full, test = load_banking77(cache_dir)
    elif dataset == "sample":
        train_full, test = _stratified_holdout(load_sample(), 0.3, seed)
    else:
        raise ValueError(f"Unknown dataset {dataset!r} (use 'banking77' or 'sample')")

    labels = sorted(set(train_full["label"]) | set(test["label"]))
    train, val = _stratified_holdout(train_full, val_size, seed)
    train = subsample(train, fraction, seed)
    return Splits(train=train, val=val, test=test.reset_index(drop=True), labels=labels)
