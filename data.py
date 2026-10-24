"""Dataset discovery + parallel feature extraction (with on-disk cache)."""
from __future__ import annotations

import os
import random
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from features import feature_row

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
AI_WORDS = ("ai", "fake", "synthetic", "generated", "diffusion", "gan", "midjourney", "dalle",
            "dall-e", "sdxl", "stable", "sd_", "gen_", "artificial", "deepfake")
REAL_WORDS = ("real", "human", "authentic", "camera", "natural", "original", "genuine", "nature")


def _label_from_parts(parts: list[str]) -> int | None:
    low = [p.lower() for p in parts]
    for p in reversed(low):  # nearest folder wins; AI keywords checked first
        tokens = p.replace("-", "_").replace(" ", "_").split("_")
        if p in AI_WORDS or any(t in AI_WORDS for t in tokens) or p.startswith(AI_WORDS):
            return 1
        if p in REAL_WORDS or any(t in REAL_WORDS for t in tokens):
            return 0
    return None


def discover(root: str | None = None, real_dir: str | None = None, ai_dir: str | None = None) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    if real_dir or ai_dir:
        for d, lab in ((real_dir, 0), (ai_dir, 1)):
            if d:
                items += [(str(p), lab) for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXT]
        return items
    root_p = Path(root)
    for p in root_p.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            lab = _label_from_parts(list(p.relative_to(root_p).parts[:-1]))
            if lab is not None:
                items.append((str(p), lab))
    return items


def from_csv(csv_path: str, image_root: str, path_col: str, label_col: str) -> list[tuple[str, int]]:
    df = pd.read_csv(csv_path)
    items = []
    for _, r in df.iterrows():
        lab = r[label_col]
        if isinstance(lab, str):
            lab = 1 if lab.strip().lower() in ("1", "ai", "fake", "ai_generated", "generated", "synthetic") else 0
        p = os.path.join(image_root, str(r[path_col]))
        if os.path.exists(p):
            items.append((p, int(lab)))
    return items


def balance_sample(items, max_per_class: int | None, seed: int = 42):
    rng = random.Random(seed)
    by = {0: [], 1: []}
    for it in items:
        by[it[1]].append(it)
    n = min(len(by[0]), len(by[1]))
    if max_per_class:
        n = min(n, max_per_class)
    out = []
    for lab in (0, 1):
        rng.shuffle(by[lab])
        out += by[lab][:n]
    return out


def build_table(items, cache: str | None = None, n_jobs: int = -1) -> pd.DataFrame:
    if cache and os.path.exists(cache):
        cached = pd.read_csv(cache)
        if set(cached["path"]) == {p for p, _ in items}:
            print(f"[data] using cached features: {cache}")
            return cached
    print(f"[data] extracting features from {len(items)} images ...")
    rows = Parallel(n_jobs=n_jobs, batch_size=8)(delayed(feature_row)(p) for p, _ in items)
    recs = []
    for (p, lab), r in zip(items, rows):
        if r is not None:
            recs.append({"path": p, "label": lab, **r})
    df = pd.DataFrame(recs)
    print(f"[data] kept {len(df)}/{len(items)} readable images")
    if cache:
        df.to_csv(cache, index=False)
    return df
