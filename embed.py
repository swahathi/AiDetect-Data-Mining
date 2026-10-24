"""Pretrained CLIP image embeddings (deep visual features) for generalising to unseen AI generators.

Each image yields two views, both taken after the same scale normalisation as the handcrafted features:
  global : whole square image downsized to 224  -> overall "look" (smoothness, composition, aesthetic)
  crop   : native-detail 224 crop from the 1024 version -> local texture / generation traces
Embeddings are L2-normalised and concatenated (2 x dim).
"""
from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image

from features import NORM, normalize_scale

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
DEFAULT_MODEL = "ViT-B-16"


def _views(rgb: np.ndarray) -> np.ndarray:
    sq = normalize_scale(rgb)  # NORM x NORM
    g = cv2.resize(sq, (224, 224), interpolation=cv2.INTER_AREA)
    o = (NORM - 224) // 2
    c = sq[o:o + 224, o:o + 224]
    v = np.stack([g, c]).astype(np.float32) / 255.0
    v = (v - MEAN) / STD
    return v.transpose(0, 3, 1, 2)  # 2 x 3 x 224 x 224


def _load_views(src) -> np.ndarray | None:
    try:
        if isinstance(src, np.ndarray):
            rgb = src
        else:
            data = src if isinstance(src, (bytes, bytearray)) else open(src, "rb").read()
            with Image.open(io.BytesIO(data)) as im:
                rgb = np.asarray(im.convert("RGB"))
        return _views(rgb)
    except Exception:
        return None


def stack_features(p_a, p_b) -> np.ndarray:
    """Logit-transformed probabilities of the two base models -> input for the stacking meta-classifier."""
    def lg(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))
    return np.column_stack([lg(p_a), lg(p_b)])


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, pretrained: str | None = "openai", device: str | None = None):
        import torch
        import open_clip

        self.torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if os.environ.get("AIDETECT_CLIP_RANDOM") == "1":  # offline smoke tests only
            pretrained = None
        self.model = open_clip.create_model(model_name, pretrained=pretrained).eval().to(self.device)
        with torch.no_grad():
            self.dim = int(self.model.encode_image(torch.zeros(1, 3, 224, 224, device=self.device)).shape[-1])
        self.out_dim = 2 * self.dim

    def _encode(self, views: np.ndarray) -> np.ndarray:
        """views: N x 2 x 3 x 224 x 224 -> N x (2*dim)"""
        t = self.torch
        n = views.shape[0]
        x = t.from_numpy(views.reshape(n * 2, 3, 224, 224)).to(self.device)
        with t.no_grad():
            f = self.model.encode_image(x).float()
            f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return f.reshape(n, 2 * self.dim).cpu().numpy()

    def embed_one(self, src) -> np.ndarray:
        v = _load_views(src)
        if v is None:
            raise ValueError("Could not read image for embedding.")
        return self._encode(v[None])[0]

    def embed_many(self, paths: list[str], batch: int = 32, workers: int = 4) -> np.ndarray:
        from tqdm import tqdm

        out = np.zeros((len(paths), self.out_dim), dtype=np.float32)
        bad = 0
        with ThreadPoolExecutor(workers) as ex:
            for s in tqdm(range(0, len(paths), batch), desc=f"CLIP {self.model_name}", unit="batch"):
                chunk = paths[s:s + batch]
                views = list(ex.map(_load_views, chunk))
                ok = [i for i, v in enumerate(views) if v is not None]
                bad += len(chunk) - len(ok)
                if ok:
                    out[[s + i for i in ok]] = self._encode(np.stack([views[i] for i in ok]))
        if bad:
            print(f"[embed] {bad} unreadable images left as zero vectors")
        return out


def embeddings_for(paths: list[str], cache: str, model_name: str = DEFAULT_MODEL, **kw) -> np.ndarray:
    """Embeddings aligned to `paths`, reusing/extending an on-disk cache keyed by path."""
    store: dict[str, np.ndarray] = {}
    if os.path.exists(cache):
        try:
            z = np.load(cache, allow_pickle=False)
            store = dict(zip(z["paths"].tolist(), z["emb"]))
        except Exception:
            store = {}
    missing = [p for p in paths if p not in store]
    if missing:
        print(f"[embed] computing CLIP embeddings for {len(missing)} images "
              f"({len(paths) - len(missing)} cached) - this is the slow step")
        emb = Embedder(model_name).embed_many(missing, **kw)
        store.update(zip(missing, emb))
        np.savez(cache, paths=np.array(list(store.keys())), emb=np.stack(list(store.values())))
    return np.stack([store[p] for p in paths])
