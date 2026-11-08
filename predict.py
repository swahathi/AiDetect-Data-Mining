"""Inference: image -> verdict, confidence, reasons."""
from __future__ import annotations

import joblib
import pandas as pd

from explain import explain
from features import extract_all


_EMB: dict = {}


def _embedder(name: str):
    if name not in _EMB:
        from embed import Embedder
        _EMB[name] = Embedder(name)
    return _EMB[name]


def load_bundle(path: str = "models/model.joblib") -> dict:
    return joblib.load(path)

#accepts input image, loaded trained model, number of important explanation reasons to return 
def analyze(src, bundle: dict, top_n: int = 5) -> dict:
    out = extract_all(src)
    if out is None:
        raise ValueError("Could not read this file as an image.")
    feats, meta_summary, rgb = out #numerical feature values, readable metadata
    X = pd.DataFrame([feats]).reindex(columns=bundle["feature_names"])
    p_hand = float(bundle["pipeline"].predict_proba(X)[0, 1])
    components = {"Handcrafted + metadata model": p_hand}
    p_ai = p_hand
    if "embed_pipeline" in bundle:
        from embed import stack_features
        e = _embedder(bundle["clip_model"]).embed_one(rgb)
        p_clip = float(bundle["embed_pipeline"].predict_proba(e[None])[0, 1])
        components["Deep visual model (CLIP)"] = p_clip
        p_ai = float(bundle["stacker"].predict_proba(stack_features([p_hand], [p_clip]))[0, 1])

    # Explicit self-declared AI provenance in the file outranks any statistical guess.
    declared = [t for k, t in (("meta_software_ai_keyword", "Metadata names a known AI generation tool"),
                               ("meta_has_ai_text_chunk", "Embedded prompt/generation parameters found in the file"),
                               ("meta_has_c2pa_or_iptc_ai", "Provenance tag (C2PA/IPTC) marks the image as AI-generated"))
                if feats.get(k)]
    #metadata forensics
    if declared:
        is_ai, p_ai = True, max(p_ai, 0.99) #probabiloty raised
    else:
        is_ai = p_ai >= 0.5
    """
    CLIP converts image into a deep visual embedding
    higher level visual information learned by pretrained model
    embedding pipeline predicts
    """
    reasons, contrib = explain(bundle, feats, is_ai, top_n)
    if "Deep visual model (CLIP)" in components:
        reasons.insert(0, {"feature": "clip", "label": "CLIP", "contribution": 0.0,
                           "text": f"Deep visual model (CLIP) rates this image {components['Deep visual model (CLIP)']:.0%} likely AI-generated"})
    if declared:
        reasons = [{"feature": "declared", "label": "Declared", "contribution": 1.0, "text": t} for t in declared] + reasons
    return {
        "label": "AI Generated" if is_ai else "Real (camera photo)",
        "is_ai": is_ai,
        "p_ai": p_ai,
        "components": components,
        "confidence": p_ai if is_ai else 1 - p_ai,
        "reasons": reasons,
        "contributions": contrib,
        "features": feats,
        "metadata": meta_summary,
        "rgb": rgb,
    }
"""
Both predicstioned passed to stacker, stacker combines
and produces final probability p_ai
Stacking ensemble learning
"""

if __name__ == "__main__":
    import sys

    b = load_bundle()
    for f in sys.argv[1:]:
        r = analyze(f, b)
        print(f"\n{f}\nPrediction: {r['label']} - Confidence: {r['confidence']:.0%}")
        for x in r["reasons"]:
            print("  *", x["text"])
