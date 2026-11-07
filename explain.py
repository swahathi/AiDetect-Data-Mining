"""Explainable output: SHAP / linear contributions with a class-profile fallback,
turned into human-readable reasons."""
from __future__ import annotations

import numpy as np
import pandas as pd

TREE_MODELS = {"Random Forest", "Decision Tree", "XGBoost"}

LABELS = {
    "vis_brightness": "Average brightness", "vis_contrast": "Contrast",
    "vis_entropy": "Tonal entropy", "vis_sharpness": "Sharpness (Laplacian variance)",
    "vis_edge_density": "Edge density", "vis_noise_std": "Image noise level",
    "vis_noise_mad": "Robust noise level", "vis_noise_uniformity": "Noise variation across regions",
    "vis_texture_uniformity": "Texture variation across regions",
    "vis_glcm_contrast": "GLCM texture contrast", "vis_glcm_homogeneity": "GLCM homogeneity",
    "vis_glcm_energy": "GLCM energy", "vis_glcm_correlation": "GLCM correlation",
    "vis_lbp_entropy": "Local-binary-pattern entropy", "vis_saturation_mean": "Mean saturation",
    "vis_saturation_std": "Saturation spread", "vis_unique_color_ratio": "Colour diversity",
    "vis_fft_mid_ratio": "Mid-frequency energy", "vis_fft_high_ratio": "High-frequency energy",
    "meta_exif_count": "Number of EXIF fields", "meta_width": "Image width", "meta_height": "Image height",
    "meta_megapixels": "Megapixels", "meta_aspect_ratio": "Aspect ratio",
    "meta_file_kb": "File size (KB)", "meta_bytes_per_pixel": "Bytes per pixel (compression)",
}
for c in "rgb":
    LABELS[f"vis_{c}_mean"] = f"{c.upper()} channel mean"
    LABELS[f"vis_{c}_std"] = f"{c.upper()} channel spread"
    LABELS[f"vis_{c}_hist_entropy"] = f"{c.upper()} histogram entropy"
    for b in range(8):
        LABELS[f"vis_hist_{c}{b}"] = f"{c.upper()} histogram bin {b + 1}/8"

# binary metadata flags: (text when 0, text when 1)
BINARY = {
    "meta_has_exif": ("No EXIF metadata at all", "EXIF metadata present"),
    "meta_has_make": ("Camera make missing", "Camera make recorded"),
    "meta_has_model": ("Camera model missing", "Camera model recorded"),
    "meta_has_lens": ("Lens information missing", "Lens information recorded"),
    "meta_has_datetime": ("Capture date/time missing", "Capture date/time recorded"),
    "meta_has_gps": ("No GPS data", "GPS data present"),
    "meta_has_software": ("No software tag", "Software tag present"),
    "meta_software_ai_keyword": ("No AI-tool keyword in metadata", "Metadata names a known AI generation tool"),
    "meta_has_ai_text_chunk": ("No embedded prompt/parameters", "Embedded prompt/parameters text found"),
    "meta_has_c2pa_or_iptc_ai": ("No provenance (C2PA/IPTC) AI tag", "Provenance tag marks AI-generated media"),
    "meta_has_icc": ("No ICC colour profile", "ICC colour profile embedded"),
    "meta_is_jpeg": ("Not a JPEG file", "JPEG file"),
    "meta_is_png": ("Not a PNG file", "PNG file"),
    "meta_dims_ai_typical": ("Dimensions not typical of AI generators", "Dimensions match typical AI generator sizes"),
}


def label_of(name: str) -> str:
    return LABELS.get(name, name.replace("meta_", "").replace("vis_", "").replace("_", " ").capitalize())


def contributions(bundle: dict, feats: dict) -> pd.Series:
    """Signed contribution per selected feature; positive => pushes towards AI-generated."""
    pipe, names = bundle["pipeline"], bundle["feature_names"]
    x = pd.DataFrame([feats]).reindex(columns=names)
    pre, clf = pipe[:-1], pipe[-1]
    Xt = pre.transform(x)
    sel = list(pre.get_feature_names_out(names))
    model = bundle.get("model_name", "")
    vals = None
    try:
        if model in TREE_MODELS:
            import shap
            sv = shap.TreeExplainer(clf).shap_values(Xt)
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[..., 1]
            vals = sv[0]
        elif model == "Logistic Regression":
            vals = clf.coef_[0] * Xt[0]
    except Exception:
        vals = None
    if vals is None:  # model-agnostic fallback (SVM, or SHAP unavailable)
        st = bundle.get("stats", {})
        vals = []
        for n, xv in zip(sel, Xt[0]):
            raw = feats.get(n, np.nan)
            s = st.get(n)
            if s is None or raw is None or (isinstance(raw, float) and np.isnan(raw)):
                vals.append(0.0)
                continue
            sd = s.get("std", 1.0) or 1.0
            if isinstance(sd, (int, float)) and (np.isnan(sd) or sd == 0):
                sd = 1.0
            real_med = s.get("real_median", 0.0)
            ai_med = s.get("ai_median", 0.0)
            vals.append(abs(raw - real_med) / sd - abs(raw - ai_med) / sd)
        vals = np.array(vals)
    return pd.Series(vals, index=sel).sort_values(key=np.abs, ascending=False)


def reason_text(name: str, value, toward_ai: bool, stats: dict) -> str:
    if name in BINARY:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return BINARY[name][0]
        try:
            idx = int(round(float(value)))
            idx = max(0, min(1, idx))
            return BINARY[name][idx]
        except Exception:
            return BINARY[name][0]
    s = stats.get(name) if stats else None
    side = "AI-generated" if toward_ai else "real"
    if s is None:
        return f"{label_of(name)} pushed the prediction towards {side}."
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"{label_of(name)} is closer to typical {side} images."
    return (f"{label_of(name)} = {value:.3g} is closer to typical {side} images "
            f"(real ~ {s.get('real_median', 0.0):.3g}, AI ~ {s.get('ai_median', 0.0):.3g})")


def explain(bundle: dict, feats: dict, is_ai: bool, top_n: int = 5) -> tuple[list[dict], pd.Series]:
    contrib = contributions(bundle, feats)
    direction = contrib[contrib > 0] if is_ai else contrib[contrib < 0]
    reasons = []
    stats = bundle.get("stats", {})
    for name, c in direction.head(top_n).items():
        reasons.append({
            "feature": name,
            "label": label_of(name),
            "contribution": float(c),
            "text": reason_text(name, feats.get(name, np.nan), c > 0, stats),
        })
    return reasons, contrib
