"""Train + compare classifiers on mined features.

Examples
  python train.py --data /path/to/kaggle_dataset
  python train.py --real data/real --ai data/ai --max-per-class 3000
  python train.py --data ds --labels-csv ds/train.csv --path-col file_name --label-col label
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from functools import partial

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import data as D
from embed import stack_features

SEED = 42
# File-format / size / dimension features describe how a dataset was saved, not what the image shows.
FORMAT_FEATURES = ["meta_is_jpeg", "meta_is_png", "meta_width", "meta_height", "meta_megapixels",
                   "meta_aspect_ratio", "meta_dims_ai_typical", "meta_file_kb", "meta_bytes_per_pixel"]


def make_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
        "SVM": SVC(kernel="rbf", probability=True, random_state=SEED),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                          subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                                          random_state=SEED, n_jobs=-1)
    except Exception:
        print("[warn] xgboost not installed - skipping")
    return models


def make_pipe(clf, k: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(partial(mutual_info_classif, random_state=SEED), k=k)),
        ("clf", clf),
    ])


def make_emb_models() -> dict:
    m = {
        "Logistic Regression": LogisticRegression(C=0.1, max_iter=3000),
        "SVM": SVC(kernel="rbf", C=3, probability=True, random_state=SEED),
    }
    try:
        from xgboost import XGBClassifier
        m["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
                                     colsample_bytree=0.4, eval_metric="logloss", random_state=SEED, n_jobs=-1)
    except Exception:
        pass
    return m


def make_emb_pipe(clf) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def save_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Real", "AI"]).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout(), fig.savefig(path, dpi=140), plt.close(fig)


def save_roc(rocs, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for n, (fpr, tpr, _) in rocs.items():
        ax.plot(fpr, tpr, label=n, lw=2.2 if n == "Ensemble" else 1.2)
    ax.plot([0, 1], [0, 1], "k--", lw=.7)
    ax.set_xlabel("False positive rate"), ax.set_ylabel("True positive rate"), ax.legend(fontsize=7)
    ax.set_title("ROC - hold-out set")
    fig.tight_layout(), fig.savefig(path, dpi=140), plt.close(fig)


def md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="dataset root (labels inferred from folder names)")
    ap.add_argument("--real"), ap.add_argument("--ai")
    ap.add_argument("--labels-csv"), ap.add_argument("--path-col", default="file_name"), ap.add_argument("--label-col", default="label")
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("--k", type=int, default=25, help="features kept by selection")
    ap.add_argument("--all-features", action="store_true",
                    help="also use file format/size/dimension features (inflates scores on most datasets)")
    ap.add_argument("--extra-real", nargs="*", default=[], help="folder(s) of YOUR real photos (e.g. phone pictures)")
    ap.add_argument("--extra-ai", nargs="*", default=[], help="folder(s) of YOUR AI images from other generators")
    ap.add_argument("--extra-weight", type=int, default=5, help="repeat extra training images this many times")
    ap.add_argument("--clip", action="store_true",
                    help="add pretrained CLIP image embeddings and a stacked ensemble (needs torch + open_clip_torch)")
    ap.add_argument("--clip-model", default="ViT-B-16", help="open_clip model name, e.g. ViT-B-16 (fast) or ViT-L-14 (stronger, slower)")
    ap.add_argument("--out", default="models")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- 1. data
    if a.labels_csv:
        items = D.from_csv(a.labels_csv, a.data or ".", a.path_col, a.label_col)
    else:
        items = D.discover(a.data, a.real, a.ai)
    if not items:
        raise SystemExit("No labelled images found. Use --real/--ai folders or --labels-csv.")
    items = D.balance_sample(items, a.max_per_class)
    df = D.build_table(items, cache=os.path.join(a.out, "features_v2.csv"), n_jobs=a.jobs)
    feat_cols = [c for c in df.columns if c.startswith(("meta_", "vis_"))]
    if not a.all_features:
        feat_cols = [c for c in feat_cols if c not in FORMAT_FEATURES]
        print(f"[data] content-only mode: dropped {len(FORMAT_FEATURES)} format/size/dimension features")
    extra = None
    if a.extra_real or a.extra_ai:
        ex_items = []
        for d_ in a.extra_real:
            ex_items += D.discover(real_dir=d_)
        for d_ in a.extra_ai:
            ex_items += D.discover(ai_dir=d_)
        extra = D.build_table(ex_items, cache=os.path.join(a.out, "features_extra_v2.csv"), n_jobs=a.jobs)
        print(f"[extra] {len(extra)} own images: real={int((extra.label == 0).sum())}, ai={int((extra.label == 1).sum())}")
        if extra.label.nunique() < 2 or extra.label.value_counts().min() < 10:
            raise SystemExit("Need at least 10 images in each of --extra-real and --extra-ai.")
    X, y = df[feat_cols], df["label"].values
    print(f"[data] {len(df)} images, {len(feat_cols)} features, class balance {np.bincount(y)}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    k = min(a.k, len(feat_cols))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    # ---- 2. compare classifiers (selection happens inside each CV fold -> no leakage)
    rows, fitted, rocs = [], {}, {}
    for name, clf in make_models().items():
        t = time.time()
        pipe = make_pipe(clf, k)
        cvres = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=1)
        pipe.fit(Xtr, ytr)
        t_fit = time.time() - t
        t0 = time.time()
        proba = pipe.predict_proba(Xte)[:, 1]
        t_pred = (time.time() - t0) / len(Xte) * 1000
        pred = (proba >= 0.5).astype(int)
        fitted[name] = pipe
        rocs[name] = roc_curve(yte, proba)
        rows.append({
            "Model": name,
            "CV F1": cvres["test_f1"].mean(), "CV F1 sd": cvres["test_f1"].std(),
            "CV AUC": cvres["test_roc_auc"].mean(),
            "Test Acc": accuracy_score(yte, pred), "Test Prec": precision_score(yte, pred),
            "Test Recall": recall_score(yte, pred), "Test F1": f1_score(yte, pred),
            "Test AUC": roc_auc_score(yte, proba), "ms/image (model)": t_pred, "Train s": t_fit,
        })
        print(f"  {name:20s} CV-F1={rows[-1]['CV F1']:.4f}  Test-F1={rows[-1]['Test F1']:.4f}  AUC={rows[-1]['Test AUC']:.4f}")
    res = pd.DataFrame(rows).sort_values(["CV AUC", "CV F1"], ascending=False).reset_index(drop=True)
    res.to_csv(os.path.join(a.out, "cv_results.csv"), index=False)
    best = res.loc[0, "Model"]  # chosen on CV only - the hold-out stays untouched
    best_pipe = fitted[best]
    print(f"[model] best by cross-validation: {best}")

    # ---- 3. ablation: do metadata / visual features carry signal on their own?
    abl = []
    for grp, cols in {
        "Metadata only": [c for c in feat_cols if c.startswith("meta_")],
        "Visual only": [c for c in feat_cols if c.startswith("vis_")],
        "Metadata + Visual": feat_cols,
    }.items():
        p = make_pipe(make_models()[best], min(k, len(cols)))
        r = cross_validate(p, Xtr[cols], ytr, cv=cv, scoring=["f1", "roc_auc"])
        abl.append({"Feature set": grp, "n features": len(cols), "CV F1": r["test_f1"].mean(), "CV AUC": r["test_roc_auc"].mean()})
    abl = pd.DataFrame(abl)
    abl.to_csv(os.path.join(a.out, "ablation.csv"), index=False)

    # ---- 4. evaluation artefacts for the best model
    proba = best_pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(yte, pred)
    save_cm(cm, f"{best} - hold-out", os.path.join(a.out, "confusion_matrix.png"))
    save_roc(rocs, os.path.join(a.out, "roc_curves.png"))

    pi = permutation_importance(best_pipe, Xte, yte, n_repeats=5, random_state=SEED, scoring="roc_auc")
    imp = pd.Series(pi.importances_mean, index=feat_cols).sort_values(ascending=False)
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top.index, top.values, color="#3b6fb6")
    ax.set_xlabel("Permutation importance (drop in AUC)"), ax.set_title(f"Top features - {best}")
    fig.tight_layout(), fig.savefig(os.path.join(a.out, "feature_importance.png"), dpi=140), plt.close(fig)
    imp.to_csv(os.path.join(a.out, "feature_importance.csv"), header=["importance"])

    # ---- 4b. CLIP embeddings + stacked ensemble (optional)
    clip_on = a.clip
    clip_rows = ""
    if clip_on:
        from embed import embeddings_for
        E = embeddings_for(df["path"].tolist(), os.path.join(a.out, f"emb_{a.clip_model}.npz"), a.clip_model)
        Etr, Ete = E[Xtr.index.values], E[Xte.index.values]
        erows, efit = [], {}
        for name, clf in make_emb_models().items():
            p_ = make_emb_pipe(clf)
            cvr = cross_validate(p_, Etr, ytr, cv=cv, scoring=["f1", "roc_auc"], n_jobs=1)
            p_.fit(Etr, ytr)
            pr = p_.predict_proba(Ete)[:, 1]
            efit[name] = p_
            erows.append({"Model": f"CLIP + {name}", "CV F1": cvr["test_f1"].mean(), "CV AUC": cvr["test_roc_auc"].mean(),
                          "Test Acc": accuracy_score(yte, pr >= .5), "Test F1": f1_score(yte, pr >= .5),
                          "Test AUC": roc_auc_score(yte, pr)})
            print(f"  CLIP + {name:20s} CV-F1={erows[-1]['CV F1']:.4f}  Test-F1={erows[-1]['Test F1']:.4f}  AUC={erows[-1]['Test AUC']:.4f}")
        eres = pd.DataFrame(erows).sort_values("CV AUC", ascending=False).reset_index(drop=True)
        eres.to_csv(os.path.join(a.out, "cv_results_clip.csv"), index=False)
        best_emb = eres.loc[0, "Model"].replace("CLIP + ", "")
        pipe_B = efit[best_emb]
        oofA = cross_val_predict(make_pipe(make_models()[best], k), Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        oofB = cross_val_predict(make_emb_pipe(make_emb_models()[best_emb]), Etr, ytr, cv=cv, method="predict_proba")[:, 1]
        stacker = LogisticRegression(C=1.0).fit(stack_features(oofA, oofB), ytr)
        pA, pB = best_pipe.predict_proba(Xte)[:, 1], pipe_B.predict_proba(Ete)[:, 1]
        pE = stacker.predict_proba(stack_features(pA, pB))[:, 1]
        fam = []
        for nm, pp in ((f"Handcrafted + metadata ({best})", pA), (f"CLIP embeddings ({best_emb})", pB), ("Ensemble (stacked)", pE)):
            fam.append({"Model family": nm, "Test Acc": accuracy_score(yte, pp >= .5), "Test F1": f1_score(yte, pp >= .5),
                        "Test AUC": roc_auc_score(yte, pp)})
        fam = pd.DataFrame(fam)
        fam.to_csv(os.path.join(a.out, "model_families.csv"), index=False)
        print("[ensemble hold-out]\n" + fam.to_string(index=False))
        clip_rows = f"""
## Deep features (CLIP) and ensemble - hold-out
{md_table(eres)}

{md_table(fam)}

Stacker weights (logit of handcrafted, logit of CLIP): {np.round(stacker.coef_[0], 3).tolist()}
"""
        rocs[f"CLIP {best_emb}"] = roc_curve(yte, pB)
        rocs["Ensemble"] = roc_curve(yte, pE)
        proba = pE
        pred = (pE >= .5).astype(int)
        cm = confusion_matrix(yte, pred)
        save_cm(cm, "Ensemble - hold-out", os.path.join(a.out, "confusion_matrix.png"))
        save_roc(rocs, os.path.join(a.out, "roc_curves.png"))

    # ---- 5. refit on everything (+ your own images) and save a bundle
    ext_report = ""
    Xfit, yfit = X, y
    Efit = E if clip_on else None
    if extra is not None:
        ex_tr, ex_te = train_test_split(extra, test_size=0.3, stratify=extra.label, random_state=SEED)
        yx = ex_te.label.values
        ext_rows = []

        def sc(label, p):
            ext_rows.append((label, accuracy_score(yx, p >= .5), roc_auc_score(yx, p)))

        pxA = best_pipe.predict_proba(ex_te[feat_cols])[:, 1]
        sc("Handcrafted - dataset only", pxA)
        if clip_on:
            Eex = embeddings_for(extra["path"].tolist(), os.path.join(a.out, f"emb_extra_{a.clip_model}.npz"), a.clip_model)
            Eex_tr, Eex_te = Eex[ex_tr.index.values], Eex[ex_te.index.values]
            pxB = pipe_B.predict_proba(Eex_te)[:, 1]
            sc("CLIP - dataset only", pxB)
            sc("Ensemble - dataset only", stacker.predict_proba(stack_features(pxA, pxB))[:, 1])
        Xfit = pd.concat([X] + [ex_tr[feat_cols]] * a.extra_weight, ignore_index=True)
        yfit = np.concatenate([y] + [ex_tr.label.values] * a.extra_weight)
        tmpA = make_pipe(make_models()[best], k).fit(Xfit, yfit)
        pxA2 = tmpA.predict_proba(ex_te[feat_cols])[:, 1]
        sc("Handcrafted + your images", pxA2)
        if clip_on:
            Efit = np.concatenate([E] + [Eex_tr] * a.extra_weight)
            tmpB = make_emb_pipe(make_emb_models()[best_emb]).fit(Efit, yfit)
            pxB2 = tmpB.predict_proba(Eex_te)[:, 1]
            sc("CLIP + your images", pxB2)
            sc("Ensemble + your images", stacker.predict_proba(stack_features(pxA2, pxB2))[:, 1])
        print(f"[external test: {len(ex_te)} of YOUR images never used in training]")
        for lab, ac, au in ext_rows:
            print(f"   {lab:28s} accuracy {ac:.3f}  AUC {au:.3f}")
        ext_report = (f"\n## External test on own images ({len(ex_te)} images never used for training)\n"
                      "| Model | Accuracy | AUC |\n|---|---|---|\n" +
                      "".join(f"| {l} | {ac:.3f} | {au:.3f} |\n" for l, ac, au in ext_rows))
    final = make_pipe(make_models()[best], k).fit(Xfit, yfit)
    bundle_extra = {}
    if clip_on:
        final_B = make_emb_pipe(make_emb_models()[best_emb]).fit(Efit, yfit)
        bundle_extra = {"embed_pipeline": final_B, "stacker": stacker, "clip_model": a.clip_model, "embed_model_name": best_emb}
    stats = {}
    for c in feat_cols:
        v = Xfit[c]
        stats[c] = {"real_median": float(v[yfit == 0].median()), "ai_median": float(v[yfit == 1].median()),
                    "std": float(v.std() or 1.0)}
    selected = list(final[:-1].get_feature_names_out(feat_cols))
    metrics = {
        "model": best + (f" + CLIP {best_emb} (ensemble)" if clip_on else ""),
        "n_images": int(len(df)), "n_features": len(feat_cols), "k_selected": k,
        "selected_features": selected,
        "holdout": {"accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred),
                    "recall": recall_score(yte, pred), "f1": f1_score(yte, pred), "auc": roc_auc_score(yte, proba),
                    "confusion_matrix": cm.tolist()},
    }
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    joblib.dump({"pipeline": final, "feature_names": feat_cols, "model_name": best, "stats": stats,
                 "metrics": metrics, **bundle_extra}, os.path.join(a.out, "model.joblib"))

    # ---- 6. markdown report
    h = metrics["holdout"]
    rep = f"""# AI-Generated Image Detection - Evaluation Report

Images: **{len(df)}** (balanced) | Features mined: **{len(feat_cols)}** | Kept after selection (mutual information): **{k}**

## Classifier comparison (5-fold CV on 80% train, then 20% hold-out)
{md_table(res)}

**Selected model:** {best} (chosen on cross-validation AUC/F1; hold-out never used for selection)

## Hold-out performance of {metrics['model']}
Accuracy {h['accuracy']:.4f} | Precision {h['precision']:.4f} | Recall {h['recall']:.4f} | F1 {h['f1']:.4f} | AUC {h['auc']:.4f}

![confusion](confusion_matrix.png) ![roc](roc_curves.png)

## Ablation - where does the signal come from?
{md_table(abl)}

> If *Metadata only* scores far above *Visual only*, the dataset likely leaks through file-format/EXIF
> differences (e.g. all real images are camera JPEGs, all AI images are stripped PNGs). Treat that score with caution
> and rely on the visual-only figure for generalisation.

## Top features (permutation importance)
![importance](feature_importance.png)

{clip_rows}{ext_report}
## Selected features
{', '.join(selected)}
"""
    open(os.path.join(a.out, "report.md"), "w").write(rep)
    print(f"[done] artefacts written to {a.out}/")


if __name__ == "__main__":
    main()