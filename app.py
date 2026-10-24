"""Streamlit UI:  streamlit run app.py"""
import html
import io
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from explain import label_of
from predict import analyze, load_bundle

MODEL_DIR = "models"
RED, GREEN, AMBER = "#d64545", "#2e9e6b", "#d9962b"

st.set_page_config(page_title="AI Image Detector", page_icon="", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero h1 {font-size: 2.1rem; margin: 0 0 .15rem 0;}
.hero p  {opacity: .7; margin: 0 0 1.2rem 0;}
.verdict {border-radius: 14px; padding: 1.1rem 1.3rem; margin-bottom: .8rem;
          border: 1px solid rgba(128,128,128,.25); border-left-width: 8px;}
.verdict .tag  {font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; opacity: .7;}
.verdict .main {font-size: 1.7rem; font-weight: 700; line-height: 1.2;}
.verdict .conf {font-size: 1rem; opacity: .85;}
.v-ai   {border-left-color: #d64545; background: rgba(214,69,69,.08);}
.v-real {border-left-color: #2e9e6b; background: rgba(46,158,107,.08);}
.v-unsure {border-left-color: #d9962b; background: rgba(217,150,43,.08);}
.bar {height: 10px; border-radius: 6px; background: rgba(128,128,128,.25); overflow: hidden; margin: .5rem 0 .2rem;}
.bar > div {height: 100%;}
.scale {display: flex; justify-content: space-between; font-size: .75rem; opacity: .6;}
.reason {padding: .45rem .7rem; margin: .3rem 0; border-radius: 8px; background: rgba(128,128,128,.10);
         border-left: 4px solid rgba(128,128,128,.5); font-size: .93rem;}
.reason.ai {border-left-color: #d64545;} .reason.real {border-left-color: #2e9e6b;}
.chip {display: inline-block; padding: .15rem .6rem; margin: .15rem .25rem .15rem 0; border-radius: 999px;
       font-size: .8rem; background: rgba(128,128,128,.15);}
.small {font-size: .85rem; opacity: .7;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>AI-Generated Image Detector</h1>'
    "<p>Metadata mining + visual feature mining + an explainable classifier</p></div>",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_bundle():
    p = os.path.join(MODEL_DIR, "model.joblib")
    return load_bundle(p) if os.path.exists(p) else None


bundle = get_bundle()
if bundle is None:
    st.error("No trained model found. Run `python train.py --data <dataset folder>` first.")
    st.stop()

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Metadata** is read from the file (camera, software, AI tags, prompts).\n"
        "2. **Pixel features** are measured: noise, sharpness, texture, colour, frequency.\n"
        "3. A trained **classifier** gives the probability the image is AI-generated.\n"
        "4. The **reasons** show which features pushed the decision.\n"
    )
    st.divider()
    st.caption(f"Model: **{bundle.get('model_name', '-')}**")
    if "stacker" in bundle:
        st.caption("Ensemble: explainable model + CLIP deep features")
    st.divider()
    st.caption("A probabilistic aid, **not proof**. Resized, re-compressed or screenshot images lose cues, "
               "and generators not seen in training may be missed.")


def verdict_card(r):
    p = r["p_ai"]
    if 0.4 <= p <= 0.6:
        cls, title, color = "v-unsure", "Uncertain", AMBER
    elif r["is_ai"]:
        cls, title, color = "v-ai", "AI-generated", RED
    else:
        cls, title, color = "v-real", "Likely real photo", GREEN
    st.markdown(
        f'<div class="verdict {cls}"><div class="tag">Verdict</div>'
        f'<div class="main">{title}</div>'
        f'<div class="conf">Confidence {r["confidence"]:.0%}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="small">Probability the image is AI-generated: <b>{p:.0%}</b></div>'
        f'<div class="bar"><div style="width:{p * 100:.0f}%;background:{color}"></div></div>'
        '<div class="scale"><span>Real</span><span>AI</span></div>',
        unsafe_allow_html=True,
    )


def reasons_block(r):
    st.markdown("**Why this verdict**")
    items = r["reasons"] or [{"text": "No single feature dominated this decision."}]
    for x in items:
        cls = "ai" if r["is_ai"] else "real"      # reasons always support the verdict
        st.markdown(f'<div class="reason {cls}">{html.escape(str(x["text"]))}</div>', unsafe_allow_html=True)
    if len(r.get("components", {})) > 1:
        chips = "".join(f'<span class="chip">{html.escape(k)}: {v:.0%}</span>' for k, v in r["components"].items())
        st.markdown(f'<div class="small" style="margin-top:.6rem">Model votes (P(AI))</div>{chips}',
                    unsafe_allow_html=True)


def contribution_chart(r):
    top = r["contributions"].head(8)[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.barh([label_of(n) for n in top.index], top.values, color=[RED if v > 0 else GREEN for v in top.values])
    ax.axvline(0, color="gray", lw=.8)
    ax.set_xlabel("← towards REAL        towards AI →", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=200, bbox_inches="tight")
    plt.close(fig)
    st.image(buf)


tab1, tab2 = st.tabs(["🖼️ Analyse images", "📊 Model performance"])

with tab1:
    files = st.file_uploader("Drop one or more images here",
                             type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
                             accept_multiple_files=True)
    if not files:
        st.info("Upload an image to see the verdict, the reasons behind it and the metadata found in the file.")
    summary_rows = []
    for f in files or []:
        try:
            r = analyze(f.getvalue(), bundle)
        except Exception as e:
            st.warning(f"{f.name}: {e}")
            continue
        label = "AI-generated" if r["is_ai"] else "Real"
        summary_rows.append({"File": f.name, "Prediction": label, "P(AI)": r["p_ai"], "Confidence": r["confidence"]})
        with st.expander(f"{'🤖' if r['is_ai'] else '📷'}  {f.name}  —  {label} ({r['confidence']:.0%})",
                         expanded=len(files) == 1):
            left, right = st.columns([1, 1.25], gap="large")
            with left:
                st.image(r["rgb"], width="stretch")
            with right:
                verdict_card(r)
                reasons_block(r)
            m_tab, c_tab = st.tabs(["Metadata found in the file", "Feature contributions"])
            with m_tab:
                meta = pd.DataFrame({"Field": list(r["metadata"].keys()),
                                     "Value": [str(v) for v in r["metadata"].values()]})
                st.dataframe(meta, hide_index=True, width="stretch")
            with c_tab:
                st.caption("Red bars push the decision towards AI, green bars towards real.")
                contribution_chart(r)

    if len(summary_rows) > 1:
        st.subheader("Batch summary")
        sm = pd.DataFrame(summary_rows)
        a, b, c = st.columns(3)
        a.metric("Images", len(sm))
        b.metric("Flagged AI", int((sm["Prediction"] == "AI-generated").sum()))
        c.metric("Looks real", int((sm["Prediction"] == "Real").sum()))
        st.dataframe(
            sm, hide_index=True, width="stretch",
            column_config={
                "P(AI)": st.column_config.ProgressColumn("P(AI)", min_value=0.0, max_value=1.0, format="%.2f"),
                "Confidence": st.column_config.NumberColumn("Confidence", format="percent"),
            },
        )
        st.download_button("Download results (CSV)", sm.to_csv(index=False), "results.csv", "text/csv")

with tab2:
    mp = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(mp):
        st.info("Train a model to see its performance here.")
    else:
        m = json.load(open(mp))
        h = m["holdout"]
        st.caption(f"Model: **{m['model']}**  ·  {m['n_images']} images  ·  hold-out test set the model never trained on")
        a, b, c, d, e = st.columns(5)
        a.metric("Accuracy", f"{h['accuracy']:.1%}")
        b.metric("Precision", f"{h['precision']:.1%}")
        c.metric("Recall", f"{h['recall']:.1%}")
        d.metric("F1", f"{h['f1']:.3f}")
        e.metric("AUC", f"{h['auc']:.3f}")
        st.caption("Precision: of images called AI, how many are. Recall: of all AI images, how many were caught. "
                   "AUC: 0.5 is guessing, 1.0 is perfect.")
        for fn, title in (("cv_results.csv", "Classifier comparison"), ("ablation.csv", "Ablation: metadata vs visual")):
            pth = os.path.join(MODEL_DIR, fn)
            if os.path.exists(pth):
                st.subheader(title)
                st.dataframe(pd.read_csv(pth).round(3), hide_index=True, width="stretch")
        st.subheader("Charts")
        cols = st.columns(3)
        for col, (fn, cap) in zip(cols, (("confusion_matrix.png", "Confusion matrix"),
                                         ("roc_curves.png", "ROC curves"),
                                         ("feature_importance.png", "Feature importance"))):
            pth = os.path.join(MODEL_DIR, fn)
            if os.path.exists(pth):
                col.image(pth, caption=cap, width="stretch")