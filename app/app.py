"""Streamlit web app for SMS / e-mail spam detection.

Run from the repository root:
    streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from model import MODEL_NAME, SpamClassifier  # noqa: E402

METRICS_PATH = APP_DIR / "metrics.json"

st.set_page_config(page_title="Spam Detector", page_icon="📨", layout="centered")


@st.cache_resource(show_spinner="Loading pre-trained model (first run downloads ~470 MB)...")
def load_model() -> SpamClassifier:
    return SpamClassifier()


def load_metrics() -> dict | None:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return None


metrics = load_metrics()
# The best-F1 threshold found in the notebook (~0.999) gives almost no gain over 0.5,
# so 0.5 (the model's natural cut-off) is used as default.
default_threshold = 0.5

st.title("📨 Spam Detector")
st.caption(
    f"Pre-trained transformer [`{MODEL_NAME}`](https://huggingface.co/{MODEL_NAME}) "
    "evaluated on the UCI SMS Spam Collection."
)

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Decision threshold (P(spam) ≥ threshold → spam)",
        min_value=0.01,
        max_value=0.999,
        value=default_threshold,
        step=0.001,
        format="%.3f",
    )
    st.caption(
        "The model was fine-tuned on e-mail subjects. On SMS it is weaker "
        "(see the *Model performance* tab and the notebook)."
    )

model = load_model()

tab_single, tab_batch, tab_about = st.tabs(["Single message", "Batch file", "Model performance"])

# --------------------------------------------------------------------------- single
with tab_single:
    examples = {
        "— choose an example —": "",
        "Prize (spam)": "WINNER!! You have been selected to receive a £900 prize reward! "
        "To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
        "Friend (ham)": "Hey, are we still meeting for lunch tomorrow at 1pm?",
        "Spanish (spam)": "¡Felicidades! Ganaste un iPhone gratis. Haz clic aquí para reclamarlo hoy.",
        "Spanish (ham)": "Hola mamá, ya llegué a la casa. Mañana te llamo.",
    }
    choice = st.selectbox("Examples", list(examples))
    text = st.text_area("Message", value=examples[choice], height=140,
                        placeholder="Paste an SMS or e-mail subject here...")
    if st.button("Classify", type="primary", disabled=not text.strip()):
        labels, probs = model.predict([text], threshold=threshold)
        p = float(probs[0])
        if labels[0] == "spam":
            st.error(f"🚫 **SPAM** — probability {p:.1%}")
        else:
            st.success(f"✅ **HAM (not spam)** — spam probability {p:.1%}")
        st.progress(p, text=f"P(spam) = {p:.3f} · threshold = {threshold:.2f}")

# --------------------------------------------------------------------------- batch
with tab_batch:
    st.write(
        "Upload a **CSV** (choose the text column) or a **TXT** file with one message per line."
    )
    up = st.file_uploader("File", type=["csv", "txt"])
    if up is not None:
        if up.name.lower().endswith(".csv"):
            df = pd.read_csv(up, encoding_errors="replace")
            col = st.selectbox("Text column", df.columns)
        else:
            lines = up.read().decode("utf-8", errors="replace").splitlines()
            df = pd.DataFrame({"text": [line for line in lines if line.strip()]})
            col = "text"
        st.caption(f"{len(df):,} rows loaded")
        if st.button("Classify file", type="primary"):
            with st.spinner("Classifying..."):
                labels, probs = model.predict(df[col].astype(str).tolist(), threshold=threshold)
            out = df.copy()
            out["prediction"] = labels
            out["spam_probability"] = probs.round(4)
            c1, c2 = st.columns(2)
            c1.metric("Spam", int((labels == "spam").sum()))
            c2.metric("Ham", int((labels == "ham").sum()))
            st.dataframe(out, width="stretch", hide_index=True)
            st.download_button(
                "Download results (CSV)",
                out.to_csv(index=False).encode("utf-8"),
                file_name="spam_predictions.csv",
                mime="text/csv",
            )

# --------------------------------------------------------------------------- metrics
with tab_about:
    if not metrics:
        st.info("No metrics yet. Execute `notebook/spam_detection.ipynb` to generate `app/metrics.json`.")
    else:
        st.subheader("Evaluation on the SMS Spam Collection")
        st.caption(
            f"{metrics['n_messages']:,} unique messages · {metrics['n_spam']:,} spam · "
            f"{metrics['n_ham']:,} ham"
        )
        rows = []
        for name, key in [("Default threshold 0.50", "threshold_0_5"),
                          (f"Best-F1 threshold {metrics['recommended_threshold']:.4f}", "threshold_best")]:
            m = metrics[key]
            rows.append({"Setting": name, "Accuracy": m["accuracy"], "Precision": m["precision"],
                         "Recall": m["recall"], "F1": m["f1"]})
        st.dataframe(pd.DataFrame(rows).style.format(precision=3), hide_index=True,
                     width="stretch")
        c1, c2 = st.columns(2)
        c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        c2.metric("PR-AUC", f"{metrics['average_precision']:.3f}")
        st.warning(
            "Out-of-the-box transfer from e-mail subjects to SMS is weak: informal short ham "
            "messages are often flagged as spam and short-code SMS spam is missed. "
            "Well-formed sentences (English or Spanish) are handled much better."
        )
        cm = metrics["threshold_0_5"]["confusion_matrix"]
        st.write("Confusion matrix (threshold 0.50)")
        st.dataframe(
            pd.DataFrame(cm, index=["true ham", "true spam"], columns=["pred ham", "pred spam"]),
            width="stretch",
        )
