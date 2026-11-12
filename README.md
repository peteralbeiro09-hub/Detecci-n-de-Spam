# 📨 Spam Detection with a Pre-trained Transformer

Web application and data-science study that classifies short text messages as **spam** or **ham** (legitimate) using the pre-trained Hugging Face model [`Goodmotion/spam-mail-classifier`](https://huggingface.co/Goodmotion/spam-mail-classifier). The model is evaluated on the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

## Team

| Member | Role |
|---|---|
| Pedro Jojoa | Data analysis, modelling and web app |

## Project structure

```
├── app/
│   ├── app.py            # Streamlit web application
│   ├── model.py          # Wrapper around the pre-trained Hugging Face model
│   └── metrics.json      # Evaluation metrics exported by the notebook
├── data/
│   ├── SMSSpamCollection # Dataset (label<TAB>text)
│   └── readme            # Original dataset description (UCI)
├── notebook/
│   └── spam_detection.ipynb  # Step-by-step data-science process (EDA + tests)
├── requirements.txt
└── README.md
```

## Data

**SMS Spam Collection** (Almeida & Gómez Hidalgo, UCI Machine Learning Repository): 5,574 English SMS messages labelled `ham` / `spam`. After removing 403 exact duplicates, 5,171 unique messages remain (4,518 ham – 87.4 %, 653 spam – 12.6 %).

## Model

[`Goodmotion/spam-mail-classifier`](https://huggingface.co/Goodmotion/spam-mail-classifier) — `microsoft/Multilingual-MiniLM-L12-H384` (≈118 M parameters) fine-tuned for binary sequence classification on **e-mail subjects** (output index 0 = NOSPAM, 1 = SPAM, `max_length = 128`, Apache-2.0). It is used **as-is (no re-training)**; the whole SMS dataset is the test set.

## Data-science process (notebook)

1. Problem definition
2. Data loading and quality checks (missing values, duplicates)
3. EDA: class balance, message length, spam signals (URLs, phone numbers, money words, uppercase), top words and word clouds
4. Preprocessing (raw text, label mapping, tokenization with truncation at 128 tokens)
5. Batch inference with the pre-trained model
6. Evaluation: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, ROC / PR curves, threshold tuning
7. Error analysis (false positives / negatives, performance by length) and manual tests in English and Spanish
8. Conclusions and export of `app/metrics.json`

## Results

| Setting | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
|---|---|---|---|---|
| Threshold 0.50 | 0.428 | 0.148 | 0.740 | 0.246 |
| Best-F1 threshold (≈0.999) | 0.674 | 0.186 | 0.469 | 0.266 |

ROC-AUC = **0.614** · PR-AUC = **0.175**

**Key findings**

- Out of the box, the model **does not transfer well from e-mail subjects to SMS**: it is over-confident, flags many short informal ham messages (*"Are you free now? can i call now?"*) as spam and misses classic SMS spam built on short codes and premium numbers.
- Threshold tuning barely helps, because the weakness is in the ranking itself (domain shift), not in the cut-off.
- On clean, well-formed sentences — in **English and Spanish** thanks to the multilingual backbone — the manual tests are classified correctly with high confidence.
- Next steps: fine-tune the backbone on an SMS train split, compare with a TF-IDF + logistic-regression baseline, and calibrate the probabilities.

## Web application

Built with **Streamlit** (`app/app.py`):

- **Single message**: paste a text (or pick an example) and get the label and the spam probability.
- **Batch file**: upload a CSV (choose the text column) or a TXT file (one message per line), classify all rows and download the results as CSV.
- **Model performance**: metrics and confusion matrix computed in the notebook.
- Adjustable decision threshold in the sidebar (default 0.5).

## How to run

```bash
# 1. Create and activate a virtual environment (Python 3.13)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the web app (the model, ~470 MB, is downloaded on first run)
streamlit run app/app.py

# 4. (Optional) Re-run the notebook end to end
jupyter nbconvert --to notebook --execute --inplace notebook/spam_detection.ipynb
```

## References

- Almeida, T.A., Gómez Hidalgo, J.M. (2011). *SMS Spam Collection v.1*. UCI Machine Learning Repository.
- Goodmotion. *spam-mail-classifier*. Hugging Face Hub.
- Wang, W. et al. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*.
