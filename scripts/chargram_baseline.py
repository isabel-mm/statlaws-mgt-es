# chargram_baseline.py - línea base SVM con n-gramas de caracteres (TF-IDF) y F1 por clase en T2
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

CORPUS_CSV = "data/corpus_robot-talk_2026-02-04.csv"
STATS_CSV  = "results/estadisticas_por_texto.csv"

SELECTED_AUTHORS = [
    "humano", "bard", "claude-3-5-sonnet-20240620",
    "deepseek-v3", "gpt-3.5-turbo", "gpt-4", "mixtral-8x7b-instruct-v0.1",
]

import re
def normalise_author(s):
    return (s or "").strip().lower()

def tokenize(text):
    return re.findall(r'\b[a-záéíóúüñ]+\b', text.lower())

# 1. Cargar corpus y alinear con el fichero de rasgos
df_raw = pd.read_csv(CORPUS_CSV)
df_raw["author"]  = df_raw["tipo_version"].map(normalise_author)
df_raw["genre"]   = df_raw["dominio"].fillna("desconocido").str.strip()
df_raw["field"]   = df_raw["ambito"].fillna("desconocido").str.strip()
df_raw["pair_id"] = df_raw["id_ref"].fillna(df_raw["id"]).astype(str)
df_raw["file"]    = df_raw["id"].astype(str)
df_raw["n_tokens"] = df_raw["text"].fillna("").map(lambda x: len(tokenize(str(x))))

df_filt = df_raw[
    (df_raw["genre"] == "artículos")
    & (df_raw["field"] == "lingüística")
    & (df_raw["author"].isin(SELECTED_AUTHORS))
    & (df_raw["n_tokens"] >= 400)
].copy()

refs_by_author = {a: set(df_filt.loc[df_filt["author"] == a, "pair_id"])
                  for a in SELECTED_AUTHORS}
common_refs = set.intersection(*(refs_by_author[a] for a in SELECTED_AUTHORS))
df_filt = df_filt[df_filt["pair_id"].isin(common_refs)].copy()

# 2. Cargar rasgos para alinear el orden de pair_id
df_stats = pd.read_csv(STATS_CSV)
# añadir el texto a cada fila de rasgos
df_merged = df_stats.merge(
    df_filt[["file", "text"]],
    on="file", how="left"
).dropna(subset=["text"])

texts  = df_merged["text"].fillna("").tolist()
corpus = df_merged["corpus"].values          # "humano" / "robot"
author = df_merged["author"].values
groups = df_merged["pair_id"].values

bin_enc = LabelEncoder()
y_bin   = bin_enc.fit_transform(corpus)      # humano=0, robot=1

# 3. T1 binary — GroupShuffleSplit 80/20
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
idx_tr, idx_te = next(splitter.split(texts, y_bin, groups=groups))

pipe_char = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                              max_features=50000, sublinear_tf=True)),
    ("clf",   CalibratedClassifierCV(LinearSVC(class_weight="balanced",
                                               max_iter=5000), cv=3)),
])

print("=" * 70)
print("CHAR N-GRAM SVM — T1 (binario, 80/20)")
print("=" * 70)
pipe_char.fit([texts[i] for i in idx_tr], y_bin[idx_tr])
y_pred = pipe_char.predict([texts[i] for i in idx_te])
y_prob = pipe_char.predict_proba([texts[i] for i in idx_te])[:, 1]

f1  = f1_score(y_bin[idx_te], y_pred, average="macro")
auc = roc_auc_score(y_bin[idx_te], y_prob)
cm  = confusion_matrix(y_bin[idx_te], y_pred)
fp, tn = cm[0, 1], cm[0, 0]
fpr = fp / (fp + tn) if (fp + tn) else 0.0
print(f"F1 macro : {f1:.3f}")
print(f"AUC      : {auc:.3f}")
print(f"FPR TH→TR: {fpr:.3f}")
print(classification_report(y_bin[idx_te], y_pred, target_names=bin_enc.classes_))

# 4. T1 GroupKFold (k=5) para comparación directa con paper
gkf = GroupKFold(n_splits=5)
f1s, aucs, fprs = [], [], []
for tr, te in gkf.split(texts, y_bin, groups):
    clf = clone(pipe_char)
    clf.fit([texts[i] for i in tr], y_bin[tr])
    pred = clf.predict([texts[i] for i in te])
    prob = clf.predict_proba([texts[i] for i in te])[:, 1]
    f1s.append(f1_score(y_bin[te], pred, average="macro"))
    aucs.append(roc_auc_score(y_bin[te], prob))
    cm_ = confusion_matrix(y_bin[te], pred)
    fp_, tn_ = cm_[0, 1], cm_[0, 0]
    fprs.append(fp_ / (fp_ + tn_) if (fp_ + tn_) else 0.0)

print("\n" + "=" * 70)
print("CHAR N-GRAM SVM — T1 GroupKFold (k=5)")
print("=" * 70)
print(f"F1 macro : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
print(f"AUC      : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
print(f"FPR TH→TR: {np.mean(fprs):.3f} ± {np.std(fprs):.3f}")

# 5. T2 multiclase: F1 por clase con GroupKFold
auth_enc = LabelEncoder()
y_auth   = auth_enc.fit_transform(author)
author_labels = list(auth_enc.classes_)

LABEL_MAP = {
    "bard": "Bard",
    "claude-3-5-sonnet-20240620": "Claude 3.5 Sonnet",
    "deepseek-v3": "DeepSeek-V3",
    "gpt-3.5-turbo": "GPT-3.5-turbo",
    "gpt-4": "GPT-4",
    "humano": "Humano",
    "mixtral-8x7b-instruct-v0.1": "Mixtral-8x7B",
}

print("\n" + "=" * 70)
print("T2 PER-CLASS F1 — estadístico-lingüístico (15 rasgos, SVM, GroupKFold k=5)")
print("=" * 70)

# reload stats features
METRICS = [
    "zipf_slope", "zipf_r2", "zipf_top_cov",
    "heaps_beta", "heaps_k", "heaps_r2",
    "brevity_slope", "brevity_corr",
    "mean_word_length", "ttr",
    "hapax_rate",
    "long_word_ratio", "short_word_ratio",
    "mean_sent_len", "std_sent_len",
]
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X_stats = df_merged[METRICS].values
pipe_stats = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    SVC(kernel="rbf", probability=True, class_weight="balanced")),
])

all_true, all_pred = [], []
for tr, te in gkf.split(X_stats, y_auth, groups):
    clf = clone(pipe_stats)
    clf.fit(X_stats[tr], y_auth[tr])
    all_true.extend(y_auth[te])
    all_pred.extend(clf.predict(X_stats[te]))

print(classification_report(all_true, all_pred,
                             target_names=author_labels, digits=3))

# guardar F1 por clase
report = classification_report(all_true, all_pred,
                                target_names=author_labels,
                                digits=3, output_dict=True)
rows = []
for cls in author_labels:
    rows.append({
        "author": cls,
        "label":  LABEL_MAP.get(cls, cls),
        "precision": report[cls]["precision"],
        "recall":    report[cls]["recall"],
        "f1":        report[cls]["f1-score"],
        "support":   int(report[cls]["support"]),
    })
df_perclass = pd.DataFrame(rows)
df_perclass.to_csv("results/t2_perclass_f1.csv", index=False)
print("\nGuardado: results/t2_perclass_f1.csv")
