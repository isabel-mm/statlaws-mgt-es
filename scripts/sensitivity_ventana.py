# sensitivity_ventana.py
# Analiza la sensibilidad de los resultados al tamaño de ventana bootstrap.
# Para cada W ∈ {200, 400, 600, 800} tokens:
#   - Extrae rasgos (n=30 iteraciones bootstrap, semilla fija)
#   - Agrega por mediana por texto
#   - Evalúa con GroupKFold(k=5) — SVM (mejor modelo en protocolo principal)
#   - Reporta F1 macro, AUC y cobertura (textos incluidos)
# La longitud oracional se calcula sobre el texto completo (independiente de W).
import re
import warnings
import numpy as np
import pandas as pd
import spacy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from functions import (zipf_slope, heaps_beta, brevity_slope, lexical_stats,
                       lexical_diversity, freq_entropy, word_length_ratios,
                       zipf_coverage)

warnings.filterwarnings("ignore")

_nlp = spacy.load("es_core_news_sm",
                  disable=["morphologizer", "ner", "attribute_ruler", "lemmatizer"])

CORPUS_CSV       = "data/corpus_robot-talk_2026-02-04.csv"
WINDOW_SIZES     = [200, 400, 600, 800]
N_BOOTSTRAP      = 30       # menos iteraciones para rapidez; suficiente para sensibilidad
RANDOM_SEED      = 42
N_FOLDS          = 5
RANDOM_STATE     = 42

SELECTED_AUTHORS = [
    "humano", "bard", "claude-3-5-sonnet-20240620", "deepseek-v3",
    "gpt-3.5-turbo", "gpt-4", "mixtral-8x7b-instruct-v0.1",
]

ALL_STAT_COLS = [
    "zipf_slope", "zipf_r2", "zipf_top_cov",
    "heaps_beta", "heaps_k", "heaps_r2",
    "brevity_slope", "brevity_corr",
    "mean_word_length", "ttr",
    "hapax_rate",
    "long_word_ratio", "short_word_ratio",
    "mean_sent_len", "std_sent_len",
]


def tokenize(text):
    return re.findall(r'\b[a-záéíóúüñ]+\b', str(text).lower())


def sentence_stats(text):
    doc = _nlp(str(text))
    lengths = [len(re.findall(r'\b[a-záéíóúüñ]+\b', sent.text.lower()))
               for sent in doc.sents]
    lengths = [l for l in lengths if l > 0]
    if not lengths:
        return 0.0, 0.0
    return float(np.mean(lengths)), float(np.std(lengths))


def extract_features(tokens, mean_sl, std_sl):
    slope_z, r2_z         = zipf_slope(tokens)
    top_cov               = zipf_coverage(tokens)
    beta_h, k_h, r2_h     = heaps_beta(tokens)
    slope_b, r_b          = brevity_slope(tokens)
    mwl, ttr              = lexical_stats(tokens)
    hapax_r, herdan, yule = lexical_diversity(tokens)
    entropy               = freq_entropy(tokens)
    long_r, short_r       = word_length_ratios(tokens)
    return [slope_z, r2_z, top_cov, beta_h, k_h, r2_h, slope_b, r_b,
            mwl, ttr, hapax_r, herdan, yule, entropy, long_r, short_r,
            mean_sl, std_sl]


# Carga corpus
df_raw = pd.read_csv(CORPUS_CSV)
df_raw["author"]  = df_raw["tipo_version"].str.strip().str.lower()
df_raw["pair_id"] = df_raw["id_ref"].fillna(df_raw["id"]).astype(str)
df_raw["file"]    = df_raw["id"].astype(str)

df_raw = df_raw[
    (df_raw["dominio"] == "artículos")
    & (df_raw["ambito"] == "lingüística")
    & df_raw["author"].isin(SELECTED_AUTHORS)
].copy()

# Panel de 90 artículos comunes a todos los autores
refs_by = {a: set(df_raw.loc[df_raw["author"] == a, "pair_id"]) for a in SELECTED_AUTHORS}
common  = set.intersection(*(refs_by[a] for a in SELECTED_AUTHORS))
df_base = df_raw[df_raw["pair_id"].isin(common)].copy().reset_index(drop=True)
print(f"Panel base: {len(df_base)} textos ({df_base['author'].value_counts().to_dict()})")

# Pre-computar tokens y sentence_stats (independientes de W)
df_base["tokens"]    = df_base["text"].fillna("").map(tokenize)
df_base["n_tokens"]  = df_base["tokens"].map(len)
df_base["corpus"]    = df_base["author"].map(lambda a: "humano" if a == "humano" else "robot")
sl_vals = df_base["text"].fillna("").map(sentence_stats)
df_base["mean_sl"] = sl_vals.map(lambda x: x[0])
df_base["std_sl"]  = sl_vals.map(lambda x: x[1])

# Análisis por tamaño de ventana
rng     = np.random.default_rng(RANDOM_SEED)
results = []

print("\nSensibilidad al tamaño de ventana — SVM (GroupKFold, k=5)")
print(f"{'W (tokens)':>10}  {'Textos':>7}  {'F1':>8}  {'AUC':>8}")

for W in WINDOW_SIZES:
    subset = df_base[df_base["n_tokens"] >= W].copy().reset_index(drop=True)
    n_texts = len(subset)

    rows = []
    for _, row in subset.iterrows():
        tokens = row["tokens"]
        max_start = len(tokens) - W
        starts = rng.integers(0, max_start + 1, size=N_BOOTSTRAP)
        feat_list = []
        for s in starts:
            window = tokens[s: s + W]
            try:
                feats = extract_features(window, row["mean_sl"], row["std_sl"])
                feat_list.append(feats)
            except Exception:
                continue
        if not feat_list:
            continue
        median_feats = np.median(feat_list, axis=0).tolist()
        rows.append({
            "file":    row["file"],
            "pair_id": row["pair_id"],
            "corpus":  row["corpus"],
            **dict(zip(ALL_STAT_COLS, median_feats)),
        })

    df_w = pd.DataFrame(rows)
    if len(df_w) < 10:
        print(f"  W={W}: insuficientes textos ({len(df_w)}), se omite.")
        continue

    encoder = LabelEncoder()
    y       = encoder.fit_transform(df_w["corpus"].values)
    X       = df_w[ALL_STAT_COLS].values
    groups  = df_w["pair_id"].values

    gkf  = GroupKFold(n_splits=N_FOLDS)
    f1s, aucs = [], []
    for tr, te in gkf.split(X, y, groups):
        sc  = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        clf  = SVC(kernel="rbf", probability=True, class_weight="balanced",
                   random_state=RANDOM_STATE)
        clf.fit(X_tr, y[tr])
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
        f1s.append(f1_score(y[te], y_pred, average="macro"))
        aucs.append(roc_auc_score(y[te], y_prob))

    f1m, f1s_std = np.mean(f1s), np.std(f1s)
    aucm, auc_std = np.mean(aucs), np.std(aucs)
    print(f"  W={W:4d}  textos={n_texts:4d}  F1={f1m:.3f}±{f1s_std:.3f}  AUC={aucm:.3f}±{auc_std:.3f}")
    results.append({"W": W, "n_textos": n_texts, "F1": f1m, "F1_std": f1s_std,
                    "AUC": aucm, "AUC_std": auc_std})

df_res = pd.DataFrame(results)
df_res.to_csv("results/sensitivity_ventana.csv", index=False)
print("\nGuardado: results/sensitivity_ventana.csv")
