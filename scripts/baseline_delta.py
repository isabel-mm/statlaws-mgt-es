# baseline_delta.py
# Baseline estilométrico: Delta de Burrows sobre palabras funcionales.
# Las top-200 palabras más frecuentes del corpus (mayoritariamente funcionales
# en español) se z-normalizan y se clasifican con LR, SVM y RF.
# El protocolo es idéntico al de ablacion.py: GroupKFold(5) por pair_id.
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from collections import Counter

warnings.filterwarnings("ignore")

CORPUS_CSV   = "data/corpus_robot-talk_2026-02-04.csv"
STAT_CSV     = "results/estadisticas_textos.csv"
N_WORDS      = 200
N_FOLDS      = 5
RANDOM_STATE = 42

SELECTED_AUTHORS = [
    "humano",
    "bard",
    "claude-3-5-sonnet-20240620",
    "deepseek-v3",
    "gpt-3.5-turbo",
    "gpt-4",
    "mixtral-8x7b-instruct-v0.1",
]

CORPUS_LABEL = {
    "humano": "TH",
}


def tokenize(text):
    return re.findall(r'\b[a-záéíóúüñ]+\b', str(text).lower())


def normalise_author(v):
    return (v or "").strip().lower()


# 1. Cargar corpus
df_raw = pd.read_csv(CORPUS_CSV)
df_raw["author"]  = df_raw["tipo_version"].map(normalise_author)
df_raw["pair_id"] = df_raw["id_ref"].fillna(df_raw["id"]).astype(str)
df_raw["file"]    = df_raw["id"].astype(str)

df_raw = df_raw[
    (df_raw["dominio"] == "artículos")
    & (df_raw["ambito"] == "lingüística")
    & (df_raw["author"].isin(SELECTED_AUTHORS))
].copy()

# Reproducir selección de los 90 artículos comunes
refs_by_author = {
    a: set(df_raw.loc[df_raw["author"] == a, "pair_id"])
    for a in SELECTED_AUTHORS
}
common_refs = set.intersection(*(refs_by_author[a] for a in SELECTED_AUTHORS))
print(f"Artículos base comunes: {len(common_refs)}")

df = df_raw[df_raw["pair_id"].isin(common_refs)].copy().reset_index(drop=True)
print(f"Textos: {len(df)}  |  autores: {df['author'].value_counts().to_dict()}")

# 2. Construir vocabulario top-N
all_tokens: list[str] = []
for text in df["text"].fillna(""):
    all_tokens.extend(tokenize(text))

counter = Counter(all_tokens)
top_words = [w for w, _ in counter.most_common(N_WORDS)]
print(f"\nTop-{N_WORDS} palabras más frecuentes (muestra):", top_words[:30])

# 3. Matriz de frecuencias relativas
freq_matrix = np.zeros((len(df), N_WORDS), dtype=float)
for i, text in enumerate(df["text"].fillna("")):
    tokens = tokenize(text)
    if not tokens:
        continue
    c = Counter(tokens)
    total = len(tokens)
    for j, w in enumerate(top_words):
        freq_matrix[i, j] = c.get(w, 0) / total

# 4. Z-score (Delta de Burrows)
mu    = freq_matrix.mean(axis=0)
sigma = freq_matrix.std(axis=0)
sigma[sigma == 0] = 1.0
X_delta = (freq_matrix - mu) / sigma

# 5. Etiquetas corpus (TH vs TR)
df["corpus"] = np.where(df["author"] == "humano", "TH", "TR")
encoder = LabelEncoder()
y       = encoder.fit_transform(df["corpus"].values)
groups  = df["pair_id"].values

print(f"\nClases: {dict(zip(encoder.classes_, np.bincount(y)))}")

# 6. Modelos
def make_models():
    return {
        "LR":  LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced",
                   random_state=RANDOM_STATE),
        "RF":  RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                      random_state=RANDOM_STATE),
    }

# 7. Validación cruzada GroupKFold
gkf     = GroupKFold(n_splits=N_FOLDS)
results = []

print(f"\nDelta de Burrows (top-{N_WORDS} palabras) — GroupKFold(k={N_FOLDS})")
for mname, clf in make_models().items():
    f1s, aucs = [], []
    for tr, te in gkf.split(X_delta, y, groups):
        c = clf.__class__(**clf.get_params())
        c.fit(X_delta[tr], y[tr])
        pred = c.predict(X_delta[te])
        prob = c.predict_proba(X_delta[te])[:, 1]
        f1s.append(f1_score(y[te], pred, average="macro"))
        aucs.append(roc_auc_score(y[te], prob))
    row = {
        "Condición": f"Delta Burrows (top-{N_WORDS})",
        "Modelo": mname,
        "n_rasgos": N_WORDS,
        "F1 macro": np.mean(f1s),
        "F1 std":   np.std(f1s),
        "AUC":      np.mean(aucs),
        "AUC std":  np.std(aucs),
    }
    results.append(row)
    print(f"  {mname:3s} | F1={row['F1 macro']:.3f}±{row['F1 std']:.3f}  AUC={row['AUC']:.3f}±{row['AUC std']:.3f}")

# 8. Guardar y comparar con ablación
df_delta = pd.DataFrame(results)
df_delta.to_csv("results/baseline_delta.csv", index=False)

try:
    df_abl = pd.read_csv("results/ablacion.csv")
    full_stat = df_abl[df_abl["Condición"] == "Estadísticos (todos)"].copy()
    print(f"\nComparación directa")
    print(f"{'Condición':35s}  {'Modelo':3s}  {'F1 macro':>8}  {'AUC':>6}")
    for _, r in pd.concat([df_delta, full_stat]).iterrows():
        print(f"  {r['Condición']:33s}  {r['Modelo']:3s}  {r['F1 macro']:.3f}±{r['F1 std']:.3f}  {r['AUC']:.3f}±{r['AUC std']:.3f}")
except FileNotFoundError:
    print("\n(results/ablacion.csv no encontrado; ejecuta ablacion.py primero)")

print("\nGuardado: results/baseline_delta.csv")
