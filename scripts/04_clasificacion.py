# 04_clasificacion.py
# Identificación multiclase de autor (T2: humano + 6 modelos) con las 15 métricas.
# Requiere haber ejecutado 02_extraccion_rasgos.py primero.
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score

RUTA_CSV = "results/estadisticas_textos.csv"
METRICS = [
    "zipf_slope", "zipf_r2", "zipf_top_cov",
    "heaps_beta", "heaps_k", "heaps_r2",
    "brevity_slope", "brevity_corr",
    "mean_word_length", "ttr",
    "hapax_rate",
    "long_word_ratio", "short_word_ratio",
    "mean_sent_len", "std_sent_len",
]

# 1. Cargar y agregar (mediana de iteraciones bootstrap por texto)
df_raw = pd.read_csv(RUTA_CSV)
df = (
    df_raw.groupby(["file", "pair_id", "genre", "corpus", "author"])[METRICS]
    .median()
    .reset_index()
)

print(f"Textos: {len(df)} | {df['corpus'].value_counts().to_dict()}")
print(f"Géneros: {df['genre'].value_counts().to_dict()}\n")

# 2. Preparar X
X = df[METRICS].values

# 3. Modelos
modelos = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced")),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)),
    ]),
}

# 4. Identificación multiclase de autor
print("\n" + "=" * 70)
print("IDENTIFICACIÓN MULTICLASE DE AUTOR — GroupKFold(pair_id)")
print("=" * 70)
author_encoder = LabelEncoder()
y_author = author_encoder.fit_transform(df["author"].values)
groups = df["pair_id"].values
gkf = GroupKFold(n_splits=5)
multi_rows = []
for nombre, pipe in modelos.items():
    f1s, accs = [], []
    for tr, te in gkf.split(X, y_author, groups):
        clf = clone(pipe)
        clf.fit(X[tr], y_author[tr])
        pred = clf.predict(X[te])
        f1s.append(f1_score(y_author[te], pred, average="macro"))
        accs.append(accuracy_score(y_author[te], pred))
    row = {
        "Modelo": nombre,
        "F1 macro": np.mean(f1s),
        "F1 std": np.std(f1s),
        "Accuracy": np.mean(accs),
        "Accuracy std": np.std(accs),
    }
    multi_rows.append(row)
    print(f"{nombre}: F1={row['F1 macro']:.3f}±{row['F1 std']:.3f} | Acc={row['Accuracy']:.3f}±{row['Accuracy std']:.3f}")
pd.DataFrame(multi_rows).to_csv("results/clasificacion_multiclase_autor.csv", index=False)

# 5. Guardar dataset agregado final
df.to_csv("results/estadisticas_por_texto.csv", index=False)
print("\nCSVs guardados en results/")
