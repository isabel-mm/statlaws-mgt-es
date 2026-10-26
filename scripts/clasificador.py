# clasificador.py
# Clasifica TH vs TR usando las métricas estilométricas.
# Requiere haber ejecutado demo.py primero.
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score

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

# 2. Preparar X, y
encoder = LabelEncoder()
y = encoder.fit_transform(df["corpus"].values)   # humano=0, robot=1 (orden alfabético)
X = df[METRICS].values
idx_all = np.arange(len(df))

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
idx_train, idx_test = next(splitter.split(X, y, groups=df["pair_id"].values))
X_train, X_test = X[idx_train], X[idx_test]
y_train, y_test = y[idx_train], y[idx_test]

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

# 4. Evaluación global (train 80 / test 20)
trained = {}
for nombre, pipe in modelos.items():
    print("=" * 70)
    print(f"Modelo: {nombre}")
    print("=" * 70)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)

    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    auc = roc_auc_score(y_test, y_prob[:, 1])
    print(f"AUC: {auc:.4f}\n")

    cm = confusion_matrix(y_test, y_pred)
    fp = cm[0, 1]
    tn = cm[0, 0]
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    print(f"FPR humano→robot: {fpr:.4f}\n")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(encoder.classes_)), labels=encoder.classes_)
    ax.set_yticks(np.arange(len(encoder.classes_)), labels=encoder.classes_)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(f"Matriz de confusión — {nombre}")
    plt.tight_layout()
    plt.savefig(f"results/confusion_{nombre.replace(' ', '_')}.png", dpi=150)
    plt.close()

    trained[nombre] = pipe

# 5. Evaluación por género (Random Forest)
print("\n" + "=" * 70)
print("EVALUACIÓN POR GÉNERO — Random Forest")
print("=" * 70)
rf = trained["Random Forest"]
test_meta = df.iloc[idx_test].copy().reset_index(drop=True)
test_meta["y_true"] = y_test
test_meta["y_pred"] = rf.predict(X_test)

for genre in sorted(test_meta["genre"].unique()):
    sub = test_meta[test_meta["genre"] == genre]
    if len(sub) < 4:
        continue
    print(f"\n  Género: {genre} (n={len(sub)})")
    print(classification_report(
        sub["y_true"], sub["y_pred"],
        target_names=encoder.classes_, zero_division=0,
    ))

# 6. Importancia de características (RF entrenado en todo el dataset)
print("\n" + "=" * 70)
print("IMPORTANCIA DE CARACTERÍSTICAS — RF (dataset completo)")
print("=" * 70)
rf_full = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
rf_full.fit(X, y)
for i in np.argsort(rf_full.feature_importances_)[::-1]:
    print(f"  {METRICS[i]}: {rf_full.feature_importances_[i]:.4f}")

# 6b. Identificación multiclase de autor
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

# 7. Escala de certeza basada en Likelihood Ratios (Random Forest)
print("\n" + "=" * 70)
print("ESCALA DE CERTEZA (LR) — Random Forest")
print("=" * 70)
pipe = trained["Random Forest"]
probs = pipe.predict_proba(X_test)
clases = pipe.named_steps["clf"].classes_
label_map = dict(zip(clases, encoder.classes_))
label0, label1 = label_map[0], label_map[1]

p0, p1 = probs[:, 0], probs[:, 1]
odds_ratio = np.divide(p1, p0, out=np.zeros_like(p1), where=p0 != 0)
log_odds = np.log(np.clip(odds_ratio, 1e-10, None))


def lr_scale(lr):
    if lr > 10000:  return f"Evidencia muy fuerte para {label1} (LR > 10 000)"
    if lr > 1000:   return f"Evidencia fuerte para {label1} (1 000–10 000)"
    if lr > 100:    return f"Evidencia mod. fuerte para {label1} (100–1 000)"
    if lr > 10:     return f"Evidencia moderada para {label1} (10–100)"
    if lr > 1:      return f"Evidencia limitada para {label1} (1–10)"
    if lr > 0.1:    return f"Evidencia limitada para {label0} (0,1–1)"
    if lr > 0.01:   return f"Evidencia moderada para {label0} (0,01–0,1)"
    if lr > 0.001:  return f"Evidencia mod. fuerte para {label0} (0,001–0,01)"
    return f"Evidencia muy fuerte para {label0} (< 0,001)"


certainty_df = pd.DataFrame({
    "file":            df.iloc[idx_test]["file"].values,
    "genre":           df.iloc[idx_test]["genre"].values,
    "true_label":      encoder.classes_[y_test],
    "predicted_label": encoder.classes_[pipe.predict(X_test)],
    f"prob_{label1}":  p1,
    f"prob_{label0}":  p0,
    "odds_ratio":      odds_ratio,
    "log_odds":        log_odds,
})
certainty_df["LR_interpretation"] = certainty_df["odds_ratio"].apply(lr_scale)
certainty_df.to_csv("results/certainty_analysis_Random_Forest.csv", index=False)
print(certainty_df.head(10).to_string(index=False))

# 8. Visualización escala de certeza
certainty_sorted = certainty_df.sort_values("log_odds", ascending=False)
fig_h = max(6, len(certainty_sorted) * 0.22)
plt.figure(figsize=(12, fig_h))
colors = plt.cm.Blues(np.linspace(0.35, 0.9, len(certainty_sorted)))
plt.barh(certainty_sorted["file"], certainty_sorted["log_odds"], color=colors)
plt.axvline(0, color="gray", linestyle="--", lw=1)
plt.title("Escala de certeza — Random Forest", fontsize=13, fontweight="bold")
plt.xlabel("Log-odds  (→ robot | ← humano)")
plt.ylabel("Texto")
plt.tight_layout()
plt.savefig("results/certainty_scale.png", dpi=150, bbox_inches="tight")
plt.close()

# 9. Guardar dataset agregado final
df.to_csv("results/estadisticas_por_texto.csv", index=False)
print("\nFiguras y CSVs guardados en results/")
