# 10_figuras_resultados.py — genera todas las figuras para el paper
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve, auc as auc_score
import warnings
warnings.filterwarnings("ignore")

OUTDIR = "figures"
CSV    = "results/estadisticas_textos.csv"
SEED   = 42

# Paleta académica azul/terracota
C_HUM  = "#2E5F9E"   # azul: humano
C_ROB  = "#B5541E"   # terracota: robot

# Colores por ley (para ablación) — gradiente de azul claro a azul oscuro
C_ZIPF  = "#B8CCE4"  # azul muy claro
C_HEAPS = "#6699CC"  # azul medio
C_BREV  = "#2E5F9E"  # azul acero
C_ALL   = "#1A3A5C"  # azul marino

# Clasificadores (color + estilo de línea, para distinguirlos sin depender del color)
CLF_STYLES = {
    "Logistic Regression": ":",
    "SVM (RBF)":           "--",
    "Random Forest":       "-",
}
CLF_COLORS = {
    "Logistic Regression": "#6699CC",
    "SVM (RBF)":           "#2E5F9E",
    "Random Forest":       "#1A3A5C",
}

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi":     150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

ALL_METRICS = [
    "zipf_slope", "zipf_r2", "zipf_top_cov",
    "heaps_beta", "heaps_k", "heaps_r2",
    "brevity_slope", "brevity_corr",
    "mean_word_length", "ttr",
    "hapax_rate",
    "long_word_ratio", "short_word_ratio",
    "mean_sent_len", "std_sent_len",
]

# Cargar y agregar
df_raw = pd.read_csv(CSV)
df = (
    df_raw.groupby(["file", "pair_id", "genre", "corpus"])[ALL_METRICS]
    .median()
    .reset_index()
)
humano = df[df["corpus"] == "humano"]
robot  = df[df["corpus"] == "robot"]

enc = LabelEncoder()
y   = enc.fit_transform(df["corpus"].values)
X   = df[ALL_METRICS].values
groups = df["pair_id"].values


# FIG 1 — Boxplots de las métricas más discriminativas
KEY_METRICS = [
    ("brevity_slope",    "Pendiente de brevedad"),
    ("brevity_corr",     "Correlación de brevedad"),
    ("mean_word_length", "Long. media de palabra"),
    ("ttr",              "Type-Token Ratio"),
    ("mean_sent_len",    "Long. media de oración"),
    ("std_sent_len",     "Desv. est. long. oración"),
]

fig, axes = plt.subplots(2, 3, figsize=(10, 6))
axes = axes.flatten()

for ax, (col, label) in zip(axes, KEY_METRICS):
    bp = ax.boxplot(
        [humano[col].values, robot[col].values],
        patch_artist=True, widths=0.5,
        medianprops=dict(color="white", linewidth=2),
        flierprops=dict(marker="o", markersize=2.5, alpha=0.4),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    for box, col_c in zip(bp["boxes"], [C_HUM, C_ROB]):
        box.set_facecolor(col_c + "BB")
        box.set_edgecolor(col_c)
    for i, w in enumerate(bp["whiskers"]):
        w.set_color([C_HUM, C_HUM, C_ROB, C_ROB][i])
    for i, c in enumerate(bp["caps"]):
        c.set_color([C_HUM, C_HUM, C_ROB, C_ROB][i])
    ax.set_title(label, fontsize=9)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["TH", "TR"])


fig.legend(
    handles=[
        mpatches.Patch(color=C_HUM + "BB", label="TH", edgecolor=C_HUM),
        mpatches.Patch(color=C_ROB + "BB", label="TR", edgecolor=C_ROB),
    ],
    loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03), frameon=False,
)
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{OUTDIR}/boxplots_metricas.{ext}", bbox_inches="tight")
plt.close()
print("Guardado: boxplots_metricas")


# FIG 2 — Ablación: gráfico de barras diferencial (ΔF1 respecto al modelo completo)
df_abl = pd.read_csv("results/ablacion.csv")
best = df_abl.groupby("Condición")["F1 macro"].max().reset_index()

ORDER = [
    "Zipf — parámetros",
    "Zipf — cobertura",
    "Zipf (todos)",
    "Heaps — parámetros",
    "Heaps — riqueza léxica",
    "Heaps (todos)",
    "Brevedad — parámetros",
    "Brevedad — léxica",
    "Brevedad — sintáctica",
    "Brevedad (todos)",
    "Estadísticos (todos)",
]
best = best.set_index("Condición").reindex(ORDER).reset_index()

F1_FULL = best.loc[best["Condición"] == "Estadísticos (todos)", "F1 macro"].values[0]
best["delta_f1"] = F1_FULL - best["F1 macro"]

def ablation_color(label):
    if label.startswith("Zipf"):     return C_ZIPF
    if label.startswith("Heaps"):    return C_HEAPS
    if label.startswith("Brevedad"): return C_BREV
    return C_ALL

colors = [ablation_color(c) for c in best["Condición"]]

fig, ax = plt.subplots(figsize=(7, 5.8))
bars = ax.barh(best["Condición"], best["delta_f1"],
               color=colors, edgecolor="white", height=0.68)

# Separadores entre leyes
for y_pos in [2.5, 5.5, 9.5]:
    ax.axhline(y_pos, color="lightgray", linewidth=0.8, linestyle="--")

# Línea de referencia: modelo completo (ΔF1 = 0)
ax.axvline(0, color="#444444", linewidth=1.1, linestyle="-", zorder=3)

ax.set_xlim(-0.01, 0.56)
ax.set_xlabel(f"$\\Delta$F1 macro (respecto al modelo completo, F1={F1_FULL:.3f})")

# Etiquetas: F1 absoluto a la derecha de cada barra
for bar, (_, row) in zip(bars, best.iterrows()):
    w = bar.get_width()
    ax.text(w + 0.006, bar.get_y() + bar.get_height() / 2,
            f"F1={row['F1 macro']:.3f}", va="center", ha="left", fontsize=7.5)

ax.legend(
    handles=[
        mpatches.Patch(facecolor=C_ZIPF,  label="Ley de Zipf"),
        mpatches.Patch(facecolor=C_HEAPS, label="Ley de Heaps"),
        mpatches.Patch(facecolor=C_BREV,  label="Principio de brevedad"),
        mpatches.Patch(facecolor=C_ALL,   label="Conjunto completo"),
    ],
    loc="upper right", frameon=False, fontsize=8,
)
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{OUTDIR}/ablacion_f1.{ext}", bbox_inches="tight")
plt.close()
print("Guardado: ablacion_f1")


# FIG 3 — Curvas ROC (15 rasgos, validación cruzada agrupada)
gkf = GroupKFold(n_splits=5)

clfs = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced",
                                              random_state=SEED),
    "SVM (RBF)":           SVC(kernel="rbf", probability=True, class_weight="balanced",
                               random_state=SEED),
    "Random Forest":    RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                                  random_state=SEED),
}

fig, ax = plt.subplots(figsize=(5.5, 4.5))
for name, clf in clfs.items():
    tprs, aucs_cv = [], []
    mean_fpr = np.linspace(0, 1, 100)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        clf.fit(X_tr, y[tr])
        prob = clf.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y[te], prob)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        aucs_cv.append(auc_score(fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs_cv)
    std_auc  = np.std(aucs_cv)
    ax.plot(mean_fpr, mean_tpr, color=CLF_COLORS[name], linewidth=2,
            linestyle=CLF_STYLES[name],
            label=f"{name}  (AUC={mean_auc:.3f}±{std_auc:.3f})")
    ax.fill_between(mean_fpr,
                    np.mean(tprs, axis=0) - np.std(tprs, axis=0),
                    np.mean(tprs, axis=0) + np.std(tprs, axis=0),
                    alpha=0.10, color=CLF_COLORS[name])

ax.plot([0, 1], [0, 1], color="lightgray", linestyle="--",
        linewidth=0.9, label="Azar (AUC=0,5)")
ax.set_xlabel("Tasa de falsos positivos")
ax.set_ylabel("Tasa de verdaderos positivos")
ax.legend(loc="lower right", frameon=False)
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{OUTDIR}/roc_curves.{ext}", bbox_inches="tight")
plt.close()
print("Guardado: roc_curves")


# FIG 4 — Scatter: economía léxica vs. sintáctica
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.scatter(humano["mean_sent_len"], humano["mean_word_length"],
           color=C_HUM, alpha=0.55, s=24, label="TH",
           edgecolors="none")
ax.scatter(robot["mean_sent_len"], robot["mean_word_length"],
           color=C_ROB, alpha=0.55, s=24, label="TR",
           edgecolors="none")
ax.set_xlabel("Long. media de oración (tokens)  [brevedad sintáctica]")
ax.set_ylabel("Long. media de palabra (caracteres)  [brevedad léxica]")
ax.legend(frameon=False)
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{OUTDIR}/scatter_economia.{ext}", bbox_inches="tight")
plt.close()
print("Guardado: scatter_economia")


print("\nTodas las figuras guardadas en figures/")
