# 05_ablacion.py
# Estudio de ablación: rasgos estadísticos independientes del contenido y la lengua
# (leyes de Zipf, Heaps, brevedad y métricas derivadas).
# Evaluación con validación cruzada agrupada por artículo base.
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings("ignore")

RUTA_STAT    = "results/estadisticas_textos.csv"
RANDOM_STATE = 42
N_FOLDS      = 5

ALL_STAT_COLS = [
    "zipf_slope", "zipf_r2", "zipf_top_cov",
    "heaps_beta", "heaps_k", "heaps_r2",
    "brevity_slope", "brevity_corr",
    "mean_word_length", "ttr",
    "hapax_rate",
    "long_word_ratio", "short_word_ratio",
    "mean_sent_len", "std_sent_len",
]

# Grupos para el ablation (organizados por ley de referencia)
GRUPOS_STAT = {
    # Ley de Zipf: parámetros clásicos + derivada
    "Zipf — parámetros":           ["zipf_slope", "zipf_r2"],
    "Zipf — cobertura":            ["zipf_top_cov"],
    "Zipf (todos)":                ["zipf_slope", "zipf_r2", "zipf_top_cov"],
    # Ley de Heaps: parámetros clásicos + derivadas
    "Heaps — parámetros":          ["heaps_beta", "heaps_k", "heaps_r2"],
    "Heaps — riqueza léxica":      ["hapax_rate", "ttr"],
    "Heaps (todos)":               ["heaps_beta", "heaps_k", "heaps_r2",
                                    "hapax_rate", "ttr"],
    # Principio de brevedad: parámetros clásicos + derivadas léxicas y sintácticas
    "Brevedad — parámetros":       ["brevity_slope", "brevity_corr"],
    "Brevedad — léxica":           ["mean_word_length", "long_word_ratio", "short_word_ratio"],
    "Brevedad — sintáctica":       ["mean_sent_len", "std_sent_len"],
    "Brevedad (todos)":            ["brevity_slope", "brevity_corr",
                                    "mean_word_length", "long_word_ratio", "short_word_ratio",
                                    "mean_sent_len", "std_sent_len"],
    # Conjunto completo
    "Estadísticos (todos)":        ALL_STAT_COLS,
}

MODEL_NAMES = ["LR", "SVM", "RF"]


# Carga de datos
df_raw = pd.read_csv(RUTA_STAT)
df = (
    df_raw.groupby(["file", "pair_id", "genre", "corpus"])[ALL_STAT_COLS]
    .median()
    .reset_index()
    .sort_values("file")
    .reset_index(drop=True)
)

encoder = LabelEncoder()
y       = encoder.fit_transform(df["corpus"].values)
X_stat  = df[ALL_STAT_COLS].values
groups  = df["pair_id"].values

print(f"Textos: {len(df)}  |  TH={int((y==0).sum())}  |  TR={int((y==1).sum())}")
print(f"Validación: GroupKFold(k={N_FOLDS}) por pair_id ({df['pair_id'].nunique()} artículos base)")


# Modelos
def make_models():
    return {
        "LR":  LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced",
                   random_state=RANDOM_STATE),
        "RF":  RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                      random_state=RANDOM_STATE),
    }


# Validación cruzada
def run_stat_cv(stat_cols, model_name):
    col_idx = [ALL_STAT_COLS.index(c) for c in stat_cols]
    gkf = GroupKFold(n_splits=N_FOLDS)
    f1s, aucs, fprs = [], [], []
    for tr, te in gkf.split(X_stat, y, groups):
        clf = make_models()[model_name]
        sc  = StandardScaler()
        X_tr = sc.fit_transform(X_stat[tr][:, col_idx])
        X_te = sc.transform(X_stat[te][:, col_idx])
        clf.fit(X_tr, y[tr])
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
        f1s.append(f1_score(y[te], y_pred, average="macro"))
        aucs.append(roc_auc_score(y[te], y_prob))
        mask_h = y[te] == 0  # humano=0 (orden alfabético LabelEncoder)
        if mask_h.sum() > 0:
            fprs.append(float((y_pred[mask_h] == 1).sum()) / mask_h.sum())
    return np.mean(f1s), np.std(f1s), np.mean(aucs), np.std(aucs), np.mean(fprs), np.std(fprs)


# Ejecutar ablación
results = []

print("\nAblación por grupo de rasgos")
for grupo, cols in GRUPOS_STAT.items():
    for mname in MODEL_NAMES:
        f1m, f1s, aucm, aucs, fprm, fprs_std = run_stat_cv(cols, mname)
        results.append({"Condición": grupo, "Modelo": mname, "n_rasgos": len(cols),
                        "F1 macro": f1m, "F1 std": f1s, "AUC": aucm, "AUC std": aucs,
                        "FPR": fprm, "FPR std": fprs_std})
        print(f"  {grupo:35s} | {mname:3s} | F1={f1m:.3f}±{f1s:.3f}  AUC={aucm:.3f}±{aucs:.3f}  FPR={fprm:.3f}±{fprs_std:.3f}")


# Tabla resumen
df_res = pd.DataFrame(results)
df_res.to_csv("results/ablacion.csv", index=False)

print("\n" + "="*72)
print("  ABLACIÓN — macro-F1 y AUC  (GroupKFold por pair_id, k=5)")
print("="*72)

for metric in ["F1 macro", "AUC", "FPR"]:
    pivot = df_res.pivot(index="Condición", columns="Modelo", values=metric)
    pivot = pivot.reindex(df_res["Condición"].unique())
    print(f"\n{metric}:")
    print(pivot.round(3).to_string())

print("\nGuardado: results/ablacion.csv")
