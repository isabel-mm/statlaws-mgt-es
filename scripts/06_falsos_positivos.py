# 06_falsos_positivos.py
# FPR (humano→robótico) con GroupKFold(k=5), mismo protocolo que 05_ablacion.py.
# Añade FPR a los grupos clave y al conjunto completo de rasgos.
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

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

GRUPOS = {
    "Brevedad — sintáctica":  ["mean_sent_len", "std_sent_len"],
    "Brevedad (todos)":       ["brevity_slope", "brevity_corr",
                               "mean_word_length", "long_word_ratio", "short_word_ratio",
                               "mean_sent_len", "std_sent_len"],
    "Estadísticos (todos)":   ALL_STAT_COLS,
}

df_raw = pd.read_csv(RUTA_STAT)
df = (
    df_raw.groupby(["file", "pair_id", "genre", "corpus"])[ALL_STAT_COLS]
    .median()
    .reset_index()
    .sort_values("file")
    .reset_index(drop=True)
)

encoder = LabelEncoder()
y       = encoder.fit_transform(df["corpus"].values)   # humano=0, robot=1
X       = df[ALL_STAT_COLS].values
groups  = df["pair_id"].values

# clase 0 = humano (LabelEncoder en orden alfabético: humano < robot)
idx_humano = np.where(encoder.classes_ == "humano")[0][0]


def make_models():
    return {
        "LR":  LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced",
                   random_state=RANDOM_STATE),
        "RF":  RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                      random_state=RANDOM_STATE),
    }


def run_cv_fpr(cols, model_name):
    col_idx = [ALL_STAT_COLS.index(c) for c in cols]
    gkf = GroupKFold(n_splits=N_FOLDS)
    f1s, aucs, fprs = [], [], []
    for tr, te in gkf.split(X, y, groups):
        clf = make_models()[model_name]
        sc  = StandardScaler()
        X_tr = sc.fit_transform(X[tr][:, col_idx])
        X_te = sc.transform(X[te][:, col_idx])
        clf.fit(X_tr, y[tr])
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
        f1s.append(f1_score(y[te], y_pred, average="macro"))
        aucs.append(roc_auc_score(y[te], y_prob))
        # FPR: humano(0) clasificado como robot(1)
        mask_hum = y[te] == idx_humano
        if mask_hum.sum() > 0:
            fprs.append((y_pred[mask_hum] != idx_humano).mean())
    return np.mean(f1s), np.std(f1s), np.mean(aucs), np.std(aucs), np.mean(fprs), np.std(fprs)


print(f"Textos: {len(df)}  |  TH={int((y==0).sum())}  |  TR={int((y==1).sum())}")
print(f"\nFPR humano→robótico — GroupKFold(k={N_FOLDS})")
print(f"{'Grupo':30s} {'Modelo':4s}  {'F1':>7}  {'AUC':>7}  {'FPR':>7}")

results = []
for grupo, cols in GRUPOS.items():
    for mname in ("LR", "SVM", "RF"):
        f1m, f1s, aucm, aucs, fprm, fprs = run_cv_fpr(cols, mname)
        results.append({
            "Condición": grupo, "Modelo": mname, "n_rasgos": len(cols),
            "F1 macro": f1m, "F1 std": f1s,
            "AUC": aucm, "AUC std": aucs,
            "FPR": fprm, "FPR std": fprs,
        })
        print(f"  {grupo:30s} {mname:4s}  "
              f"F1={f1m:.3f}±{f1s:.3f}  "
              f"AUC={aucm:.3f}±{aucs:.3f}  "
              f"FPR={fprm:.3f}±{fprs:.3f}")

pd.DataFrame(results).to_csv("results/fpr_ablacion.csv", index=False)
print("\nGuardado: results/fpr_ablacion.csv")
