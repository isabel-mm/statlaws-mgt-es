# 03_analisis_estadistico.py
# Comparación global y pareada (TH vs TR) mediante pruebas no paramétricas y tamaño del efecto.
# Requiere haber ejecutado 02_extraccion_rasgos.py primero.
import pandas as pd
import numpy as np
from scipy import stats

METRICS = [
    "zipf_slope", "zipf_r2",
    "heaps_beta", "heaps_r2",
    "brevity_slope", "brevity_corr",
]
METRIC_LABELS = {
    "zipf_slope":    "Zipf α (pendiente)",
    "zipf_r2":       "Zipf R²",
    "heaps_beta":    "Heaps β (exponente)",
    "heaps_r2":      "Heaps R²",
    "brevity_slope": "Brevedad (pendiente)",
    "brevity_corr":  "Brevedad (correlación r)",
}

# Comparación global (Cuadro de significación): las 15 variables estilométricas.
# La comparación pareada usa solo los parámetros de las leyes (METRICS).
METRICS_GLOBAL = [
    "zipf_slope", "zipf_r2", "zipf_top_cov",
    "heaps_beta", "heaps_k", "heaps_r2",
    "brevity_slope", "brevity_corr",
    "mean_word_length", "ttr", "hapax_rate",
    "long_word_ratio", "short_word_ratio",
    "mean_sent_len", "std_sent_len",
]
METRIC_LABELS.update({
    "zipf_top_cov":     "Zipf cobertura top-10",
    "heaps_k":          "Heaps k",
    "mean_word_length": "Longitud media de palabra",
    "ttr":              "TTR",
    "hapax_rate":       "Tasa de hápax",
    "long_word_ratio":  "Ratio palabras largas",
    "short_word_ratio": "Ratio palabras cortas",
    "mean_sent_len":    "Longitud media de oración",
    "std_sent_len":     "Desviación estándar de oración",
})


# Tamaño del efecto
def cohens_d_independent(a, b):
    """Cohen's d para muestras independientes (denominador: std agrupada)."""
    pooled = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def cohens_d_paired(a, b):
    """Cohen's d para pares: media(diff) / std(diff)."""
    diff = np.asarray(a) - np.asarray(b)
    return diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0


def effect_label(d):
    d = abs(d)
    if d < 0.2:   return "negligible"
    if d < 0.5:   return "small"
    if d < 0.8:   return "medium"
    return "large"


# Pruebas
def global_tests(df_agg):
    """Mann-Whitney U entre todos los TH y todos los TR."""
    hum = df_agg[df_agg["corpus"] == "humano"]
    rob = df_agg[df_agg["corpus"] == "robot"]
    rows = []
    for m in METRICS_GLOBAL:
        h, r = hum[m].dropna().values, rob[m].dropna().values
        U, p = stats.mannwhitneyu(h, r, alternative="two-sided")
        d = cohens_d_independent(h, r)
        rows.append({
            "Métrica":           METRIC_LABELS[m],
            "Mediana TH":        np.median(h),
            "Mediana TR":        np.median(r),
            "U":                 U,
            "p":                 p,
            "p (BH)":            None,      # se rellena después
            "Cohen's d":         d,
            "Tamaño efecto":     effect_label(d),
        })
    df_res = pd.DataFrame(rows)
    # Corrección de Benjamini-Hochberg
    df_res["p (BH)"] = _bh_correction(df_res["p"].values)
    return df_res


def paired_tests(df_agg):
    """Wilcoxon signed-rank sobre pares TH_i vs TR_i."""
    pivot = df_agg.pivot_table(
        index="pair_id", columns="corpus", values=METRICS, aggfunc="median"
    )
    rows = []
    for m in METRICS:
        try:
            a = pivot[(m, "humano")]
            b = pivot[(m, "robot")]
            valid = a.notna() & b.notna()
            av, bv = a[valid].values, b[valid].values
            W, p = stats.wilcoxon(av, bv, alternative="two-sided")
            d = cohens_d_paired(av, bv)
            rows.append({
                "Métrica":           METRIC_LABELS[m],
                "N pares":           int(valid.sum()),
                "Mediana diff TH−TR": np.median(av - bv),
                "W":                 W,
                "p":                 p,
                "p (BH)":            None,
                "Cohen's d":         d,
                "Tamaño efecto":     effect_label(d),
            })
        except Exception as e:
            rows.append({"Métrica": METRIC_LABELS[m], "error": str(e)})
    df_res = pd.DataFrame(rows)
    if "p" in df_res.columns:
        df_res["p (BH)"] = _bh_correction(df_res["p"].values)
    return df_res


def _bh_correction(pvals):
    """Corrección de Benjamini-Hochberg para comparaciones múltiples."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / np.arange(1, n + 1)
    # Monotonía (en orden de p creciente): cada p ajustado no puede superar el siguiente
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    corrected = np.empty(n)
    corrected[order] = np.clip(ranked, 0, 1)
    return corrected


# Formato de salida
def print_table(title, df):
    sep = "─" * 72
    print(f"\n{sep}\n  {title}\n{sep}")
    float_cols = df.select_dtypes("float").columns
    fmt = {c: "{:.4f}".format for c in float_cols}
    print(df.to_string(index=False, formatters=fmt))


def run_analysis(df_agg, label="GLOBAL"):
    print(f"\n{'='*72}")
    print(f"  ANÁLISIS {label}")
    n = df_agg.groupby("corpus").size().to_dict()
    print(f"  N textos: TH={n.get('humano',0)}, TR={n.get('robot',0)}")
    print(f"{'='*72}")

    res_global = global_tests(df_agg)
    print_table("Comparación global — Mann-Whitney U (independiente)", res_global)

    res_paired = paired_tests(df_agg)
    print_table("Comparación pareada — Wilcoxon signed-rank (TH_i vs TR_i)", res_paired)

    return res_global, res_paired


# Main
if __name__ == "__main__":
    df = pd.read_csv("results/estadisticas_textos.csv")

    # Agregar iteraciones bootstrap → mediana por texto
    df_agg = (
        df.groupby(["file", "pair_id", "genre", "corpus"])[METRICS_GLOBAL]
        .median()
        .reset_index()
    )
    df_agg.to_csv("results/estadisticas_por_texto.csv", index=False)
    print(f"Estadísticas por texto guardadas ({len(df_agg)} textos).")

    # Análisis global
    res_global, res_paired = run_analysis(df_agg, "GLOBAL — todos los géneros")
    res_global.to_csv("results/analisis_global.csv", index=False)
    res_paired.to_csv("results/analisis_pareado_global.csv", index=False)

    print("\nResultados guardados en results/")
