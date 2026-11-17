# 12_figuras_leyes.py
# Genera tres figuras de ejemplo sobre el corpus humano:
#   1. Ley de Zipf  (rango–frecuencia, log-log)
#   2. Ley de Heaps (tokens–vocabulario, log-log)
#   3. Principio de brevedad (longitud–frecuencia)
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter
from scipy import stats

# Colores del paper
HUM   = "#2E5F9E"   # azul: humano
LIGHT = "#9BB8D4"   # blue tenue para líneas individuales
GREY  = "#AAAAAA"

OUT   = "figures"
DPI   = 300

CORPUS_CSV = "data/corpus_robot-talk_2026-02-04.csv"
N_SHOW     = 12   # textos individuales a mostrar semitransparentes

def tokenize(text):
    return re.findall(r'\b[a-záéíóúüñ]+\b', str(text).lower())

# Carga corpus humano (90 textos del panel)
df = pd.read_csv(CORPUS_CSV)
df["author"]  = df["tipo_version"].str.strip().str.lower()
df["pair_id"] = df["id_ref"].fillna(df["id"]).astype(str)

SELECTED = ["humano","bard","claude-3-5-sonnet-20240620","deepseek-v3",
            "gpt-3.5-turbo","gpt-4","mixtral-8x7b-instruct-v0.1"]
df = df[(df["dominio"]=="artículos") & (df["ambito"]=="lingüística")
        & df["author"].isin(SELECTED)].copy()

refs_by = {a: set(df.loc[df["author"]==a,"pair_id"]) for a in SELECTED}
common  = set.intersection(*(refs_by[a] for a in SELECTED))
human   = df[(df["author"]=="humano") & df["pair_id"].isin(common)].copy()
texts   = [tokenize(t) for t in human["text"].fillna("") if len(tokenize(t)) >= 400]
print(f"Textos humanos cargados: {len(texts)}")

all_tokens = [t for doc in texts for t in doc]

# 1. LEY DE ZIPF
fig, ax = plt.subplots(figsize=(7, 4.5))

# Textos individuales (semitransparentes)
rng = np.random.default_rng(42)
idx_show = rng.choice(len(texts), size=min(N_SHOW, len(texts)), replace=False)
for i in idx_show:
    c   = Counter(texts[i])
    freqs = np.array(sorted(c.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs)+1)
    ax.plot(ranks, freqs, color=LIGHT, lw=0.7, alpha=0.5)

# Corpus humano agregado
c_all = Counter(all_tokens)
freqs_all = np.array(sorted(c_all.values(), reverse=True), dtype=float)
ranks_all  = np.arange(1, len(freqs_all)+1)

ax.plot(ranks_all, freqs_all, color=HUM, lw=1.6, label="TH")

# Ajuste log-log
mask = (ranks_all >= 1) & (ranks_all <= 5000)
slope, intercept, r, *_ = stats.linregress(np.log10(ranks_all[mask]),
                                             np.log10(freqs_all[mask]))
x_fit = ranks_all[mask]
y_fit = 10**intercept * x_fit**slope
ax.plot(x_fit, y_fit, "--", color=GREY, lw=1.2,
        label=f"Ajuste log-log  $\\alpha={slope:.2f}$, $R^2={r**2:.3f}$")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Rango", fontsize=11)
ax.set_ylabel("Frecuencia", fontsize=11)
ax.legend(fontsize=9, framealpha=0.9)
ax.text(0.97, 0.97,
        f"Textos individuales: $n={len(idx_show)}$ (gris claro)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color=GREY)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{int(x):,}".replace(",",".")))
plt.tight_layout()
plt.savefig(f"{OUT}/ley_zipf.pdf", dpi=DPI, bbox_inches="tight")
plt.savefig(f"{OUT}/ley_zipf.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Guardado: ley_zipf.pdf")

# 2. LEY DE HEAPS
fig, ax = plt.subplots(figsize=(7, 4.5))

# Textos individuales
for i in idx_show:
    toks = texts[i]
    vocab, ns = set(), []
    for t in toks:
        vocab.add(t)
        ns.append(len(vocab))
    ax.plot(np.arange(1, len(ns)+1), ns, color=LIGHT, lw=0.7, alpha=0.5)

# Corpus agregado
vocab_global, ns_global = set(), []
for tok in all_tokens:
    vocab_global.add(tok)
    ns_global.append(len(vocab_global))
N_arr = np.arange(1, len(ns_global)+1)
V_arr = np.array(ns_global, dtype=float)

ax.plot(N_arr, V_arr, color=HUM, lw=1.6, label="TH")

# Ajuste log-log
mask = N_arr >= 100
slope_h, intercept_h, r_h, *_ = stats.linregress(np.log10(N_arr[mask]),
                                                    np.log10(V_arr[mask]))
x_fit = N_arr[mask]
y_fit = 10**intercept_h * x_fit**slope_h
ax.plot(x_fit, y_fit, "--", color=GREY, lw=1.2,
        label=f"Ajuste log-log  $\\beta={slope_h:.2f}$, $R^2={r_h**2:.3f}$")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Número de tokens", fontsize=11)
ax.set_ylabel("Tamaño del vocabulario", fontsize=11)
ax.legend(fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(f"{OUT}/ley_heaps.pdf", dpi=DPI, bbox_inches="tight")
plt.savefig(f"{OUT}/ley_heaps.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Guardado: ley_heaps.pdf")

# 3. PRINCIPIO DE BREVEDAD
fig, ax = plt.subplots(figsize=(7, 4.5))

# Scatter: cada tipo léxico = un punto (longitud, frecuencia)
c_all = Counter(all_tokens)
words      = np.array([len(w)  for w in c_all])
log_freqs  = np.array([np.log10(f) for f in c_all.values()])

# Muestra aleatoria para no saturar el scatter (máx 4000 puntos)
rng2 = np.random.default_rng(0)
n_sample = min(4000, len(words))
idx_s = rng2.choice(len(words), size=n_sample, replace=False)
ax.scatter(words[idx_s], log_freqs[idx_s],
           color=LIGHT, s=6, alpha=0.35, linewidths=0, zorder=2)

# Mediana por longitud (sobre el corpus completo)
by_len = {}
for w, f in c_all.items():
    L = len(w)
    if 1 <= L <= 14:
        by_len.setdefault(L, []).append(np.log10(f))
lengths  = sorted(by_len.keys())
med_lf   = [np.median(by_len[L]) for L in lengths]

ax.plot(lengths, med_lf, "o-", color=HUM, lw=2, ms=7, zorder=4,
        label="Mediana por longitud")

# Ajuste lineal (log-frecuencia ~ longitud)
slope_b, intercept_b, r_b, *_ = stats.linregress(lengths, med_lf)
x_fit = np.linspace(1, 14, 100)
ax.plot(x_fit, intercept_b + slope_b * x_fit, "--", color=GREY, lw=1.4, zorder=3,
        label=f"Regresión  pend.={slope_b:.3f}, $R^2={r_b**2:.3f}$")

ax.set_xlabel("Longitud de palabra (caracteres)", fontsize=11)
ax.set_ylabel("log$_{10}$(frecuencia)", fontsize=11)
ax.set_xticks(range(1, 15))
ax.legend(fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(f"{OUT}/ley_brevedad.pdf", dpi=DPI, bbox_inches="tight")
plt.savefig(f"{OUT}/ley_brevedad.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Guardado: ley_brevedad.pdf")

print("\nTodas las figuras guardadas en figures/")
