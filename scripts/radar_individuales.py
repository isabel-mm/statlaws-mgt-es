# radar_individuales.py — 6 radares individuales (humano vs. modelo) en cuadrícula 3×2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTDIR = "figures"
CSV    = "results/estadisticas_textos.csv"
C_HUM  = "#2E5F9E"   # azul — humano
C_ROB  = "#B5541E"   # naranja — robot (todos los modelos)

RADAR_GROUPS = {
    "Zipf":                  ["zipf_slope", "zipf_r2", "zipf_top_cov"],
    "Heaps":                 ["heaps_beta", "heaps_k", "heaps_r2"],
    "Brevedad\nparáms.":    ["brevity_slope", "brevity_corr"],
    "Brevedad\nléxica":     ["mean_word_length", "long_word_ratio", "short_word_ratio"],
    "Brevedad\nsintáctica": ["mean_sent_len", "std_sent_len"],
}

MODEL_CFG = [
    ("claude-3-5-sonnet-20240620", "Claude 3.5 Sonnet"),
    ("deepseek-v3",                "DeepSeek-V3"),
    ("gpt-4",                      "GPT-4"),
    ("gpt-3.5-turbo",              "GPT-3.5-turbo"),
    ("bard",                       "Bard"),
    ("mixtral-8x7b-instruct-v0.1", "Mixtral-8x7B"),
]

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      8,
    "axes.titlesize": 9,
    "figure.dpi":     150,
})

df_raw = pd.read_csv(CSV)

hum_r   = df_raw[df_raw["author"] == "humano"]
hum_med = {g: hum_r[cols].median() for g, cols in RADAR_GROUPS.items()}

def group_distance(author_df, g, cols):
    m_a = author_df[cols].median()
    m_h = hum_med[g]
    return ((m_a - m_h).abs() / m_h.abs()).mean()

raw_dist = {}
for author, label in MODEL_CFG:
    sub = df_raw[df_raw["author"] == author]
    raw_dist[author] = [group_distance(sub, g, cols)
                        for g, cols in RADAR_GROUPS.items()]

n_axes = len(RADAR_GROUPS)
max_per_axis = [max(raw_dist[a][i] for a, _ in MODEL_CFG) for i in range(n_axes)]
sim = {a: [1 - raw_dist[a][i] / max_per_axis[i] for i in range(n_axes)]
       for a, _ in MODEL_CFG}

labels_radar = list(RADAR_GROUPS.keys())
angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(2, 3, figsize=(11, 7.5),
                         subplot_kw=dict(polar=True))
axes = axes.flatten()

for ax, (author, label) in zip(axes, MODEL_CFG):
    for r, ls_r in [(0.25, ":"), (0.5, "--"), (0.75, ":"), (1.0, "-")]:
        ax.plot(angles, [r] * (n_axes + 1),
                color="lightgray", lw=0.5, ls=ls_r, zorder=0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0,5", "", "1,0"], fontsize=5.5, color="gray")
    ax.set_ylim(0, 1.08)

    # Referencia humana
    ax.fill(angles, [1.0] * (n_axes + 1), alpha=0.10, color=C_HUM, zorder=1)
    ax.plot(angles, [1.0] * (n_axes + 1),
            color=C_HUM, lw=1.4, zorder=2)

    # Perfil del modelo (siempre naranja)
    vals = sim[author] + sim[author][:1]
    ax.plot(angles, vals, lw=1.8, color=C_ROB, zorder=3)
    ax.fill(angles, vals, alpha=0.18, color=C_ROB, zorder=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_radar, fontsize=6.5, fontweight="bold")
    ax.tick_params(axis="x", pad=10)
    ax.spines["polar"].set_visible(False)
    ax.set_title(label, fontsize=9, fontweight="bold", pad=14)

fig.legend(
    handles=[
        mpatches.Patch(color=C_HUM, alpha=0.55, label="Humano (referencia)"),
        mpatches.Patch(color=C_ROB, alpha=0.55, label="Modelo"),
    ],
    loc="lower center", bbox_to_anchor=(0.5, -0.02),
    ncol=2, frameon=False, fontsize=8,
)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{OUTDIR}/radar_individuales.{ext}", bbox_inches="tight")
plt.close()
print("Guardado: radar_individuales guardado en figures/")
