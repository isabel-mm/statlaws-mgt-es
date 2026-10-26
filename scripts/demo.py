# demo.py — extracción de métricas estilométricas con muestreo bootstrap
import os
import re
import numpy as np
import pandas as pd
import spacy
from functions import (zipf_slope, heaps_beta, brevity_slope, lexical_stats,
                       lexical_diversity, freq_entropy, word_length_ratios,
                       zipf_coverage)

_nlp = spacy.load("es_core_news_sm",
                  disable=["morphologizer", "ner", "attribute_ruler", "lemmatizer"])

N_BOOTSTRAP = 100   # iteraciones bootstrap por texto
WINDOW_SIZE = 400   # tokens por ventana; compatible con los 90 artículos comunes
RANDOM_SEED = 42
CORPUS_CSV = "data/corpus_robot-talk_2026-02-04.csv"
SELECTED_AUTHORS = [
    "humano",
    "bard",
    "claude-3-5-sonnet-20240620",
    "deepseek-v3",
    "gpt-3.5-turbo",
    "gpt-4",
    "mixtral-8x7b-instruct-v0.1",
]


def tokenize(text):
    """Tokenización alfabética en minúsculas, filtra números y puntuación."""
    return re.findall(r'\b[a-záéíóúüñ]+\b', text.lower())


def sentence_stats(text):
    """Media y desviación típica de palabras por oración (segmentación con spaCy)."""
    doc = _nlp(str(text))
    lengths = [len(re.findall(r'\b[a-záéíóúüñ]+\b', sent.text.lower()))
               for sent in doc.sents]
    lengths = [l for l in lengths if l > 0]
    if not lengths:
        return 0.0, 0.0
    return float(np.mean(lengths)), float(np.std(lengths))


def bootstrap_windows(tokens, window_size, n_samples, rng):
    """Muestrea n_samples ventanas contiguas de window_size tokens con reemplazo."""
    n = len(tokens)
    max_start = n - window_size
    starts = rng.integers(0, max_start + 1, size=n_samples)
    return [tokens[s : s + window_size] for s in starts]


def normalise_author(tipo_version):
    return (tipo_version or "").strip().lower()


def load_balanced_corpus(csv_path, rng):
    """Carga el CSV y selecciona los 90 artículos comunes a humano + 6 modelos."""
    df = pd.read_csv(csv_path)
    df["author"] = df["tipo_version"].map(normalise_author)
    df["genre"] = df["dominio"].fillna("desconocido").str.strip()
    df["field"] = df["ambito"].fillna("desconocido").str.strip()
    df["pair_id"] = df["id_ref"].fillna(df["id"]).astype(str)
    df["file"] = df["id"].astype(str)
    df["n_tokens"] = df["text"].fillna("").map(lambda x: len(tokenize(str(x))))

    df = df[
        (df["genre"] == "artículos")
        & (df["field"] == "lingüística")
        & (df["author"].isin(SELECTED_AUTHORS))
        & (df["n_tokens"] >= WINDOW_SIZE)
    ].copy()

    refs_by_author = {
        author: set(df.loc[df["author"] == author, "pair_id"])
        for author in SELECTED_AUTHORS
    }
    common_refs = set.intersection(*(refs_by_author[author] for author in SELECTED_AUTHORS))
    if len(common_refs) != 90:
        raise ValueError(f"Se esperaban 90 artículos comunes y hay {len(common_refs)}.")

    out = df[df["pair_id"].isin(common_refs)].copy()
    expected = len(SELECTED_AUTHORS) * len(common_refs)
    if len(out) != expected:
        raise ValueError(f"Panel incompleto: {len(out)} filas, esperadas {expected}.")

    out = out.sort_values(["author", "genre", "file"]).reset_index(drop=True)
    return out


def process_row(row, rng):
    text = str(row["text"])
    tokens = tokenize(text)
    if len(tokens) < WINDOW_SIZE:
        return []

    windows = bootstrap_windows(tokens, WINDOW_SIZE, N_BOOTSTRAP, rng)
    mean_sl, std_sl = sentence_stats(text)

    rows = []
    for i, window in enumerate(windows):
        try:
            slope_z, r2_z           = zipf_slope(window)
            top_cov                 = zipf_coverage(window)
            beta_h, k_h, r2_h       = heaps_beta(window)
            slope_b, r_b            = brevity_slope(window)
            mwl, ttr                = lexical_stats(window)
            hapax_r, herdan, yule   = lexical_diversity(window)
            entropy                 = freq_entropy(window)
            long_r, short_r         = word_length_ratios(window)
        except Exception:
            continue
        rows.append({
            "file":             row["file"],
            "pair_id":          row["pair_id"],
            "genre":            row["genre"],
            "corpus":           "humano" if row["author"] == "humano" else "robot",
            "author":           row["author"],
            "iteration":        i,
            "zipf_slope":       slope_z,
            "zipf_r2":          r2_z,
            "zipf_top_cov":     top_cov,
            "heaps_beta":       beta_h,
            "heaps_k":          k_h,
            "heaps_r2":         r2_h,
            "brevity_slope":    slope_b,
            "brevity_corr":     r_b,
            "mean_word_length": mwl,
            "ttr":              ttr,
            "hapax_rate":       hapax_r,
            "herdan_c":         herdan,
            "yule_k":           yule,
            "shannon_entropy":  entropy,
            "long_word_ratio":  long_r,
            "short_word_ratio": short_r,
            "mean_sent_len":    mean_sl,
            "std_sent_len":     std_sl,
        })
    return rows


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    all_rows, skipped = [], []
    corpus = load_balanced_corpus(CORPUS_CSV, rng)
    corpus.to_csv("results/corpus_balanceado.csv", index=False)

    for _, row in corpus.iterrows():
        rows = process_row(row, rng)
        if rows:
            all_rows.extend(rows)
        else:
            skipped.append(row["file"])

    if skipped:
        print(f"Omitidos por extensión insuficiente (<{WINDOW_SIZE} tokens): {skipped}")

    df = pd.DataFrame(all_rows)
    df.to_csv("results/estadisticas_textos.csv", index=False)

    n_textos = df["file"].nunique()
    print(f"\nGuardado: results/estadisticas_textos.csv")
    print(f"  {len(df)} filas | {n_textos} textos | {N_BOOTSTRAP} iteraciones bootstrap c/u")
    print("\nTextos por autor:")
    print(df.drop_duplicates("file")["author"].value_counts().sort_index().to_string())
    print(f"\nMedianas por corpus:")
    cols = ["zipf_slope", "heaps_beta", "brevity_slope", "brevity_corr", "mean_word_length", "ttr"]
    print(df.groupby("corpus")[cols].median().round(4))
