# descriptivas.py — longitud (tokens) de los 90 artículos pareados por autor
import re
import pandas as pd

CORPUS_CSV  = "data/corpus_robot-talk_2026-02-04.csv"
WINDOW_SIZE = 400
SELECTED_AUTHORS = [
    "humano", "bard", "claude-3-5-sonnet-20240620", "deepseek-v3",
    "gpt-3.5-turbo", "gpt-4", "mixtral-8x7b-instruct-v0.1",
]


def tokenize(text):
    return re.findall(r'\b[a-záéíóúüñ]+\b', text.lower())


df = pd.read_csv(CORPUS_CSV)
df["author"]   = df["tipo_version"].fillna("").str.strip().str.lower()
df["pair_id"]  = df["id_ref"].fillna(df["id"]).astype(str)
df["n_tokens"] = df["text"].fillna("").map(lambda x: len(tokenize(str(x))))
df = df[
    (df["dominio"].fillna("").str.strip() == "artículos")
    & (df["ambito"].fillna("").str.strip() == "lingüística")
    & (df["author"].isin(SELECTED_AUTHORS))
    & (df["n_tokens"] >= WINDOW_SIZE)
]
common = set.intersection(*(set(df.loc[df["author"] == a, "pair_id"]) for a in SELECTED_AUTHORS))
df = df[df["pair_id"].isin(common)]

out = (df.groupby("author")["n_tokens"]
         .agg(n="count", min_tokens="min", median_tokens="median",
              mean_tokens="mean", max_tokens="max")
         .round(1).reset_index())
out.to_csv("results/descriptivas_articulos_90x7.csv", index=False)
print(out.to_string(index=False))
print("\nGuardado: results/descriptivas_articulos_90x7.csv")
