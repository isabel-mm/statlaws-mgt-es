# =====================================
# functions.py
# Leyes estadísticas del lenguaje (simplificado)
# =====================================

import numpy as np
from collections import Counter
from scipy.stats import linregress

# ================================
# 1. Ley de Zipf
# ================================
def zipf_slope(tokens):
    """
    Estima la pendiente (alpha) de la Ley de Zipf.
    log(freq) ~ alpha * log(rango)

    Parámetros:
    - tokens: lista de palabras.

    Devuelve:
    - slope: pendiente (Zipf alpha ≈ slope)
    - r2: coeficiente de determinación
    """
    freqs = Counter(tokens)
    sorted_freqs = sorted(freqs.values(), reverse=True)
    ranks = np.arange(1, len(sorted_freqs) + 1)

    log_ranks = np.log(ranks)
    log_freqs = np.log(sorted_freqs)

    slope, _, r_value, _, _ = linregress(log_ranks, log_freqs)
    return slope, r_value**2


# ================================
# 2. Ley de Heaps
# ================================
def heaps_beta(tokens):
    """
    Estima el exponente beta de la Ley de Heaps.
    log(V) = log(k) + beta * log(N)

    Parámetros:
    - tokens: lista de palabras.

    Devuelve:
    - beta: exponente de Heaps
    - k: constante
    - r2: coeficiente de determinación
    """
    V, N = [], []
    seen = set()
    for i, token in enumerate(tokens, 1):
        seen.add(token)
        N.append(i)
        V.append(len(seen))

    logN, logV = np.log(N), np.log(V)
    slope, intercept, r_value, _, _ = linregress(logN, logV)
    beta = slope
    k = np.exp(intercept)
    return beta, k, r_value**2


# ================================
# 3. Métricas léxicas básicas
# ================================
def lexical_stats(tokens):
    """
    Métricas directas del principio de brevedad y la ley de Heaps.

    Parámetros:
    - tokens: lista de palabras.

    Devuelve:
    - mean_word_length: longitud media de palabra (en caracteres)
    - ttr: Type-Token Ratio = tipos únicos / total tokens
    """
    if not tokens:
        raise ValueError("Lista de tokens vacía.")
    lengths = np.array([len(t) for t in tokens])
    ttr = len(set(tokens)) / len(tokens)
    return lengths.mean(), ttr


# ================================
# 4. Principio de brevedad de Zipf
# ================================
def brevity_slope(tokens):
    """
    Estima la pendiente de la relación entre longitud de palabra y frecuencia.
    log(freq) ~ slope * log(longitud)

    Parámetros:
    - tokens: lista de palabras.

    Devuelve:
    - slope: pendiente (más negativa = mayor brevedad)
    - r: correlación de Pearson (nota: devuelve r, no r², a diferencia de zipf/heaps)

    Lanza ValueError si hay menos de 2 longitudes distintas.
    """
    freqs = Counter(tokens)
    word_lengths = np.array([len(word) for word in freqs.keys()])
    frequencies = np.array(list(freqs.values()))

    if len(np.unique(word_lengths)) < 2:
        raise ValueError("Varianza nula en longitudes de palabra; no se puede ajustar la recta.")

    log_len = np.log(word_lengths)
    log_freq = np.log(frequencies)

    slope, _, r_value, _, _ = linregress(log_len, log_freq)
    return slope, r_value


# ================================
# 5. Cobertura de Zipf
# ================================
def zipf_coverage(tokens, k=10):
    """
    Proporción de tokens cubiertos por los k tipos más frecuentes.
    Operacionaliza la concentración zipfiana sin regresión:
    valores altos → distribución desigual (α más empinado).
    """
    freqs = Counter(tokens)
    N = len(tokens)
    top_k = sorted(freqs.values(), reverse=True)[:k]
    return sum(top_k) / N


# ================================
# 6. Riqueza léxica avanzada
# ================================
def lexical_diversity(tokens):
    """
    Devuelve:
    - hapax_rate: proporción de palabras que aparecen exactamente una vez
    - herdan_c: log(V) / log(N)  (independiente de la longitud del texto)
    - yule_k: 10^4 * M2 / (N*(N-1))  (constante de Yule)
    """
    freqs = Counter(tokens)
    N, V = len(tokens), len(freqs)
    hapax = sum(1 for f in freqs.values() if f == 1)
    hapax_rate = hapax / N
    herdan_c = np.log(V) / np.log(N) if N > 1 and V > 1 else 0.0
    M2 = sum(f * (f - 1) for f in freqs.values())
    yule_k = 1e4 * M2 / (N * (N - 1)) if N > 1 else 0.0
    return hapax_rate, herdan_c, yule_k


# ================================
# 6. Entropía de frecuencias
# ================================
def freq_entropy(tokens):
    """
    Entropía de Shannon de la distribución de frecuencias de palabras.
    Mayor entropía = distribución más uniforme (menos predictible).
    """
    freqs = Counter(tokens)
    N = len(tokens)
    probs = np.array([f / N for f in freqs.values()])
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


# ================================
# 7. Ratios de longitud de palabra
# ================================
def word_length_ratios(tokens):
    """
    Devuelve:
    - long_word_ratio:  proporción de palabras con más de 6 caracteres
    - short_word_ratio: proporción de palabras con 3 o menos caracteres
    """
    N = len(tokens)
    long_ratio  = sum(1 for t in tokens if len(t) > 6) / N
    short_ratio = sum(1 for t in tokens if len(t) <= 3) / N
    return long_ratio, short_ratio
# ================================
