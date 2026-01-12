import re
from collections import Counter
import math

def loadLexicon(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        return set(f.read().splitlines())

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def ttr(text):
    tokens = tokenize(text)
    if not tokens:
        return 0
    return len(set(tokens)) / len(tokens)

def lexicalEntropy(text):
    tokens = tokenize(text)
    if not tokens:
        return 0.0

    freqs = Counter(tokens)
    total = len(tokens)

    # precalcular 1/total para evitar divisiones repetidas
    inv_total = 1 / total

    entropy = 0.0
    for count in freqs.values():
        p = count * inv_total
        entropy -= p * math.log2(p)

    return entropy

def relativeLanguageFrequency(text, lexicon_target, lexicon_spanish):
    tokens = tokenize(text)

    target_count = sum(1 for t in tokens if t in lexicon_target)
    spanish_count = sum(1 for t in tokens if t in lexicon_spanish)

    total = len(tokens)
    if total == 0:
        return 0, 0

    return target_count / total, spanish_count / total

def ngrams(tokens, n):
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def ngramOverlap(text, reference_text, n=3):
    tokens_gen = tokenize(text)
    tokens_ref = tokenize(reference_text)

    ngrams_gen = ngrams(tokens_gen, n)
    ngrams_ref = ngrams(tokens_ref, n)

    if not ngrams_gen:
        return 0

    overlap = ngrams_gen.intersection(ngrams_ref)
    return len(overlap) / len(ngrams_gen) 

def normalize(x, min_val, max_val):
    if max_val == min_val:
        return 0.0
    return (x - min_val) / (max_val - min_val)


def calidadLengua(
    text,
    lexicon_target,
    lexicon_spanish,
    lexicon_french,
    reference_text,
    ngram_n=3
):
    """
    Devuelve:
    - métricas lingüísticas
    - probabilidad de pertenencia al idioma objetivo
    - diagnóstico textual
    Incluye comparación con español y francés.
    """

    # 1. Métricas
    ttr_score = ttr(text)
    entropy_score = lexicalEntropy(text)
    ngram_score = ngramOverlap(text, reference_text, n=ngram_n)

    # Frecuencias relativa con español
    freq_target, freq_spanish = relativeLanguageFrequency(
        text, lexicon_target, lexicon_spanish
    )
    # Frecuencia relativa con francés
    freq_target, freq_french = relativeLanguageFrequency(
        text, lexicon_target, lexicon_french
    )

    # 2. Normalización 
    ttr_norm = normalize(ttr_score, 0.2, 0.8)
    entropy_norm = normalize(entropy_score, 2.0, 6.0)
    ngram_norm = ngram_score

    freq_target_norm = freq_target
    freq_spanish_norm = 1 - freq_spanish   
    freq_french_norm = 1 - freq_french     

    # 3. Score combinado
    score = (
        0.32 * freq_target_norm +
        0.22 * ngram_norm +
        0.15 * ttr_norm +
        0.15 * entropy_norm +
        0.08 * freq_spanish_norm +
        0.08 * freq_french_norm
    )
    prob = 1 / (1 + math.exp(-5 * (score - 0.5)))

    return {
        "ttr": ttr_score,
        "entropy": entropy_score,
        "freq_target": freq_target,
        "freq_spanish": freq_spanish,
        "freq_french": freq_french,
        "ngram_overlap": ngram_score,
        "calidad": prob
    }
