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
    reference_text,
    lexicons_comparison=None,
    ngram_n=3
):
    """
    Calcula métricas lingüísticas y compara el texto con n idiomas.
    lexicons_comparison: dict { 'es': lexicon_es, 'fr': lexicon_fr, ... }
    """

    if lexicons_comparison is None:
        lexicons_comparison = {}

    # 1. Métricas básicas
    ttr_score = ttr(text)
    entropy_score = lexicalEntropy(text)
    ngram_score = ngramOverlap(text, reference_text, n=ngram_n)

    # 2. Frecuencias relativas
    freq_target = relativeLanguageFrequency(text, lexicon_target, lexicon_target)[0]

    freq_comparison = {}
    for lang, lexicon in lexicons_comparison.items():
        _, freq_other = relativeLanguageFrequency(text, lexicon_target, lexicon)
        freq_comparison[lang] = freq_other

    # 3. Normalización
    ttr_norm = normalize(ttr_score, 0.2, 0.8)
    entropy_norm = normalize(entropy_score, 2.0, 6.0)
    ngram_norm = ngram_score

    # 4. Score dinámico
    # pesos base
    score = (
        0.32 * freq_target +
        0.22 * ngram_norm +
        0.15 * ttr_norm +
        0.15 * entropy_norm
    )

    # pesos para idiomas comparados (repartidos equitativamente)
    if freq_comparison:
        peso = 0.16 / len(freq_comparison)
        for lang, freq in freq_comparison.items():
            score += peso * (1 - freq)

    prob = 1 / (1 + math.exp(-5 * (score - 0.5)))

    return {
        "ttr": ttr_score,
        "entropy": entropy_score,
        "ngram_overlap": ngram_score,
        "freq_target": freq_target,
        "freq_comparison": freq_comparison,
        "calidad": prob
    }

