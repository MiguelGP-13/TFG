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

def relativeLanguageFrequency(text, lexicon):
    tokens = tokenize(text)

    target_count = sum(1 for t in tokens if t in lexicon)

    total = len(tokens)
    if total == 0:
        return 0, 0

    return target_count / total

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


def buildStoryPrompt(lang):
    prompts = {
        "es": (
            "Eres un modelo que solo puede hablar en español. "
            "Genera una historia original, coherente, completa y corta en español. "
            "No utilices ningún otro idioma.\nHistoria:"
        ),
        "ast": (
            "Tu yes un modelu que namás pue falar n'asturianu. "
            "Xenera una hestoria orixinal, coherente, completa y curtia n'asturianu. "
            "Nun uses nengún otru idioma.\nHestoria:"
        ),
        "gl": (
            "Es un modelo que só pode falar en galego. "
            "Xera unha historia orixinal, coherente, completa e curta en galego. "
            "Non empregues ningún outro idioma.\nHistoria:"
        ),
        "aran": (
            "Es un modèl que pòt parlar sonque en aranés. "
            "Genèra ua istòria originau, coerenta, completa e braca en aranés. "
            "Non emplegues cap d’auti idiòmas.\nIstòria:"
        ),
        "fr": (
            "Tu es un modèle qui ne peut parler qu’en français. "
            "Génère une histoire originale, cohérente, complète et courte en français. "
            "N’utilise aucune autre langue.\nHistoire:"
        ),
    }
    return prompts[lang]



def generar_texto(model, tokenizer, prompt: str, device, max_new_tokens=200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True
    )

    # Cortar exactamente los tokens del prompt
    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_len:]

    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def calidadLengua(
    model, tokenizer,
    lexicon_target,
    reference_text,
    source_lang,
    lexicons_comparison:dict=None,
    ngram_n=3,
    max_new_tokens=1000,
    device="cuda"
):
    """
    Calcula métricas lingüísticas y compara el texto con n idiomas.
    lexicons_comparison: dict { 'es': lexicon_es, 'fr': lexicon_fr, ... }
    """

    if lexicons_comparison is None:
        lexicons_comparison = {}

    # 0. Generar historia en la lengua
    text = generar_texto(model, tokenizer, buildStoryPrompt(source_lang), device,max_new_tokens)

    # 1. Métricas básicas
    ttr_score = ttr(text)
    entropy_score = lexicalEntropy(text)
    ngram_score = ngramOverlap(text, reference_text, n=ngram_n)

    # 2. Frecuencias relativas
    freq_target = relativeLanguageFrequency(text, lexicon_target)

    freq_comparison = {}
    for lang, lexicon in lexicons_comparison.items():
        freq_other = relativeLanguageFrequency(text, lexicon)
        freq_comparison[lang] = freq_other

    # 3. Normalización
    ttr_norm = normalize(ttr_score, 0.2, 0.8)
    entropy_norm = normalize(entropy_score, 2.0, 6.0)
    ngram_norm = ngram_score

    # 4. Score dinámico
    # pesos base
    score = (0.40 * freq_target + 0.30 * ngram_norm ) -  ((1 - ttr_norm)**2 + (1 - entropy_norm)**2) * 0.5 
    # Si ambos ttr y entropy son muy bajos, es que genera mal en el sentido de no generar, no de idioma. Por eso se eleva al cuadrado, para disminuir su importancia si va bien

    # pesos para idiomas comparados (repartidos equitativamente)
    if freq_comparison:
        peso = 0.30 / len(freq_comparison)
        for lang, freq in freq_comparison.items():
            score += peso * (1 - freq)


        

    prob = 1 / (1 + math.exp(-5 * (score - 0.5)))

    return {
        "text":text,
        "ttr": ttr_score,
        "entropy": entropy_score,
        "ngram_overlap": ngram_score,
        "freq_target": freq_target,
        "freq_comparison": freq_comparison,
        "calidad": prob
    }

