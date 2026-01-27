from sacrebleu.metrics import BLEU, CHRF

bleu = BLEU()
chrf = CHRF()

def buildTranslationPrompt(text, target_lang, source_lang):
    """
    Devuelve un prompt en el idioma adecuado según source_lang.
    """
    prompts = {
    "es": (
        f"Traduce al {target_lang} el siguiente texto:\n\n{text}\n\nResponde únicamente con la frase traducida, sin añadir nada más.\nTraducción:"
    ),

    "fr": (
        f"Traduis en {target_lang} le texte suivant:\n\n{text}\n\nRéponds uniquement avec la phrase traduite, sans rien ajouter.\nTraduction:"
    ),

    "ast": (
        f"Traduce al {target_lang} esti testu:\n\n{text}\n\nRespuende namás cola frase traducida, ensin amestar nada más.\nTraducción:"
    ),

    "aran": (
        f"Tradusís eth tèxte seguent ara lengua {target_lang}:\n\n{text}\n\nRespòn sonque damb era frasa tradusida, sense híger cap aute tèxte.\nTraduccion:"
    ),

    "gl": (
        f"Traduce ao {target_lang} o seguinte texto:\n\n{text}\n\nResponde só coa frase traducida, sen engadir nada máis.\nTradución:"
    )
}


    if source_lang not in prompts:
        raise ValueError(f"Idioma non soportáu: {source_lang}")

    return prompts[source_lang]

def translate(model, tokenizer, text, target_lang, source_lang, max_new_tokens=128):
    prompt = buildTranslationPrompt(text, target_lang, source_lang)
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    marker = prompt.split("\n")[-1]  # última línea del prompt
    translation = decoded.split(marker)[-1].strip()

    return translation

def evaluateTranslation(model, tokenizer, original, reference, target_lang, source_lang):
    """
    1. Traduce el texto a un idioma
    2. Calcula BLEU y chrF entre ambas traducciones
    """
    
    translated = translate(model, tokenizer, original, target_lang, source_lang)

    # SacreBLEU espera listas
    bleu_score = bleu.corpus_score([translated], [[reference]]).score
    chrf_score = chrf.corpus_score([translated], [[reference]]).score

    return {
        "reference": reference,
        "translated": translated,
        "BLEU": bleu_score,
        "chrF": chrf_score
    }

def roundTripEvaluation(model, tokenizer, text, target_lang, source_lang):
    """
    1. Traduce del idioma original al español
    2. Traduce del español de vuelta al idioma original
    3. Compara original vs. vuelta con BLEU y chrF
    """
    # 1. Ida
    intermedio = translate(model, tokenizer, text, target_lang, source_lang)

    # 2. Vuelta
    vuelta = translate(model, tokenizer, intermedio, target_lang, source_lang)

    # 3. Métricas
    bleu_score = bleu.corpus_score([vuelta], [[text]]).score
    chrf_score = chrf.corpus_score([vuelta], [[text]]).score

    return {
        "original": text,
        "intermedio": intermedio,
        "vuelta": vuelta,
        "BLEU": bleu_score,
        "chrF": chrf_score
    }

def roundTripMulti(model, tokenizer, text, langs, source_lang):
    resultados = {}
    for lang in langs:
        try:
            res = roundTripEvaluation(model, tokenizer, text, lang, source_lang)
        except:
            res = None
        resultados[lang] = res
    return resultados
