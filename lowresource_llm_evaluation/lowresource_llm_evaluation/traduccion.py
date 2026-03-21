from sacrebleu.metrics import BLEU, CHRF

bleu = BLEU()
chrf = CHRF()

def buildTranslationPrompt(text, target_lang, source_lang):
    """
    Devuelve un prompt en el idioma adecuado según source_lang,
    usando nombres de idiomas adaptados a cada lengua.
    """

    # Diccionario de nombres de idiomas por idioma origen
    LANG_NAMES = {
        "es": {"es": "español", "fr": "francés", "en": "inglés", "gl": "gallego", "ast": "asturiano", "aran": "aranés", "pt": "portugués", "it": "italiano", "de": "alemán"},
        "fr": {"es": "espagnol", "fr": "français", "en": "anglais", "gl": "galicien", "ast": "asturien", "aran": "aranés", "pt": "portugais", "it": "italien", "de": "allemand"},
        "en": {"es": "spanish", "fr": "french", "en": "english", "gl": "galician", "ast": "asturian", "aran": "aranese", "pt": "portuguese", "it": "italian", "de": "german"},
        "gl": {"es": "español", "fr": "francés", "en": "inglés", "gl": "galego", "ast": "asturiano", "aran": "aranés", "pt": "portugués", "it": "italiano", "de": "alemán"},
        "ast": {"es": "español", "fr": "francés", "en": "inglés", "gl": "gallego", "ast": "asturiano", "aran": "aranés", "pt": "portugués", "it": "italiano", "de": "alemán"},
        "aran": {"es": "espanhòu", "fr": "francés", "en": "anglés", "gl": "galèc", "ast": "asturianu", "aran": "aranés", "pt": "portugués", "it": "italian", "de": "alemand"}
    }

    if source_lang not in LANG_NAMES:
        raise ValueError(f"Idioma no soportado: {source_lang}. Elija uno de [{LANG_NAMES.keys()}]")

    if target_lang not in LANG_NAMES[source_lang]:
        raise ValueError(f"Idioma destino non reconocido: {target_lang}. Elija uno de [{LANG_NAMES.keys()}]")

    # Nombre del idioma destino adaptado al idioma origen
    target_name = LANG_NAMES[source_lang][target_lang]

    prompts = {
        "es": (
            f"Traduce al {target_name} el siguiente texto y responde únicamente con la frase traducida, sin añadir nada más:\n\n{text}\n\nTraducción:"
        ),

        "fr": (
            f"Traduisez en {target_name} le texte suivant et répondez uniquement avec la phrase traduite, sans rien ajouter:\n\n{text}\n\nTraduction:"
        ),

        "ast": (
            f"Traduce al {target_name} esti testu y respuende namás cola frase traducida, ensin amestar nada más:\n\n{text}\n\nTraducción:"
        ),

        "aran": (
            f"Tradusís eth tèxte seguent ara lengua {target_name} e respòn sonque damb era frasa tradusida, sense híger cap aute tèxte:\n\n{text}\n\nTraduccion:"
        ),

        "gl": (
            f"Traduce ao {target_name} o seguinte texto e responde unicamente coa frase traducida, sen engadir nada máis:\n\n{text}\n\nTradución:"
        )
    }

    return prompts[source_lang]


def translate(model, tokenizer, text, target_lang, source_lang, device="cuda", max_new_tokens=128):
    prompt = buildTranslationPrompt(text, target_lang, source_lang)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # To avoid warning
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    #

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    for marker in [prompt.split("\n")[-1], ":"]:  # última línea del prompt
        decoded = decoded.split(marker)[-1].strip()
    
    # print(prompt)

    return decoded

def evaluateTranslation(model, tokenizer, original, reference, target_lang, source_lang, device, max_new_tokens):
    """
    1. Traduce el texto a un idioma
    2. Calcula BLEU y chrF entre ambas traducciones
    """
    
    translated = translate(model, tokenizer, original, target_lang, source_lang, device, max_new_tokens)

    # SacreBLEU espera listas
    bleu_score = bleu.corpus_score([translated], [[reference]]).score
    chrf_score = chrf.corpus_score([translated], [[reference]]).score

    return {
        "translated": translated,
        "BLEU": bleu_score,
        "chrF": chrf_score
    }

def roundTripEvaluation(model, tokenizer, text, target_lang, source_lang, device, max_new_tokens):
    """
    1. Traduce del idioma original al español
    2. Traduce del español de vuelta al idioma original
    3. Compara original vs. vuelta con BLEU y chrF
    """
    # 1. Ida
    intermedio = translate(model, tokenizer, text, target_lang, source_lang, device, max_new_tokens)

    # 2. Vuelta
    vuelta = translate(model, tokenizer, intermedio, source_lang, target_lang, device, max_new_tokens)

    # 3. Métricas
    bleu_score = bleu.corpus_score([vuelta], [[text]]).score
    chrf_score = chrf.corpus_score([vuelta], [[text]]).score

    return {
        "intermediate": intermedio,
        "return": vuelta,
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
