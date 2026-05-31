from sacrebleu.metrics import BLEU, CHRF
from .utils import generar_texto
from .constants.prompts import buildTranslationPrompt

bleu = BLEU()
chrf = CHRF()


def translate(model, tokenizer, text, target_lang, source_lang, device="cuda", max_new_tokens=128, kaggle=False):
    prompt = buildTranslationPrompt(text, target_lang, source_lang)
    decoded = generar_texto(model, tokenizer, prompt, device, max_new_tokens, kaggle)

    for marker in [prompt.split("\n")[-1], ":"]:  # última línea del prompt
        decoded = decoded.split(marker)[-1].strip()
    
    # print(prompt)

    return decoded

def evaluateTranslation(model, tokenizer, original, reference, target_lang, source_lang, device, max_new_tokens, kaggle= False):
    """
    1. Traduce el texto a un idioma
    2. Calcula BLEU y chrF entre ambas traducciones
    """
    
    translated = translate(model, tokenizer, original, target_lang, source_lang, device, max_new_tokens, kaggle)

    # SacreBLEU espera listas
    bleu_score = bleu.corpus_score([translated], [[reference]]).score
    chrf_score = chrf.corpus_score([translated], [[reference]]).score

    return {
        "translated": translated,
        "BLEU": bleu_score,
        "chrF": chrf_score
    }

def roundTripEvaluation(model, tokenizer, text, intermediary_lang, source_lang, device, max_new_tokens, kaggle=False):
    """
    1. Traduce del idioma original al español
    2. Traduce del español de vuelta al idioma original
    3. Compara original vs. vuelta con BLEU y chrF
    """
    # 1. Ida
    intermedio = translate(model, tokenizer, text, intermediary_lang, source_lang, device, max_new_tokens, kaggle)

    # 2. Vuelta
    vuelta = translate(model, tokenizer, intermedio, source_lang, intermediary_lang, device, max_new_tokens, kaggle)

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
