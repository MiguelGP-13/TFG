import re
from sacrebleu.metrics import BLEU, CHRF
import Levenshtein
from .utils import generar_texto

bleu = BLEU()
chrf = CHRF()

def buildCorrectionPrompt(text, lang):
    prompts = {
        "es": (
            f"Corrige esta frase en español. "
            f"No añadas contenido nuevo. "
            f"Devuelve solo la frase corregida:\n{text}\nCorrección:"
        ),
        "ast": (
            f"Corrige esta frase n'asturianu. "
            f"Nun añadas conteníu nuevu. "
            f"Devuelve namás la frase correxida:\n{text}\nCorreición:"
        ),
        "gl": (
            f"Corrige esta frase en galego. "
            f"Non engadas contido novo. "
            f"Devolve só a frase corrixida:\n{text}\nCorrección:"
        ),
        "aran": (
            f"Corrigís aguesta frasa en aranés. "
            f"Non híges contengut nau. "
            f"Da sonque era frasa corregida:\n{text}\nCorreccion:"
        ),
        "fr": (
            f"Corrige cette phrase en français. "
            f"N’ajoute aucun contenu. "
            f"Donne seulement la phrase corrigée:\n{text}\nCorrection:"
        ),
    }

    return prompts[lang]


def evaluacionOrtograficaAnotado(model, tokenizer, annotated, original, lang="es", device="cuda", max_new_tokens=128, kaggle= False, cortar=False):
    """
    annotated = frase con <err t=...>...</err>
    original  = frase correcta
    lang      = idioma del modelo evaluado
    """

    # 1. Extraer errores anotados
    errores = re.findall(r'<err t=(.*?)>(.*?)</err>', annotated) # (tipo, error)
    n_errores = len(errores)

    # 2. Obtener frase incorrecta sin etiquetas
    incorrect = re.sub(r"</?err.*?>", "", annotated).strip()

    # 3. Construir prompt 
    prompt = buildCorrectionPrompt(incorrect, lang)

    # 4. Generar corrección con modelo a evaluar
    decoded = generar_texto(model, tokenizer, prompt, device, max_new_tokens, kaggle)
    corrected = decoded.split(prompt.split("\n")[-1])[-1].strip()
    if cortar:
        corrected = corrected.split(".", 1)[0].strip()

    # -----------------------------
    # 5. Métricas original vs errores corregidos
    bleu_score = bleu.corpus_score([corrected], [[original]]).score
    chrf_score = chrf.corpus_score([corrected], [[original]]).score
    edit_distance = Levenshtein.distance(corrected, original)


    # 6. Métricas basadas en anotación
    errores_corregidos = 0
    for tipo, contenido in errores:
        if contenido not in corrected:
            errores_corregidos += 1

    errores_no_corregidos = n_errores - errores_corregidos

    # Errores nuevos
    errores_nuevos = 0
    for tok in corrected.split():
        if tok not in original.split() and all(tok not in e[1] for e in errores):
            errores_nuevos += 1

    # Precisión, recall, F1
    recall = errores_corregidos / n_errores if n_errores > 0 else 1.0
    precision = errores_corregidos / (errores_corregidos + errores_nuevos) if (errores_corregidos + errores_nuevos) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "original": original,
        "incorrect": incorrect,
        "corrected": corrected,

        # métricas clásicas
        "BLEU": bleu_score,
        "chrF": chrf_score,
        "Levenshtein": edit_distance,

        # anotación
        "errores_totales": n_errores,
        "errores_corregidos": errores_corregidos,
        "errores_no_corregidos": errores_no_corregidos,
        "errores_nuevos": errores_nuevos,

        # métricas de corrección
        "precision": precision,
        "recall": recall,
        "F1": f1,

        # errores con tipo
        "errores_detalle": errores
    }
