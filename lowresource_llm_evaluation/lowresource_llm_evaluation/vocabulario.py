import re
from sacrebleu.metrics import BLEU, CHRF
import Levenshtein
from .utils import generar_texto
from .constants.prompts import buildFillMaskPrompt

bleu = BLEU()
chrf = CHRF()

def extraer_palabra(decoded, prompt, cortar=False):
    # Parte del prompt donde empieza la respuesta
    anchor = prompt.split("\n")[-1]

    if not decoded:
        return ""

    # Intentar separar por el anchor
    if anchor in decoded:
        clean = decoded.split(anchor, 1)[-1].strip()
    else:
        # Si no aparece, usar todo el texto generado
        clean = decoded.strip()
    if cortar:

            m = re.match(r"[^\W\d_]+", clean, flags=re.UNICODE)
            if m:
                clean = m.group(0)
            else:
                clean = decoded.strip() 

    return clean


def evaluacionHuecos(model, tokenizer, masked_sentence, missing_word, lang="es", device="cuda",
                     max_new_tokens=50, kaggle =False, cortar= False):
    """
    Evalúa la capacidad del modelo para predecir la palabra que falta (<mask>).
    Devuelve:
      - accuracy exacta
      - accuracy ignorando mayúsculas
      - distancia Levenshtein
    """

    # 1. Construir prompt
    prompt = buildFillMaskPrompt(masked_sentence, lang)

    # 2. Generar predicción determinista (top-1)
    decoded = generar_texto(model, tokenizer, prompt, device, max_new_tokens, kaggle)

    # 3. Extraer solo la palabra predicha
    pred = extraer_palabra(decoded, prompt, cortar)


    # 4. Métricas principales
    accuracy = 1.0 if pred == missing_word else 0.0
    accuracy_lower = 1.0 if pred.lower() == missing_word.lower() else 0.0

    # 5. Distancia Levenshtein (similaridad)
    lev_distance = Levenshtein.distance(pred, missing_word)

    return {
        "masked": masked_sentence,
        "missing_word": missing_word,
        "predicted": pred,

        # métricas útiles
        "accuracy": accuracy,
        "accuracy_lower": accuracy_lower,
        "levenshtein": lev_distance
    }
