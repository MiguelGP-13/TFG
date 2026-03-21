import re
from sacrebleu.metrics import BLEU, CHRF
import Levenshtein

bleu = BLEU()
chrf = CHRF()

def buildFillMaskPrompt(masked_sentence, lang):
    prompts = {
        "es": (
            f"En la siguiente frase falta una palabra, reemplazada por <mask>. "
            f"Devuelve solo y únicamente la palabra faltante.\n"
            f"{masked_sentence}\nPalabra:"
        ),
        "ast": (
            f"Na siguiente frase falta una pallabra, reemplazada por <mask>. "
            f"Devuelve namás la pallabra que falta.\n"
            f"{masked_sentence}\nPallabra:"
        ),
        "gl": (
            f"Na seguinte frase falta unha palabra, substituída por <mask>. "
            f"Devolve só a palabra que falta.\n"
            f"{masked_sentence}\nPalabra:"
        ),
        "aran": (
            f"En aguesta frasa manque ua paraula, remplaçada per <mask>. "
            f"Da sonque era paraula que manque.\n"
            f"{masked_sentence}\nParaula:"
        ),
        "fr": (
            f"Dans la phrase suivante il manque un mot, remplacé par <mask>. "
            f"Donne seulement le mot manquant.\n"
            f"{masked_sentence}\nMot:"
        ),
    }
    return prompts[lang]



def extraer_palabra(decoded, prompt):
    # Parte del prompt donde empieza la respuesta
    anchor = prompt.split("\n")[-1]

    if not decoded:
        return ""

    # Intentar separar por el anchor
    if anchor in decoded:
        tail = decoded.split(anchor, 1)[-1].strip()
    else:
        # Si no aparece, usar todo el texto generado
        tail = decoded.strip()

    clean = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ ]", "", tail)

    return clean


def evaluacionHuecos(model, tokenizer, masked_sentence, missing_word, lang="es", device="cuda",
                     max_new_tokens=50):
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
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # To avoid warning
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    #
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 3. Extraer solo la palabra predicha
    pred = extraer_palabra(decoded, prompt)


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
