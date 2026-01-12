import re
import math
import random
from sacrebleu.metrics import BLEU, CHRF
import Levenshtein

bleu = BLEU()
chrf = CHRF()

def buildFillMaskPrompt(masked_sentence, lang):
    prompts = {
        "es": (
            f"En esta frase falta una palabra. "
            f"Reemplaza <mask> por la palabra correcta. "
            f"Devuelve solo la palabra que falta:\n{masked_sentence}\nPalabra:"
        ),
        "ast": (
            f"Nesta frase falta una pallabra. "
            f"Rellena <mask> cola pallabra correuta. "
            f"Devuelve namás la pallabra que falta:\n{masked_sentence}\nPallabra:"
        ),
        "gl": (
            f"Nesta frase falta unha palabra. "
            f"Substitúe <mask> pola palabra correcta. "
            f"Devolve só a palabra que falta:\n{masked_sentence}\nPalabra:"
        ),
        "aran": (
            f"En aguesta frasa manque ua paraula. "
            f"Remplace <mask> pera paraula corrècta. "
            f"Da sonque era paraula que manque:\n{masked_sentence}\nParaula:"
        ),
        "fr": (
            f"Il manque un mot dans cette phrase. "
            f"Remplace <mask> par le mot correct. "
            f"Donne seulement le mot manquant:\n{masked_sentence}\nMot:"
        ),
    }
    return prompts[lang]


def evaluacionHuecos(model, tokenizer, masked_sentence, missing_word, lang="es",
                     max_new_tokens=8, top_k=3):
    """
    Evalúa la capacidad del modelo para predecir la palabra que falta (<mask>).
    Devuelve:
      - accuracy exacta
      - accuracy ignorando mayúsculas
      - distancia Levenshtein
      - top-k accuracy
      - lista de candidatos generados
    """

    # 1. Construir prompt
    prompt = buildFillMaskPrompt(masked_sentence, lang)

    # 2. Generar predicción determinista (top-1)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 3. Extraer solo la palabra predicha
    pred = decoded.split(prompt.split("\n")[-1])[-1].strip().split()[0]

    # 4. Métricas principales
    accuracy = 1.0 if pred == missing_word else 0.0
    accuracy_lower = 1.0 if pred.lower() == missing_word.lower() else 0.0

    # 5. Distancia Levenshtein (similaridad)
    lev_distance = Levenshtein.distance(pred, missing_word)

    # 6. Top‑k accuracy (sampling)
    topk_preds = set([pred])
    for _ in range(top_k - 1):
        outputs_k = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )
        decoded_k = tokenizer.decode(outputs_k[0], skip_special_tokens=True)
        pred_k = decoded_k.split(prompt.split("\n")[-1])[-1].strip().split()[0]
        topk_preds.add(pred_k)

    topk_accuracy = 1.0 if missing_word in topk_preds else 0.0

    return {
        "masked": masked_sentence,
        "missing_word": missing_word,
        "predicted": pred,

        # métricas útiles
        "accuracy": accuracy,
        "accuracy_lower": accuracy_lower,
        "levenshtein": lev_distance,
        "topk_accuracy": topk_accuracy,
        "topk_candidates": list(topk_preds)
    }
