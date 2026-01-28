from .interferenciaLinguistica import calidadLengua
from .ortografica import evaluacionOrtograficaAnotado
from .traduccion import evaluateTranslation, roundTripEvaluation
from .vocabulario import evaluacionHuecos

import shutil
import numpy as np
import pandas as pd


def _titulo(texto: str):
    ancho = shutil.get_terminal_size().columns
    print("\n" + "=" * ancho)
    print(texto.center(ancho))
    print("=" * ancho + "\n")


def _tabla(diccionario):
    """
    Convierte un diccionario en una tabla ASCII bonita.
    """
    if diccionario is None or len(diccionario) == 0:
        return "(sin datos)"

    filas = [(str(k), str(v)) for k, v in diccionario.items()]

    ancho_k = max(len(k) for k, _ in filas)
    ancho_v = max(len(v) for _, v in filas)

    linea = "+" + "-" * (ancho_k + 2) + "+" + "-" * (ancho_v + 2) + "+"
    out = [linea, f"| {'Clave'.ljust(ancho_k)} | {'Valor'.ljust(ancho_v)} |", linea]

    for k, v in filas:
        out.append(f"| {k.ljust(ancho_k)} | {v.ljust(ancho_v)} |")

    out.append(linea)
    return "\n".join(out)

def _benchmark(
    model,
    tokenizer,
    df_textos,          # (source, target) o solo target
    lang_eval,
    df_huecos=None,
    df_anotado=None,
    lexicon_target=None,
    lexicons_comparison=None,
    n_samples_calidad=5,
    max_new_tokens=80,
    roundtrip_langs=None,
    debug: bool = False
):
    """
    Benchmark interno. Si debug=False, devuelve solo medias y ejemplos.
    """

    resultados = {}

    # ---------------------------------------------------------
    # Inferir columnas
    # ---------------------------------------------------------
    if df_textos.shape[1] == 1:
        col_target = df_textos.columns[0]
        col_source = None
    elif df_textos.shape[1] == 2:
        col_source, col_target = df_textos.columns.tolist()
    else:
        raise ValueError("df_textos debe tener 1 o 2 columnas.")

    # ---------------------------------------------------------
    # 1. CALIDAD DE LENGUA (generativa)
    # ---------------------------------------------------------
    calidad_raw = None
    if lexicon_target is not None and col_source is not None:
        try:
            reference_text = " ".join(df_textos[col_target].astype(str).tolist())

            textos_generados = []
            for _ in range(n_samples_calidad):
                res = calidadLengua(
                    model, tokenizer,
                    lexicon_target=lexicon_target,
                    reference_text=reference_text,
                    lexicons_comparison=lexicons_comparison
                )
                calidad_raw.append(res)
        except Exception:
            calidad_raw = None

    if debug or calidad_raw is None:
        resultados["calidad_lengua"] = calidad_raw
    else:
        # medias
        ttr_vals = [r["ttr"] for r in calidad_raw]
        ent_vals = [r["entropy"] for r in calidad_raw]
        ngram_vals = [r["ngram_overlap"] for r in calidad_raw]
        freq_target_vals = [r["freq_target"] for r in calidad_raw]
        calidad_vals = [r["calidad"] for r in calidad_raw]

        # freq_comparison es dict por idioma
        all_langs = set()
        for r in calidad_raw:
            all_langs.update(r["freq_comparison"].keys())

        freq_comp_media = {}
        for lang in all_langs:
            vals = []
            for r in calidad_raw:
                fc = r["freq_comparison"]
                if lang in fc:
                    vals.append(fc[lang])
            if vals:
                freq_comp_media[lang] = float(np.mean(vals))

        resultados["calidad_lengua"] = {
            "ejemplo": calidad_raw[0]["text"] if calidad_raw else None,
            "media": {
                "ttr": float(np.mean(ttr_vals)) if ttr_vals else None,
                "entropy": float(np.mean(ent_vals)) if ent_vals else None,
                "ngram_overlap": float(np.mean(ngram_vals)) if ngram_vals else None,
                "freq_target": float(np.mean(freq_target_vals)) if freq_target_vals else None,
                "freq_comparison": freq_comp_media,
                "calidad": float(np.mean(calidad_vals)) if calidad_vals else None,
            }
        }

    # ---------------------------------------------------------
    # 2. TRADUCCIÓN DIRECTA
    # ---------------------------------------------------------
    trad_raw = None
    if col_source is not None and col_target is not None:
        trad_raw = []
        for _, row in df_textos.iterrows():
            try:
                res = evaluateTranslation(
                    model, tokenizer,
                    row[col_source],
                    row[col_target],
                    target_lang=lang_eval,
                    source_lang=None
                )
            except Exception:
                res = None
            trad_raw.append({
                "source": row[col_source],
                "reference": row[col_target],
                "resultado": res
            })

    if debug or trad_raw is None:
        resultados["traduccion"] = trad_raw
    else:
        bleu_vals = []
        chrf_vals = []
        for r in trad_raw:
            if r["resultado"] is not None:
                bleu_vals.append(r["resultado"].get("BLEU"))
                chrf_vals.append(r["resultado"].get("chrF"))

        ejemplo = None
        for r in trad_raw:
            if r["resultado"] is not None:
                ejemplo = {
                    "source": r["source"],
                    "reference": r["reference"],
                    "translated": r["resultado"].get("translated", None)
                }
                break

        resultados["traduccion"] = {
            "ejemplo": ejemplo,
            "media": {
                "BLEU": float(np.mean(bleu_vals)) if bleu_vals else None,
                "chrF": float(np.mean(chrf_vals)) if chrf_vals else None
            }
        }

    # ---------------------------------------------------------
    # 3. ROUND TRIP
    # ---------------------------------------------------------
    round_raw = None
    if col_target is not None and roundtrip_langs is not None and lang_eval is not None:
        round_raw = []
        for _, row in df_textos.iterrows():
            text = row[col_target]
            for lang in roundtrip_langs:
                try:
                    res = roundTripEvaluation(
                        model, tokenizer,
                        text,
                        lang,       # lengua intermedia
                        lang_eval   # lengua a evaluar
                    )
                except Exception:
                    res = None

                round_raw.append({
                    "original": text,
                    "intermedio_lang": lang,
                    "resultado": res
                })

    if debug or round_raw is None:
        resultados["round_trip"] = round_raw
    else:
        bleu_vals = []
        chrf_vals = []
        ejemplo = None

        for r in round_raw:
            if r["resultado"] is not None:
                bleu_vals.append(r["resultado"].get("BLEU"))
                chrf_vals.append(r["resultado"].get("chrF"))
                if ejemplo is None:
                    ejemplo = {
                        "original": r["resultado"].get("original", r["original"]),
                        "intermedio": r["resultado"].get("intermedio", None),
                        "vuelta": r["resultado"].get("vuelta", None)
                    }

        resultados["round_trip"] = {
            "ejemplo": ejemplo,
            "media": {
                "BLEU": float(np.mean(bleu_vals)) if bleu_vals else None,
                "chrF": float(np.mean(chrf_vals)) if chrf_vals else None
            }
        }

    # ---------------------------------------------------------
    # 4. VOCABULARIO
    # ---------------------------------------------------------
    vocab_raw = None
    if df_huecos is not None:
        vocab_raw = []
        for _, row in df_huecos.iterrows():
            try:
                res = evaluacionHuecos(
                    model, tokenizer,
                    row["masked_sentence"],
                    row["missing_word"],
                    lang=lang_eval
                )
            except Exception:
                res = None
            vocab_raw.append(res)

    if debug or vocab_raw is None:
        resultados["vocabulario"] = vocab_raw
    else:
        # medias
        acc_vals, acc_low_vals, lev_vals, topk_acc_vals = [], [], [], []
        for r in vocab_raw:
            if r is None:
                continue
            acc_vals.append(r.get("accuracy"))
            acc_low_vals.append(r.get("accuracy_lower"))
            lev_vals.append(r.get("levenshtein"))
            topk_acc_vals.append(r.get("topk_accuracy"))

        # ejemplos: df con 5 filas
        df_ej = None
        if df_huecos is not None and len(df_huecos) > 0:
            n = min(5, len(df_huecos))
            df_ej = df_huecos.copy().iloc[:n].reset_index(drop=True)
            df_ej["resultado"] = vocab_raw[:n]

        resultados["vocabulario"] = {
            "ejemplos": df_ej,
            "media": {
                "accuracy": float(np.mean(acc_vals)) if acc_vals else None,
                "accuracy_lower": float(np.mean(acc_low_vals)) if acc_low_vals else None,
                "levenshtein": float(np.mean(lev_vals)) if lev_vals else None,
                "topk_accuracy": float(np.mean(topk_acc_vals)) if topk_acc_vals else None
            }
        }

    # ---------------------------------------------------------
    # 5. ORTOGRAFÍA
    # ---------------------------------------------------------
    orto_raw = None
    if df_anotado is not None:
        orto_raw = []
        for _, row in df_anotado.iterrows():
            try:
                res = evaluacionOrtograficaAnotado(
                    model, tokenizer,
                    row["annotated_sentence"],
                    row["original_sentence"],
                    lang=lang_eval
                )
            except Exception:
                res = None
            orto_raw.append(res)

    if debug or orto_raw is None:
        resultados["ortografia"] = orto_raw
    else:
        acc_vals, acc_low_vals, lev_vals, topk_acc_vals = [], [], [], []
        for r in orto_raw:
            if r is None:
                continue
            acc_vals.append(r.get("accuracy"))
            acc_low_vals.append(r.get("accuracy_lower"))
            lev_vals.append(r.get("levenshtein"))
            topk_acc_vals.append(r.get("topk_accuracy"))

        df_ej = None
        if df_anotado is not None and len(df_anotado) > 0:
            n = min(5, len(df_anotado))
            df_ej = df_anotado.copy().iloc[:n].reset_index(drop=True)
            df_ej["resultado"] = orto_raw[:n]

        resultados["ortografia"] = {
            "ejemplos": df_ej,
            "media": {
                "accuracy": float(np.mean(acc_vals)) if acc_vals else None,
                "accuracy_lower": float(np.mean(acc_low_vals)) if acc_low_vals else None,
                "levenshtein": float(np.mean(lev_vals)) if lev_vals else None,
                "topk_accuracy": float(np.mean(topk_acc_vals)) if topk_acc_vals else None
            }
        }

    return resultados

def benchmark(
    model,
    tokenizer,
    df_textos,
    lang_eval,
    df_huecos=None,
    df_anotado=None,
    lexicon_target=None,
    lexicons_comparison=None,
    n_samples_calidad=5,
    max_new_tokens=80,
    roundtrip_langs=None,
    debug: bool = False
):
    """
    Ejecuta el benchmark completo del modelo lingüístico.

    Parámetros
    ----------
    model : objeto de modelo
        Modelo de lenguaje (por ejemplo, un modelo tipo Transformer).
    tokenizer : objeto tokenizer
        Tokenizador compatible con el modelo.
    df_textos : pandas.DataFrame
        DataFrame con 1 o 2 columnas:
        - 1 columna: solo idioma a evaluar (col_target).
        - 2 columnas: (source, target), donde:
            * source: otro idioma (entrada al modelo).
            * target: idioma a evaluar (referencia y corpus).
    lang_eval : str
        Código de la lengua a evaluar (por ejemplo, "es", "en").
    df_huecos : pandas.DataFrame, opcional
        DataFrame con columnas:
        - "masked_sentence"
        - "missing_word"
        Usado para la evaluación de vocabulario (huecos).
    df_anotado : pandas.DataFrame, opcional
        DataFrame con columnas:
        - "annotated_sentence"
        - "original_sentence"
        Usado para la evaluación ortográfica.
    lexicon_target : cualquier estructura léxica, opcional
        Léxico de la lengua objetivo para la métrica de calidad de lengua.
    lexicons_comparison : dict, opcional
        Diccionario {codigo_idioma: lexicon} para comparar interferencia lingüística
        con otros idiomas.
    n_samples_calidad : int, por defecto 5
        Número de textos generados para la evaluación de calidad de lengua.
    max_new_tokens : int, por defecto 80
        Número máximo de tokens nuevos a generar en cada muestra.
    roundtrip_langs : lista de str, opcional
        Lista de códigos de lenguas intermedias para la evaluación round-trip.
    debug : bool, por defecto False
        - Si True: devuelve todas las salidas completas de cada métrica.
        - Si False: devuelve solo medias y ejemplos representativos.

    Devuelve
    --------
    dict
        Diccionario con las claves:
        - "calidad_lengua"
        - "traduccion"
        - "round_trip"
        - "vocabulario"
        - "ortografia"

        Cada una contiene:
        - Si debug=True: lista/datos completos.
        - Si debug=False: medias y ejemplos, según el diseño acordado.
    """

    resultados = _benchmark(
        model=model,
        tokenizer=tokenizer,
        df_textos=df_textos,
        lang_eval=lang_eval,
        df_huecos=df_huecos,
        df_anotado=df_anotado,
        lexicon_target=lexicon_target,
        lexicons_comparison=lexicons_comparison,
        n_samples_calidad=n_samples_calidad,
        max_new_tokens=max_new_tokens,
        roundtrip_langs=roundtrip_langs,
        debug=debug
    )

    # ============================
    # Impresión bonita
    # ============================

    # 1. Calidad de lengua
    _titulo("Evaluación de Calidad de Lengua")
    print(_tabla(
        resultados.get("calidad_lengua", {}).get("media", {})
        if not debug and isinstance(resultados.get("calidad_lengua"), dict)
        else {"detalle": "ver estructura completa (debug=True)"}
    ))

    # 2. Traducción
    _titulo("Evaluación de Traducción")
    print(_tabla(
        resultados.get("traduccion", {}).get("media", {})
        if not debug and isinstance(resultados.get("traduccion"), dict)
        else {"detalle": "ver estructura completa (debug=True)"}
    ))

    # 3. Round Trip
    _titulo("Evaluación Round Trip")
    print(_tabla(
        resultados.get("round_trip", {}).get("media", {})
        if not debug and isinstance(resultados.get("round_trip"), dict)
        else {"detalle": "ver estructura completa (debug=True)"}
    ))

    # 4. Vocabulario
    _titulo("Evaluación de Vocabulario (Huecos)")
    print(_tabla(
        resultados.get("vocabulario", {}).get("media", {})
        if not debug and isinstance(resultados.get("vocabulario"), dict)
        else {"detalle": "ver estructura completa (debug=True)"}
    ))

    # 5. Ortografía
    _titulo("Evaluación Ortográfica")
    print(_tabla(
        resultados.get("ortografia", {}).get("media", {})
        if not debug and isinstance(resultados.get("ortografia"), dict)
        else {"detalle": "ver estructura completa (debug=True)"}
    ))

    _titulo("Benchmark completado")

    return resultados
