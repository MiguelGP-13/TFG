
import random
import time
import pandas as pd
from groq import Groq
from LanguageDatasets import LanguageDataset

def safe_chat_completion(client, model, messages, sleep_time=0.5, max_retries=5): 
    """ Llama a client.chat.completions.create con reintentos automáticos. Si falla (por ejemplo, error 429 o timeout), espera sleep_time y reintenta. """ 
    for attempt in range(max_retries): 
        try: 
            return client.chat.completions.create( model=model, messages=messages ) 
        except Exception as e: # Último intento → relanzar error 
            if attempt == max_retries - 1: 
                raise e # Espera antes del siguiente intento 
            wait = sleep_time * (attempt + 1) # backoff lineal 
            print(f"Error en intento {attempt+1}: {e}. Reintentando en {wait} segundos...") 
            time.sleep(wait)

def generateDatasetOrtografico(dataset: LanguageDataset, api_key, model="openai/gpt-oss-120b", save = True, max_errors=4, min_errors=0, sleep_time=0.15, max_retries=5):
    client = Groq(api_key=api_key)
    res_list = []
    i = 0
    total = len(dataset)
    for original in dataset:
        callBegin = time.time()
        n_errors = random.randint(min_errors, max_errors)
        try:
            modified = safe_chat_completion(client, sleep_time=sleep_time, max_retries=max_retries,
                model=model,
                messages=[
                    {"role": "user", "content": f"Modifica esta frase, añadiendole {n_errors} errores gramaticales, ortográficos o léxicos. No añadas más contenido a la frase ni cambies el significado. Devuelve solo la frase modificada: '{original["text"]}'"}
                ]
            )
            res_list.append((original["text"], modified.choices[0].message.content, n_errors))
            callEnd = time.time()
        except Exception as e: # No perder todo lo conseguido si hay token limit o keybord interrupt
            print(e)
            return pd.DataFrame(res_list, columns=["original","modified", "n_errors"])
        # ---- PROGRESO ---- 
        i += 1
        pct = (i / total) * 100 
        step = int(callEnd - callBegin)
        remaining = step * (total - i)
        print(f"\rProgreso: {pct:5.1f}% ({i}/{total}) | step: {step} s, remaining time: {remaining// 60} min y {remaining - 60 * (remaining // 60)} s", end="") 
    print("\rDataset Generado") # salto de línea al terminar

    res_df = pd.DataFrame(res_list, columns=["original","modified", "n_errors"])
    if save:
        date = time.localtime(time.time())
        res_df.to_csv(f"{dataset.language}_{model.split("/")[-1]}_{time.strftime("%m-%d_%H-%M-%S", date)}")
    return res_df


def generateDatasetOrtograficoAnotado(
        dataset: LanguageDataset,
        api_key,
        model="openai/gpt-oss-120b",
        save=True,
        max_errors=4,
        min_errors=0,
        sleep_time=0.15,
        max_retries=5):

    client = Groq(api_key=api_key)
    res_list = []
    i = 0
    total = len(dataset)

    for original in dataset:
        callBegin = time.time()
        n_errors = random.randint(min_errors, max_errors)
        try:
            if n_errors == 0:
                annotated = original['text']
            else:
                prompt = (
                    f"Introduce {n_errors} errores en esta frase. "
                    f"Usa los tipos (elige cual aleatoriamente): ort (ortográfico), lex (léxico), add (añadir palabra), reord (reordenar)."
                    f"Marca cada error con <err t=TIPO>...</err>. "
                    f"No cambies el significado general. "
                    f"Devuelve solo la frase anotada:\n{original['text']}"
                )

                modified = safe_chat_completion(
                    client,
                    sleep_time=sleep_time,
                    max_retries=max_retries,
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                annotated = modified.choices[0].message.content.strip()
                
            res_list.append((original["text"], annotated, n_errors))
            callEnd = time.time()

        except Exception as e:
            print(e)
            return pd.DataFrame(res_list, columns=["original", "annotated", "n_errors"])

        # progreso
        i += 1
        pct = (i / total) * 100
        step = int(callEnd - callBegin)
        remaining = step * (total - i)
        print(
            f"\rProgreso: {pct:5.1f}% ({i}/{total}) | step: {step} s, "
            f"remaining time: {remaining//60} min {remaining%60} s",
            end=""
        )

    print("\nDataset Generado")

    res_df = pd.DataFrame(res_list, columns=["original", "annotated", "n_errors"])

    if save:
        date = time.localtime(time.time())
        filename = f"{dataset.language}_{model.split('/')[-1]}_{time.strftime('%m-%d_%H-%M-%S', date)}.csv"
        res_df.to_csv(filename, index=False)

    return res_df

def generateDatasetHuecos(
        dataset: LanguageDataset,
        save=True):

    res_list = []

    for original in dataset:
        text = original["text"].strip()

        # separamos por palabras 
        tokens = text.split()
        if len(tokens) < 2:
            # si la frase es demasiado corta, la saltamos
            continue

        # elegir palabra aleatoria
        idx = random.randint(0, len(tokens) - 1)
        missing_word = tokens[idx]

        # crear frase con hueco
        tokens_masked = tokens.copy()
        tokens_masked[idx] = "<mask>"
        masked_sentence = " ".join(tokens_masked)

        res_list.append((text, masked_sentence, missing_word))

    df = pd.DataFrame(res_list, columns=["original", "masked_sentence", "missing_word"])

    if save:
        date = time.localtime(time.time())
        filename = f"huecos_{dataset.language}_{time.strftime('%m-%d_%H-%M-%S', date)}.csv"
        df.to_csv(filename, index=False)

    return df
