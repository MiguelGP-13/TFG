
import random
import time
import pandas as pd
from groq import Groq
from .LanguageDatasets import LanguageDataset
from .constants.prompts import GenerateInstructivePrompts

def quitarYaAnotadas(dataset: LanguageDataset, anotadas: pd.DataFrame):
    ya = set(anotadas["original"].astype(str))
    res = LanguageDataset(dataset.language)
    res.json = [entry for entry in dataset if entry["text"] not in ya]
    return res

def safe_chat_completion_groq(client, model, messages, sleep_time=0.5, max_retries=5): 
    """ Llama a client.chat.completions.create con reintentos automáticos. Si falla (por ejemplo, error 429 o timeout), espera sleep_time y reintenta. """ 
    for attempt in range(max_retries): 
        try: 
            return client.chat.completions.create( model=model, messages=messages ) 
        except Exception as e: # Último intento → relanzar error 
            if attempt == max_retries - 1: 
                raise e # Espera antes del siguiente intento 
            wait = sleep_time * (attempt + 1) # backoff lineal 
            print(f"\nError en intento {attempt+1}: {e}. Reintentando en {wait} segundos...") 
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
            modified = safe_chat_completion_groq(client, sleep_time=sleep_time, max_retries=max_retries,
                model=model,
                messages=[
                    {"role": "user", "content": f"Modifica esta frase, añadiendole {n_errors} errores gramaticales, ortográficos o léxicos. No añadas más contenido a la frase ni cambies el significado. Devuelve solo la frase modificada: \"{original['text']}\""}
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
        res_df.to_csv(f"{dataset.language}_{model.split('/')[-1]}_{time.strftime('%m-%d_%H-%M-%S', date)}")
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

                modified = safe_chat_completion_groq(
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
            print("Guardando progreso actual")
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
        save=True, directory:str= None):

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
        if directory:
            filename = directory + "/" + filename
        df.to_csv(filename, index=False)

    return df

def generateInstructivoDataset(
        dataset,
        api_key,
        model="openai/gpt-oss-120b",
        N=2000,
        sleep_time=0.5,
        max_retries=5,
        save=False
    ):

    client = Groq(api_key=api_key)
    i= 0
    language = dataset.language
    if language not in GenerateInstructivePrompts.TEMPLATES:
        raise ValueError(f"Idioma '{language}' non soportado.")

    templates = GenerateInstructivePrompts.TEMPLATES[language]
    res_list = []

    base_texts = random.sample(dataset.json, N) if len(dataset.json) > N else dataset.json
    total = len(base_texts)
    i = 0

    for item in base_texts:
        callBegin = time.time()
        try:
            original = item["text"]
            template = random.choice(templates)
            instr = template + '\n' + original
            prompt_user = (
                f"{instr}\n\n"
                f"Devuelve SOLO en este formato EXACTO y respondiendo solo en {language}:\n"
                "<respuesta> ... </respuesta>"
            )
            # Llamada segura
            modified = safe_chat_completion_groq(client, sleep_time=sleep_time, max_retries=max_retries,
                model=model,
                messages=[
                    {"role": "user", "content": prompt_user}
                ]
            ).choices[0].message.content
            if modified is None:
                continue

            content = modified.strip()

            if "<respuesta>" not in content or "</respuesta>" not in content:
                continue

            
            resp  = content.split("<respuesta>")[1].split("</respuesta>")[0].strip()

            if len(instr) < 3 or len(resp) < 3:
                continue

            instruct_text = (
                "<|user|>\n" + instr + "\n"
                "<|assistant|>\n" + resp + "\n"
            )

            res_list.append(instruct_text)
            callEnd = time.time()

        except Exception as e:
            print(e)
            print("Guardando progreso actual")
            df = res_list
            return res_list

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

    print(f"\nDataset instructivo generado ({language}): {len(res_list)} ejemplos válidos.")

    if save:
        df = pd.DataFrame({"text": res_list})
        date = time.localtime(time.time())
        filename = f"{dataset.language}_{model.split('/')[-1]}_{time.strftime('%m-%d_%H-%M-%S', date)}"
        df.to_csv(filename, index=False)
        print(f"Guardado en {filename}")

    return res_list

def generateInstructivoQADataset(
        dataset,
        api_key,
        model="openai/gpt-oss-120b",
        N=2000,
        sleep_time=0.5,
        max_retries=5,
        save=False
    ):

    client = Groq(api_key=api_key)
    i= 0
    language = dataset.language
    if language not in GenerateInstructivePrompts.QA_TEMPLATES:
        raise ValueError(f"Idioma '{language}' non soportado.")

    templates = GenerateInstructivePrompts.QA_TEMPLATES[language]
    res_list = []

    base_texts = random.sample(dataset.json, N) if len(dataset.json) > N else dataset.json
    total = len(base_texts)
    i = 0

    for item in base_texts:
        callBegin = time.time()
        try:
            original = item["text"]
            template = random.choice(templates)

            prompt_user = (
                f"{template}\n\n"
                f"Texto base:\n{original}\n\n"
                f"Devuelve SOLO en este formato EXACTO, todo en {language}:\n"
                "<pregunta> GENERA SIEMPRE PREGUNTA </pregunta>\n"
                "<respuesta> GENERA SIEMPRE RESUPESTA </respuesta>"
            )

            modified = modified = safe_chat_completion_groq(client, sleep_time=sleep_time, max_retries=max_retries,
                model=model,
                messages=[
                    {"role": "user", "content": prompt_user}
                ]
            ).choices[0].message.content

            if modified is None:
                continue

            content = modified.strip()

            if "<pregunta>" not in content or "</pregunta>" not in content:
                continue
            if "<respuesta>" not in content or "</respuesta>" not in content:
                continue

            q = content.split("<pregunta>")[1].split("</pregunta>")[0].strip()
            a = content.split("<respuesta>")[1].split("</respuesta>")[0].strip()

            if len(q) < 3 or len(a) < 3:
                continue

            instruct_text = (
                "<|user|>\n" + q + "\n"
                "<|assistant|>\n" + a + "\n"
            )

            res_list.append(instruct_text)
            callEnd = time.time()

        except Exception as e:
            print(e)
            print("Guardando progreso actual")
            return res_list

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

    print(f"\nDataset QA generado ({language}): {len(res_list)} ejemplos válidos.")

    if save:
        df = pd.DataFrame({"text": res_list})
        date = time.localtime(time.time())
        filename = f"{dataset.language}_{model.split('/')[-1]}_{time.strftime('%m-%d_%H-%M-%S', date)}"
        df.to_csv(filename, index=False)
        print(f"Guardado en {filename}")

    return res_list
