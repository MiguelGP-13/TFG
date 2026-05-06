
import random
import time
import pandas as pd
from groq import Groq
from .LanguageDatasets import LanguageDataset

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


TEMPLATES = {
    "asturiano": [
        "Resume esti testu n'asturiano:",
        "Descríbeme esti conteníu con otres pallabres:",
        "Cambia esti testu a un tonu más formal:",
        "Cambia esti testu a un tonu más informal:",
        "Esplica esti conteníu como si fuera pa un neñu:",
        "Esplica esti conteníu como si fuera pa un adultu:",
        "Inventa un diálogu curtín inspiráu nesti testu:",
        "Inventa un cuentu curtín basáu nesti fragmentu:",
        "Da un conseyu relacionáu con esti conteníu:",
        "Resume esti testu como si fuera un titular:"
    ],
    "gallego": [
        "Resume este texto en galego:",
        "Descríbeme este contido con outras palabras:",
        "Cambia este texto a un ton máis formal:",
        "Cambia este texto a un ton máis informal:",
        "Explica este contido como se fose para un neno:",
        "Explica este contido como se fose para un adulto:",
        "Inventa un diálogo curto inspirado neste texto:",
        "Inventa un conto curto baseado neste fragmento:",
        "Dá un consello relacionado con este contido:",
        "Resume este texto como se fose un titular:"
    ],
    "aranes": [
        "Resumís aguest tèxte en aranés:",
        "Descriu aguest contengut damb d'autes paraules:",
        "Càmbia aguest tèxte a un ton mès formau:",
        "Càmbia aguest tèxte a un ton mès informau:",
        "Explique aguest contengut coma entà un mainatge:",
        "Explique aguest contengut coma entà un adult:",
        "Invente un dialòg brac inspirat en aguest tèxte:",
        "Invente un petit raconte basat en aguest fragment:",
        "Done un conselh relacionat damb aguest contengut:",
        "Resumís aguest tèxte coma s'ère un titular:"
    ]
}

QA_TEMPLATES = {
    "asturiano": [
        "Inventa una pregunta razonable n'asturiano sobre esti testu y da una respuesta clara.",
        "Crea una pregunta útil basada nesti conteníu y da una respuesta completa.",
        "Xenera una pregunta de comprensión lectora y respóndela con detalle.",
        "Inventa una pregunta sencilla sobre esti fragmentu y da una respuesta curtia pero natural.",
        "Crea una pregunta abierta inspirada nesti testu y da una respuesta razonada."
    ],
    "gallego": [
        "Inventa unha pregunta razoable en galego sobre este texto e dá unha resposta clara.",
        "Crea unha pregunta útil baseada neste contido e ofrece unha resposta completa.",
        "Xera unha pregunta de comprensión lectora e respóndea con detalle.",
        "Inventa unha pregunta sinxela sobre este fragmento e dá unha resposta curta pero natural.",
        "Crea unha pregunta aberta inspirada neste texto e dá unha resposta razoada."
    ],
    "aranes": [
        "Invente ua question rasonabla en aranés sus aqueste tèxte e done ua responsa clara.",
        "Cree ua question util basada en aguest contengut e done ua responsa completa.",
        "Gènere ua question de compreneson deth tèxte e respòn en detalh.",
        "Invente ua question simpla sus aguest fragment e done ua responsa braca mès naturau.",
        "Cree ua question dubèrta inspirada en aguest tèxte e done ua responsa rasonada."
    ]
}

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
    if language not in TEMPLATES:
        raise ValueError(f"Idioma '{language}' non soportado.")

    templates = TEMPLATES[language]
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
                "Devuelve SOLO en este formato EXACTO:\n"
                "<instruccion> ... </instruccion>\n"
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

            # Parseo seguro
            if "<instruccion>" not in content or "</instruccion>" not in content:
                continue
            if "<respuesta>" not in content or "</respuesta>" not in content:
                continue

            instr = content.split("<instruccion>")[1].split("</instruccion>")[0].strip()
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

def generateInstructivoQA(
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
    if language not in QA_TEMPLATES:
        raise ValueError(f"Idioma '{language}' non soportado.")

    templates = QA_TEMPLATES[language]
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
                "Devuelve SOLO en este formato EXACTO:\n"
                "<pregunta> ... </pregunta>\n"
                "<respuesta> ... </respuesta>"
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
