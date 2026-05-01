import os, re, json, html, hashlib, gzip, io, unicodedata
import regex
import pandas as pd
from datasets import Dataset
import numpy as np
import fasttext
import requests

lang_code = {
    "asturiano": {
        "tatoeba": "ast",
        "opus": "ast",
        "fasttext": "__label__ast"
    },
    "aranes": {
        "tatoeba": "oci",   # Tatoeba usa 'oci' (Occitan)
        "opus": "oc",       # OPUS usa 'oc'
        "fasttext": "__label__oc"
    },
    "aragones": {
        "tatoeba": "arg",
        "opus": "an",
        "fasttext": None    # FastText NO soporta aragonés
    },
    "gallego": {
        "tatoeba": "glg",
        "opus": "gl",
        "fasttext": "__label__gl"
    }
}


class LanguageDataset():
    def __init__(self, language=None,
             min_words=4, max_words=np.inf, max_word_len=25, 
             do_anonymize=True, filter_language_thr: int = False,
             initializeTatoeba=False, initializeLocal=False):

        if language is None:
            raise ValueError("Debe pasar 'language' o 'path' obligatoriamente.")

        if language not in lang_code.keys():
            raise KeyError(
                f'Lenguaje no contemplado ({list(lang_code.keys())})\n'
                'Si quiere crear el dataset desde un checkpoint previo, necesita poner path="path_to_your_dataset"'
            )

        # --- Atributos base ---
        self.raw_datasets = {}
        self.language = language
        self.language_codes = lang_code[language]
        self.json = []
        self.MIN_WORDS = min_words
        self.MAX_WORDS = max_words
        self.MAX_WORD_LEN = max_word_len
        self.do_anonymize = do_anonymize
        self.filter_language_thr = filter_language_thr

        # --- FastText model ---
        self.ft_model = None

        if self.filter_language_thr:
            # Validar soporte FastText
            if self.language_codes["fasttext"] is None:
                raise ValueError(
                    f"La lengua '{self.language}' no está soportada por FastText LID-176. "
                    "No se puede activar filter_language_thr=True."
                )

            # Ruta interna del modelo dentro del paquete
            model_dir = os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(model_dir, exist_ok=True)

            self.model_path = os.path.join(model_dir, "lid.176.ftz")

            # Descargar si no existe
            if not os.path.exists(self.model_path):
                print(f"[INFO] Descargando FastText LID-176 a {self.model_path} ...")
                url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
                import requests
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(self.model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("[INFO] Modelo FastText descargado correctamente.")

            # Cargar modelo
            self.ft_model = fasttext.load_model(self.model_path)

        # --- Inicializaciones opcionales ---
        if initializeLocal:
            self.startLocal()
        if initializeTatoeba:
            self.startTatoeba()

        

    def __getitem__(self, key):
        """
        Allow slicing or indexing directly on the object.
        - ast[:5] -> self.json[:5]
        - ast[0]  -> self.json[0]
        - ast["tatoeba"] -> all lines belonging to the 'tatoeba' dataset
        """
        if isinstance(key, (int, slice)):
            return self.json[key]
        elif isinstance(key, str):
            if key not in self.raw_datasets:
                raise KeyError(f"Dataset '{key}' not found in {self.raw_datasets.keys()}")
            bounds = self.raw_datasets[key]
            return self.json[bounds["start"]:bounds["end"]]
        else:
            raise TypeError("Key must be int, slice, or str")
        
    def __len__(self):
        return len(self.json)

    @property
    def hf_dataset(self):
        """Devuelve un Dataset de HuggingFace a partir de self.json"""
        return Dataset.from_list(self.json)
    
    def _saveIndexer(self, name, start, end):
        self.raw_datasets[name] = {"start": start, "end": end, "anonymous":self.do_anonymize}

    def is_target_language(self, text):
        """Devuelve True si el texto está en la lengua objetivo usando FastText."""
        ft_code = self.language_codes["fasttext"]
        lang, prob = self.ft_model.predict(text)
        return lang[0] == ft_code and prob[0] > self.filter_language_thr
    
    def filter_by_language(self, batch_size=512, top_k=3):
        """
        Filtra self.json por idioma respetando los índices de cada dataset.
        Recorre cada dataset por separado, filtra sus líneas y reconstruye
        self.json y self.raw_datasets con los nuevos índices.
        
        Usa FastText en batch y permite top_k para lenguas minoritarias.
        """

        if self.ft_model is None:
            raise RuntimeError(
                "FastText no está cargado. Active filter_language_thr en el __init__."
            )

        ft_code = self.language_codes["fasttext"]
        thr = self.filter_language_thr

        new_json = []
        new_raw = {}

        # Recorremos cada dataset en orden de carga
        for name in self.raw_datasets.keys():

            subset = self[name]   # gracias a __getitem__
            texts = [item["text"] for item in subset]
            keep = []

            # Procesar en batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                # FastText con top_k
                langs, probs = self.ft_model.predict(batch, k=top_k)

                for j in range(len(batch)):
                    if ft_code in langs[j]:
                        idx = langs[j].index(ft_code)
                        if probs[j][idx] >= thr:
                            keep.append(subset[i + j])

            # Guardar nuevos índices usando tu función
            new_start = len(new_json)
            new_json.extend(keep)
            new_end = len(new_json)

            self._saveIndexer(name, new_start, new_end)

        # Actualizar dataset global
        self.json = new_json

        return self

    def tokenize_batch(self, tokenizer, max_length=512):
        """
        Devuelve una función que tokeniza un batch de textos.
        Reutilizable por LanguageDataset.tokenize() y por split().
        """
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


        def _fn(batch):
            out = tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_length,
                padding="longest",
            )
            out["labels"] = out["input_ids"].copy()
            return out

        return _fn



    def tokenize(self, tokenizer, max_length=512, batched=True):
        fn = self.tokenize_batch(tokenizer, max_length)

        return self.hf_dataset.map(
            fn,
            batched=batched,
            remove_columns=self.hf_dataset.column_names,
        )

    def split(
        self,
        test_size=0.05,
        seed=42,
        tokenizer=None,
        max_length=512,
        batched=True,
    ):
        splits = self.hf_dataset.train_test_split(test_size=test_size, seed=seed)
        train = splits["train"]
        test = splits["test"]

        # Si no hay tokenizer → devolver texto crudo
        if tokenizer is None:
            return train, test

        # Reusar la misma función
        fn = self.tokenize_batch(tokenizer, max_length)

        train_tok = train.map(fn, batched=batched, remove_columns=train.column_names)
        test_tok  = test.map(fn, batched=batched, remove_columns=test.column_names)

        return train_tok, test_tok

    def concatenate(self, tokenizer, max_tokens=1024):
        """
        Concatena líneas respetando el origen de cada dataset.
        No mezcla textos de distintos datasets en el mismo bloque.
        """
        if not self.json or not self.raw_datasets:
            print("Error inesperado en el concatenate!!")
            return self

        print(f"🔗 Concatenando por orígenes...")
        
        # 1. Estimación de ratio (igual que antes, para velocidad)
        sample_text = " ".join([self.json[i]['text'] for i in range(min(500, len(self.json)))])
        num_tokens = len(tokenizer.encode(sample_text, add_special_tokens=False))
        chars_per_token = len(sample_text) / num_tokens if num_tokens > 0 else 3.5
        max_chars = max_tokens * chars_per_token

        new_json = []
        new_raw_datasets = {}

        # 2. Procesamos cada dataset por separado
        for name, bounds in self.raw_datasets.items():
            start_idx = len(new_json) # Nuevo inicio para este dataset
            
            # Extraemos las líneas que pertenecen a este dataset original
            subset = self.json[bounds["start"]:bounds["end"]]
            
            current_batch = []
            current_chars = 0

            for item in subset:
                line = item["text"].strip()
                line_chars = len(line)

                if current_chars + line_chars > max_chars and current_batch:
                    new_json.append({"text": "\n".join(current_batch)})
                    current_batch = []
                    current_chars = 0
                
                current_batch.append(line)
                current_chars += line_chars + 1

            if current_batch:
                new_json.append({"text": "\n".join(current_batch)})

            # 3. Guardamos los nuevos límites para este dataset específico
            new_raw_datasets[name] = {
                "start": start_idx, 
                "end": len(new_json),
                "anonymous": bounds.get("anonymous", self.do_anonymize)
            }

        # 4. Actualizamos la instancia con los datos agrupados
        self.json = new_json
        self.raw_datasets = new_raw_datasets

        print(f"✅ Concatenación finalizada respetando orígenes.")
        self.summary() # Para ver cómo han quedado los nuevos tamaños
        return self

    def startTatoeba(self):
        print(f"Descargando tatoeba para {self.language}:")
        try:
            self.read_tatoeba_url(
                f"https://downloads.tatoeba.org/exports/per_language/"
                f"{self.language_codes['tatoeba']}/"
                f"{self.language_codes['tatoeba']}_sentences_detailed.tsv.bz2"
            )
            print("Completado con éxito")
        except Exception as e:
            print("No se pudo completar por:", e)

        return self

    def startTxTLocal(self):
        print(f"Cargando txt locales para {self.language}:")
        self.read_folder(f"datasets/{self.language}")

    def read_tatoeba_url(self, url):
        df = pd.read_csv(
            url,
            sep="\t",
            compression="bz2",
            header=None,
            names=["id", "lang", "text", "author", "created_at", "updated_at"]
        )
        if df.iloc[0]["lang"] != self.language_codes["tatoeba"]:
            raise ValueError("El dataset descargado no corresponde al idioma esperado")
        elif "tatoeba" in self.raw_datasets:
            raise ValueError("El dataset tatoeba ya está cargado")

        json_data = self.pandas_to_json(df)
        start = len(self.json)
        self.json += json_data
        end = len(self.json)
        self._saveIndexer("tatoeba", start, end)
        return df

    def read_folder(self, directory):
        for file in os.listdir(directory):
            if file.endswith(".txt"):
                try:
                    self.read_local_file(directory, file)
                    print(f"\rArchivo {file} cargado")
                except Exception as e:
                    print(f"\rFallo al cargar el archivo {file}: {e}")

    def read_local_file(self, directory, file):
        dataset_name = file.split(".")[0]
        if dataset_name in self.raw_datasets.keys():
            raise ValueError(f"El dataset {directory}/{file} ya está cargado")
        extension = file.split(".")[-1]
        print(f"Leyendo archivo {file}...")
        if extension=="csv":
            try:
                json_data = self.pandas_to_json(pd.read_csv(os.path.join(directory, file)))
            except Exception as e:
                raise ValueError("It has to have a text column.\n","Error: ",e)
        elif extension == "json":
            json_data = self.pandas_to_json(pd.read_json(os.path.join(directory, file)))
        elif extension == "txt":
            json_data = []
            with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
                for line in f:
                    clean_line = self.clean_text(line)
                    if clean_line:
                        json_data.append({"text": clean_line})
        else:
            raise ValueError("La extensión no se reconoce (txt, json, csv)")

        start = len(self.json)
        self.json += json_data
        end = len(self.json)
        self._saveIndexer(dataset_name, start, end)
        return self
    

    def read_dataframe(self, df, dataset_name=None):
        if not dataset_name: # Hash del dataframe como nombre
            dataset_name = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values).hexdigest()
        if dataset_name in self.raw_datasets.keys():
            raise ValueError(f"El dataset {dataset_name} ya está cargado")
        print("Preparando líneas")
        try:
            json_data = self.pandas_to_json(df)
        except Exception as e:
            raise ValueError("It has to have a text column.\n","Error: ",e)
        print("Dataset preparado")
        start = len(self.json)
        self.json += json_data
        end = len(self.json)
        self._saveIndexer(dataset_name, start, end)
        return self

    def read_list(self, list_of_strings, dataset_name=None):
        """
        Carga una lista de strings como dataset.
        Si no se pasa dataset_name, se genera un hash único.
        """
        if dataset_name is None:
            joined = "\n".join(list_of_strings[:20])
            dataset_name = hashlib.sha256(joined.encode("utf-8")).hexdigest()

        if dataset_name in self.raw_datasets:
            raise ValueError(f"El dataset '{dataset_name}' ya está cargado")

        json_data = []
        for line in list_of_strings:
            clean = self.clean_text(line)
            if clean:
                json_data.append({"text": clean})

        start = len(self.json)
        self.json += json_data
        end = len(self.json)

        self._saveIndexer(dataset_name, start, end)
        return self

    def read_opus(self, source="NLLB", version=1):
        """
        Descarga y carga un corpus OPUS monolingüe en RAM.
        source puede ser: 'NLLB', 'OpenSubtitles', 'JW300', etc.
        version según el dataset que querais descargar (en OPUS)
        """
        url = f"https://object.pouta.csc.fi/OPUS-{source}/v{version}/mono/{self.language_codes['opus']}.txt.gz"

        dataset_name = f"opus_{source}_{version}"
        if dataset_name in self.raw_datasets:
            raise ValueError(f"El dataset '{dataset_name}' ya está cargado")

        # Descargar en RAM
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError(f"No se pudo descargar OPUS desde {url}")
        print("Empezando descarga")
        gz_bytes = io.BytesIO(r.content)

        # Descomprimir en RAM
        with gzip.GzipFile(fileobj=gz_bytes, mode="rb") as f:
            text = f.read().decode("utf-8")
        print("Descarga completada. \nProcesando las líneas")
        # Procesar líneas
        json_data = []
        for line in text.split("\n"):
            clean = self.clean_text(line)
            if clean:
                json_data.append({"text": clean})

        # Registrar dataset
        start = len(self.json)
        self.json += json_data
        end = len(self.json)
        print("Dataset cargado")
        self._saveIndexer(dataset_name, start, end)
        return self

    # def is_mostly_latin(self, text, threshold=0.8):
    #     letters = regex.findall(r"\p{L}", text)
    #     latin = regex.findall(r"\p{Latin}", text)
    #     return len(letters) > 0 and len(latin) / len(letters) >= threshold


    def clean_text(self, text):
        
        # 1. Remove hashtags and the word following them
        text = re.sub(r"#\S+", "", text)

        # 2. Remove HTML tags
        text = re.sub(r"<.*?>", " ", text)

        # 3. Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # 4. Decode HTML entities
        text = html.unescape(text)

        # 5. Remove emojis / non-ASCII symbols (pero mantener letras latinas con tildes y ñ)
        text = regex.sub(r"\p{Emoji}+", " ", text)

        # 6. Replace non-letter/number characters except whitespace/newline and basic punctuation y apostrofes
        text = unicodedata.normalize("NFC", text) # Normalize to make all ´ equal.
        text = regex.sub(r"[^\p{L}\p{N}\s\n!?.,;:'’-]", " ", text)


        # 7. Normalize spaces/tabs
        text = re.sub(r"[ \t]+", " ", text)

        # 8. Limit character repetitions (más de 5 → 5)
        text = re.sub(r"(.)\1{5,}", r"\1"*5, text)

        # 9. Limit word repetitions (más de 5 → 5)
        def limit_word_reps(match):
            word = match.group(1)
            return " ".join([word]*5)
        text = re.sub(r"\b(\w+)( \1){5,}\b", limit_word_reps, text)

        # 10. Lowercase and strip
        text = text.strip()#.lower()


        # 11. Filtrar basura: líneas demasiado cortas, repetitivas o sin sentido 
        words = text.split()
        if len(words) <= self.MIN_WORDS or len(words) >= self.MAX_WORDS:
            return None
        if words and all(w == words[0] for w in words):
            return None
        # if not self.is_mostly_latin(text, 0.75):
        #     return None
        if any(len(w) > self.MAX_WORD_LEN for w in words):
            return None

        one_letter_count = sum(1 for w in words if len(w) == 1)
        if one_letter_count >= len(words) * 0.5:
            return None
        
        # Una palabra domina la frase (Muy pocos casos, no merece la pena)
        # counts = Counter(words)
        # _ , freq = counts.most_common(1)[0]
        # if freq / len(words) > 0.35:  
        #     return None

        if self.do_anonymize:
            return self.anonymize(text)
        else:
            return text


    def anonymize(self, text):
        """
        Replace sensitive info like emails, phone numbers, and names with placeholders.
        """
        # Mask emails
        text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", text)

        # Mask phone numbers (basic patterns)
        text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", text)

        return text


    def pandas_to_json(self, df, clean=True, save:str=False):
        json_data = []
        for t in df["text"].tolist():
            clean_line = self.clean_text(t) if clean else t
            if clean_line:
                json_data.append({"text": clean_line})
        if save:
            with open(save, "w", encoding="utf-8") as f:
                f.write(json.dumps(json_data, ensure_ascii=False))
        return json_data
    
    def save(self, path):
        data = {
            "language": self.language,
            "language_codes": self.language_codes,
            "MIN_WORDS": self.MIN_WORDS,
            "MAX_WORDS": self.MAX_WORDS,
            "MAX_WORD_LEN": self.MAX_WORD_LEN,
            "do_anonymize": self.do_anonymize,
            "raw_datasets": self.raw_datasets,
            "ft_model": self.ft_model,
            "filter_language_thr": self.filter_language_thr,
            "json": self.json
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Dataset guardado en {path}")

    @classmethod
    def from_json(cls, path):
        """
        Carga un archivo JSON guardado previamente y devuelve 
        una instancia completa de LanguageDataset.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Usamos __new__ para evitar que el __init__ pida 'language' obligatoriamente
        instance = cls.__new__(cls)

        # Mapeamos los datos del JSON directamente a la instancia
        instance.language = data["language"]
        instance.language_codes = data["language_codes"]
        instance.MIN_WORDS = data["MIN_WORDS"]
        instance.MAX_WORDS = data["MAX_WORDS"]
        instance.MAX_WORD_LEN = data["MAX_WORD_LEN"]        
        instance.do_anonymize = data["do_anonymize"]
        instance.raw_datasets = data["raw_datasets"]
        instance.ft_model = data["ft_model"] 
        instance.filter_language_thr = data["filter_language_thr"] 
        instance.json = data["json"]

        print(f"Dataset cargado desde {path} ({len(instance.json)} líneas)")
        return instance
    
    @classmethod
    def from_hf_dataset(cls, hf_ds, language, **kwargs):
        """
        Crea una instancia de LanguageDataset a partir de un Dataset de HuggingFace.
        """
        # Creamos la instancia (esto inicializa atributos base)
        instance = cls(language=language, **kwargs)
        
        # Convertimos el Dataset de HF a nuestra estructura de lista de dicts
        # Asumimos que el dataset tiene una columna llamada 'text'
        instance.json = hf_ds.to_list()
        
        # Registramos este bloque en raw_datasets para mantener la coherencia
        dataset_name = f"hf_imported_{hashlib.md5(language.encode()).hexdigest()[:6]}"
        instance._saveIndexer(dataset_name, 0, len(instance.json))
        
        print(f"[INFO] Dataset importado desde HuggingFace: {len(instance.json)} líneas.")
        return instance

    def _summary(self):
        total = len(self.json)
        lines = []
        lines.append(f"\nResumen del dataset para '{self.language}':")
        lines.append(f"Total de líneas: {total}\n")

        # --- Resumen por dataset ---
        for name, bounds in self.raw_datasets.items():
            count = bounds["end"] - bounds["start"]
            pct = (count / total) * 100 if total > 0 else 0
            lines.append(f"- {name}: {count} líneas ({pct:.2f}%)")

        # --- Parámetros internos ---
        lines.append("\nParámetros internos:")
        lines.append(f"  MIN_WORDS: {self.MIN_WORDS}")
        lines.append(f"  MAX_WORDS: {self.MAX_WORDS}")
        lines.append(f"  MAX_WORD_LEN: {self.MAX_WORD_LEN}")
        lines.append(f"  Se anonimiza el texto: {self.do_anonymize}")

        # --- Códigos de idioma ---
        lines.append("\nCódigos de idioma:")
        lines.append(f"  {self.language_codes}")

        # --- Ejemplos globales ---
        lines.append("\nEjemplos del dataset (primeras 3 líneas):")
        for ex in self.json[:3]:
            lines.append(f"  • {ex['text']}")

        # --- Ejemplos por dataset ---
        lines.append("\nEjemplo por dataset:")
        for name, bounds in self.raw_datasets.items():
            start = bounds["start"]
            if start < len(self.json):
                example = self.json[start]["text"]
                lines.append(f"  [{name}] → {example}")

        # --- Lista de datasets cargados ---
        lines.append("\nDatasets cargados:")
        lines.append(f"(Dataset, está anonimizado): {[(i, self.raw_datasets[i]['anonymous']) for i in self.raw_datasets.keys()]}")

        return "\n".join(lines)



    def summary(self):
        """
        Muestra un resumen completo del dataset:
        - número de líneas por dataset
        - porcentaje del total
        - parámetros internos
        - códigos de idioma
        """
        print(self._summary())

    def __str__(self):
        return self._summary()

    def get_stats(self, tokenizer, N_max=20000, do_print=True):
        """
        Calcula estadísticas de tokens del dataset actual.
        """
        if not self.json:
            print("[Vacio] No hay datos para analizar.")
            return None

        # 1. Seleccionar subconjunto para no eternizarnos si el dataset es gigante
        n_to_analyze = min(N_max, len(self.json))
        subset_texts = [self.json[i]["text"] for i in range(n_to_analyze)]

        if do_print:
            print(f"\n📊 Analizando estadísticas de tokens (N={n_to_analyze})...")

        # 2. Tokenización en batch (esto es mucho más rápido que uno a uno)
        # Usamos fast tokenizer si está disponible
        tokenized = tokenizer(subset_texts, truncation=False, padding=False, add_special_tokens=False)
        lengths = np.array([len(ids) for ids in tokenized["input_ids"]])

        # 3. Cálculos
        mean_val = lengths.mean()
        median_val = np.median(lengths)
        p95 = np.percentile(lengths, 95)
        p98 = np.percentile(lengths, 98)

        if do_print:
            print("------------------------------------------------")
            print(f"Total líneas en dataset: {len(self.json)}")
            print(f"Media: {mean_val:.2f} | Mediana: {median_val:.2f}")
            print(f"Percentil 95: {p95:.2f} (recomendado para max_length)")
            print(f"Percentil 98: {p98:.2f}")
            print(f"Máximo: {lengths.max()} | Mínimo: {lengths.min()}")
        
        # Moda aproximada
        from collections import Counter
        bins = Counter((lengths // 10) * 10)
        moda = bins.most_common(1)[0][0]
        if do_print:
            print(f"Moda (aprox): {moda} tokens")
            print("------------------------------------------------\n")

        return lengths