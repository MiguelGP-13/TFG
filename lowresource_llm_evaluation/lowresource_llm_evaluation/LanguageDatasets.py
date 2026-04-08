import os, re, json, html, regex
import pandas as pd
from datasets import Dataset
import numpy as np
import unicodedata
import hashlib
import requests
import gzip
import io

lang_code = {
    "asturiano": {
        "tatoeba": "ast",
        "opus": "ast"
    },
    "aranes": {
        "tatoeba": "oci",
        "opus": "oc"
    },
    "aragones": {
        "tatoeba": "arg",
        "opus": "an"
    },
    "gallego" : {
        "tatoeba" : "glg",
        "opus" : "gl"
    }
}

class LanguageDataset():
    def __init__(self, language=None, path=None,
             initializeTatoeba=False, initializeLocal=False,
             min_words=4, max_words=np.inf, max_word_len=25):

        # --- Caso 1: cargar desde archivo ---
        if path is not None:
            if language is not None:
                raise ValueError("No puede pasar 'language' y 'path' a la vez.")
            self.load_dataset(path)

        # --- Caso 2: inicialización normal ---
        elif language is None:
            raise ValueError("Debe pasar 'language' o 'path' obligatoriamente.")

        if language not in ["gallego", "asturiano", "aranes"]:
            raise KeyError('Lenguaje no contemplado (["gallego", "asturiano", "aranes"])\n Si quiere crearlo desde un dataset previamente guardado, necesita poner path="path_to_your_dataset"')

        self.raw_datasets = {}
        self.language = language
        self.language_codes = lang_code[language]
        self.json = []
        self.MIN_WORDS = min_words
        self.MAX_WORDS = max_words
        self.MAX_WORD_LEN = max_word_len

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

    def tokenize(self, tokenizer, max_length=512):
        """
        Aplica un tokenizer externo al dataset.
        Devuelve un HuggingFace Dataset tokenizado listo para entrenamiento.
        """
            # Aseguramos que el tokenizer tenga pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        def _tokenize(example):
            result = tokenizer(
                example["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result

        return self.hf_dataset.map(_tokenize)

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

    def startTLocal(self):
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
        self.raw_datasets["tatoeba"] = {"start": start, "end": end}
        return df

    def read_folder(self, directory):
        for file in os.listdir(directory):
            if file.endswith(".txt"):
                print(f"Leyendo archivo {file}...")
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
        self.raw_datasets[dataset_name] = {"start": start, "end": end}
        return json_data
    

    def read_dataframe(self, df, dataset_name=None):
        if not dataset_name: # Hash del dataframe como nombre
            dataset_name = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values).hexdigest()
        if dataset_name in self.raw_datasets.keys():
            raise ValueError(f"El dataset {dataset_name} ya está cargado")
        try:
            json_data = self.pandas_to_json(df)
        except Exception as e:
            raise ValueError("It has to have a text column.\n","Error: ",e)

        start = len(self.json)
        self.json += json_data
        end = len(self.json)
        self.raw_datasets[dataset_name] = {"start": start, "end": end}
        return json_data

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

        self.raw_datasets[dataset_name] = {"start": start, "end": end}
        return json_data

    def read_opus(self, source="NLLB", version=1):
        """
        Descarga y carga un corpus OPUS monolingüe en RAM.
        source puede ser: 'NLLB', 'OpenSubtitles', 'JW300', etc.
        version según el dataset que querais descargar (en OPUS)
        """
        source = source.lower()
        url = f"https://object.pouta.csc.fi/OPUS-{source}/v{version}/mono/{self.language_codes['opus']}.txt.gz"
        dataset_name = f"opus_{source}_{version}"
        if dataset_name in self.raw_datasets:
            raise ValueError(f"El dataset '{dataset_name}' ya está cargado")

        # Descargar en RAM
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError(f"No se pudo descargar OPUS desde {url}")

        gz_bytes = io.BytesIO(r.content)

        # Descomprimir en RAM
        with gzip.GzipFile(fileobj=gz_bytes, mode="rb") as f:
            text = f.read().decode("utf-8")

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

        self.raw_datasets[dataset_name] = {"start": start, "end": end}
        return json_data

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
            "raw_datasets": self.raw_datasets,
            "json": self.json
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Dataset guardado en {path}")

    def load_dataset(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.language = data["language"]
        self.language_codes = data["language_codes"]
        self.MIN_WORDS = data["MIN_WORDS"]
        self.MAX_WORDS = data["MAX_WORDS"]
        self.MAX_WORD_LEN = data["MAX_WORD_LEN"]

        self.raw_datasets = data["raw_datasets"]
        self.json = data["json"]

        print(f"Dataset cargado desde {path} ({len(self.json)} líneas)")

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
        lines.append(f"  {list(self.raw_datasets.keys())}")

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