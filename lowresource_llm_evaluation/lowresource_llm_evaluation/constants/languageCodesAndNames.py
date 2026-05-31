# Diccionario de nombres de idiomas por idioma origen
LANG_NAMES = {
    "es": {"es": "español", "fr": "francés", "en": "inglés", "gl": "gallego", "ast": "asturiano", "aran": "aranés", "pt": "portugués", "it": "italiano", "de": "alemán"},
    "fr": {"es": "espagnol", "fr": "français", "en": "anglais", "gl": "galicien", "ast": "asturien", "aran": "aranés", "pt": "portugais", "it": "italien", "de": "allemand"},
    "en": {"es": "spanish", "fr": "french", "en": "english", "gl": "galician", "ast": "asturian", "aran": "aranese", "pt": "portuguese", "it": "italian", "de": "german"},
    "gl": {"es": "español", "fr": "francés", "en": "inglés", "gl": "galego", "ast": "asturiano", "aran": "aranés", "pt": "portugués", "it": "italiano", "de": "alemán"},
    "ast": {"es": "español", "fr": "francés", "en": "inglés", "gl": "gallego", "ast": "asturiano", "aran": "aranés", "pt": "portugués", "it": "italiano", "de": "alemán"},
    "aran": {"es": "espanhòu", "fr": "francés", "en": "anglés", "gl": "galèc", "ast": "asturianu", "aran": "aranés", "pt": "portugués", "it": "italian", "de": "alemand"}
}

LANG_CODES = {
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