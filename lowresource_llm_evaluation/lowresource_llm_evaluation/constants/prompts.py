from types import SimpleNamespace
from.languageCodesAndNames import LANG_NAMES

GenerateInstructivePrompts = SimpleNamespace(
TEMPLATES = {
    "asturiano": [
        # ——— ALARGAR / AMPLIAR ———
        "Amplía esti testu añadiendo detalles, exemplos y esplicaciones:",
        "Desarrolla esti conteníu con una versión más llarga y completa:",
        "Esplica esti testu con más fondura y razonamientu:",
        "Amplía esti fragmentu como si fuera pa un adultu interesáu nel tema:",
        "Da una interpretación detallada y razonada d'esti conteníu:",
        "Crea una versión más estensa d'esti testu, añadiendo matices:",
        "Inventa un cuentu más llargu inspiráu nesti fragmentu:",
        "Crea un diálogu más desarrolláu basáu nesti conteníu:",
        "Da exemplos prácticos y amplía la información d'esti testu:",
        "Elabora una esplicación más completa d'esti conteníu:",

        # ——— VARIEDAD / REFORMULAR ———
        "Reformula esti testu con otres pallabres:",
        "Resume esti testu n'asturiano:",
        "Cambia esti testu a un tonu más formal:",
        "Cambia esti testu a un tonu más informal:"
    ],

    "gallego": [
        # ——— ALARGAR / AMPLIAR ———
        "Amplía este texto engadindo detalles, exemplos e explicacións:",
        "Desenvolve este contido cunha versión máis longa e completa:",
        "Explica este texto con máis profundidade e razoamento:",
        "Amplía este fragmento como se fose para un adulto interesado no tema:",
        "Dá unha interpretación detallada e razoada deste contido:",
        "Crea unha versión máis extensa deste texto, engadindo matices:",
        "Inventa un conto máis longo inspirado neste fragmento:",
        "Crea un diálogo máis desenvolvido baseado neste contido:",
        "Dá exemplos prácticos e amplía a información deste texto:",
        "Elabora unha explicación máis completa deste contido:",

        # ——— VARIEDAD / REFORMULAR ———
        "Reformula este texto con outras palabras:",
        "Resume este texto en galego:",
        "Cambia este texto a un ton máis formal:",
        "Cambia este texto a un ton máis informal:"
    ],

    "aranes": [
        # ——— ALARGAR / AMPLIAR ———
        "Amplie aguest tèxte en tot includir detalhs, exemples e explicacions:",
        "Desvolòpe aguest contengut damb ua version mès longa e completa:",
        "Explique aguest tèxte damb mès prigondor e rasonament:",
        "Amplie aguest fragment coma entà un adult interessat en eth tèma:",
        "Done ua interpretacion detalhada e rasonada d'aguest contengut:",
        "Cree ua version mès estensa d'aguest tèxte, includint matices:",
        "Invente un raconte mès long inspirat en aguest fragment:",
        "Cree un dialòg mès desvolopat basat en aguest contengut:",
        "Done exemples practics e amplie era informacion d'aguest tèxte:",
        "Elabòre ua explicacion mès completa d'aguest contengut:",

        # ——— VARIEDAD / REFORMULAR ———
        "Reformule aguest tèxte damb d'autes paraules:",
        "Resumís aguest tèxte en aranés:",
        "Càmbia aguest tèxte a un ton mès formau:",
        "Càmbia aguest tèxte a un ton mès informau:"
    ]
},

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
)

def buildStoryPrompt(lang):
    prompts = {
        "es": (
            "Eres un modelo que solo puede hablar en español. "
            "Genera una historia original, coherente, completa y corta en español. "
            "No utilices ningún otro idioma.\nHistoria:"
        ),
        "ast": (
            "Tu yes un modelu que namás pue falar n'asturianu. "
            "Xenera una hestoria orixinal, coherente, completa y curtia n'asturianu. "
            "Nun uses nengún otru idioma.\nHestoria:"
        ),
        "gl": (
            "Es un modelo que só pode falar en galego. "
            "Xera unha historia orixinal, coherente, completa e curta en galego. "
            "Non empregues ningún outro idioma.\nHistoria:"
        ),
        "aran": (
            "Es un modèl que pòt parlar sonque en aranés. "
            "Genèra ua istòria originau, coerenta, completa e braca en aranés. "
            "Non emplegues cap d’auti idiòmas.\nIstòria:"
        ),
        "fr": (
            "Tu es un modèle qui ne peut parler qu’en français. "
            "Génère une histoire originale, cohérente, complète et courte en français. "
            "N’utilise aucune autre langue.\nHistoire:"
        ),
    }
    return prompts[lang]


def buildCorrectionPrompt(text, lang):
    prompts = {
        "es": (
            f"Corrige esta frase en español. "
            f"No añadas contenido nuevo. "
            f"Devuelve solo la frase corregida:\n{text}\nCorrección:"
        ),
        "ast": (
            f"Corrige esta frase n'asturianu. "
            f"Nun añadas conteníu nuevu. "
            f"Devuelve namás la frase correxida:\n{text}\nCorreición:"
        ),
        "gl": (
            f"Corrige esta frase en galego. "
            f"Non engadas contido novo. "
            f"Devolve só a frase corrixida:\n{text}\nCorrección:"
        ),
        "aran": (
            f"Corrigís aguesta frasa en aranés. "
            f"Non híges contengut nau. "
            f"Da sonque era frasa corregida:\n{text}\nCorreccion:"
        ),
        "fr": (
            f"Corrige cette phrase en français. "
            f"N’ajoute aucun contenu. "
            f"Donne seulement la phrase corrigée:\n{text}\nCorrection:"
        ),
    }

    return prompts[lang]



def buildTranslationPrompt(text, target_lang, source_lang):
    """
    Devuelve un prompt en el idioma adecuado según source_lang,
    usando nombres de idiomas adaptados a cada lengua.
    """

    

    if source_lang not in LANG_NAMES:
        raise ValueError(f"Idioma no soportado: {source_lang}. Elija uno de [{LANG_NAMES.keys()}]")

    if target_lang not in LANG_NAMES[source_lang]:
        raise ValueError(f"Idioma destino non reconocido: {target_lang}. Elija uno de [{LANG_NAMES.keys()}]")

    # Nombre del idioma destino adaptado al idioma origen
    target_name = LANG_NAMES[source_lang][target_lang]

    prompts = {
        "es": (
            f"Traduce al {target_name} el siguiente texto y responde únicamente con la frase traducida, sin añadir nada más:\n\n{text}\n\nTraducción:"
        ),

        "fr": (
            f"Traduisez en {target_name} le texte suivant et répondez uniquement avec la phrase traduite, sans rien ajouter:\n\n{text}\n\nTraduction:"
        ),

        "ast": (
            f"Traduce al {target_name} esti testu y respuende namás cola frase traducida, ensin amestar nada más:\n\n{text}\n\nTraducción:"
        ),

        "aran": (
            f"Tradusís eth tèxte seguent ara lengua {target_name} e respòn sonque damb era frasa tradusida, sense híger cap aute tèxte:\n\n{text}\n\nTraduccion:"
        ),

        "gl": (
            f"Traduce ao {target_name} o seguinte texto e responde unicamente coa frase traducida, sen engadir nada máis:\n\n{text}\n\nTradución:"
        )
    }

    return prompts[source_lang]


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

