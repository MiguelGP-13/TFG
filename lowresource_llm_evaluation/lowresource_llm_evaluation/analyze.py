import json
from pathlib import Path
from tabulate import tabulate
import html, re

# Colores ANSI
RESET = "\033[0m"
BOLD = "\033[1m"

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"

BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"

def color_masked(sentence):
    """Resalta <mask> en amarillo."""
    return sentence.replace("<mask>", f"{BG_YELLOW}{BOLD}<mask>{RESET}")

def color_ort_errors(annotated):
    """
    Convierte:
        <err t=ort>he</err>
    en:
        [he] coloreado según tipo
    """
    import re

    def repl(match):
        tipo = match.group(1)
        palabra = match.group(2)

        if tipo == "ort":
            col = RED
        elif tipo == "reord":
            col = BLUE
        elif tipo == "add":
            col = GREEN
        else:
            col = MAGENTA

        return f"{col}[{palabra}]{RESET}"

    return re.sub(r"<err t=(.*?)>(.*?)</err>", repl, annotated)


def _short(text, n=180):
    if text is None:
        return f"{RED}None{RESET}"
    if not isinstance(text, str):
        return text
    return text if len(text) <= n else text[:n] + "..."

def load_results(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def pretty_print_results(data):
    print(f"\n{BOLD}{CYAN}==============================")
    print("      RESULTADOS DEL MODELO")
    print("==============================\n" + RESET)

    for task, contenido in data.items():
        print(f"\n\n{BOLD}{MAGENTA}########################################")
        print(f"### TAREA: {task.upper()}")
        print("########################################\n" + RESET)

        media = contenido["media"]
        ejemplos = contenido["ejemplo"]  

        # -------------------------
        # MÉTRICAS GLOBALES
        # -------------------------
        print(f"{BOLD}▶ MÉTRICAS GLOBALES{RESET}\n")
        table = [[k, v] for k, v in media.items()]
        print(tabulate(table, headers=["Métrica", "Valor"], floatfmt=".4f"))
        print("\n")

        # -------------------------
        # EJEMPLOS
        # -------------------------
        print(f"{BOLD}▶ EJEMPLOS{RESET}\n")

        # Si no es lista, lo convertimos en lista
        if not isinstance(ejemplos, list):
            ejemplos = [ejemplos]

        for idx, ej in enumerate(ejemplos):
            print(f"\n{BOLD}--- Ejemplo {idx+1} ---{RESET}")

            # ============================
            # TRADUCCIÓN
            # ============================
            if task == "traduccion":
                print(f"{CYAN}source:{RESET}     {_short(ej['source'])}")
                print(f"{CYAN}reference:{RESET}  {_short(ej['reference'])}")
                print(f"{CYAN}translated:{RESET} {_short(ej['translated'])}")
                print(f"{YELLOW}BLEU:{RESET} {ej['BLEU']:.4f}   {YELLOW}chrF:{RESET} {ej['chrF']:.4f}")

            # ============================
            # ROUND TRIP
            # ============================
            elif task == "round_trip":
                print(f"{CYAN}intermediate_language:{RESET} {ej['intermediate_language']}")
                print(f"{CYAN}source:{RESET}       {_short(ej['source'])}")
                print(f"{CYAN}intermediate:{RESET} {_short(ej['intermediate'])}")
                print(f"{CYAN}return:{RESET}   {_short(ej['return'])}")
                print(f"{YELLOW}BLEU:{RESET} {ej['BLEU']:.4f}   {YELLOW}chrF:{RESET} {ej['chrF']:.4f}")

            # ============================
            # CALIDAD LENGUA
            # ============================
            elif task == "calidad_lengua":
                print(ej)
                print(f"{CYAN}text:{RESET} {_short(ej['text'])}")
                for m in ["ttr", "entropy", "ngram_overlap", "freq_target", "freq_comparison", "calidad"]:
                    print(f"{YELLOW}{m}:{RESET} {ej[m]}")

            # ============================
            # VOCABULARIO
            # ============================
            elif task == "vocabulario":
                print(f"{CYAN}original:{RESET} {_short(ej['original'])}")
                print(f"{CYAN}masked_sentence:{RESET} {color_masked(_short(ej['masked_sentence']))}")
                print(f"{CYAN}missing_word:{RESET} {GREEN}{ej['missing_word']}{RESET}")

                r = ej["resultado"]
                print(f"{BOLD}resultado:{RESET}")
                print(f"  predicted: {_short(r['predicted'])}")
                print(f"  accuracy: {r['accuracy']}")
                print(f"  accuracy_lower: {r['accuracy_lower']}")
                print(f"  levenshtein: {r['levenshtein']}")

            # ============================
            # ORTOGRAFÍA
            # ============================
            elif task == "ortografia":
                print(f"{CYAN}original:{RESET} {_short(ej['original'])}")
                print(f"{CYAN}annotated:{RESET} {color_ort_errors(_short(ej['annotated']))}")
                print(f"{CYAN}n_errors:{RESET} {ej['n_errors']}")

                r = ej["resultado"]
                print(f"{BOLD}resultado:{RESET}")
                print(f"  incorrect: {_short(r['incorrect'])}")
                print(f"  corrected: {_short(r['corrected'])}")
                print(f"  BLEU: {r['BLEU']:.4f}   chrF: {r['chrF']:.4f}")
                print(f"  Levenshtein: {r['Levenshtein']}")
                print(f"  errores_totales: {r['errores_totales']}")
                print(f"  errores_corregidos: {r['errores_corregidos']}")
                print(f"  errores_no_corregidos: {r['errores_no_corregidos']}")
                print(f"  errores_nuevos: {r['errores_nuevos']}")
                print(f"  precision: {r['precision']:.4f}")
                print(f"  recall: {r['recall']:.4f}")
                print(f"  F1: {r['F1']:.4f}")

                print("  errores_detalle:")
                for tipo, palabra in r["errores_detalle"]:
                    col = RED if tipo == "ort" else BLUE if tipo == "reord" else GREEN
                    print(f"    - {col}{tipo}{RESET}: {palabra}")

            print("---------------------------")

# === Helpers ===
def esc(x):
    return html.escape(str(x))

def highlight_mask(text):
    return text.replace(
        "<mask>",
        "<span style='background:gold;font-weight:bold;text-decoration:underline'>&lt;mask&gt;</span>"
    )

def highlight_err(text):
    def repl(match):
        tipo = match.group(1)
        palabra = match.group(2)
        color = {"ort": "red", "reord": "blue", "add": "green"}.get(tipo, "purple")
        return f"<span style='color:{color};font-weight:bold;text-decoration:underline'>[{palabra}]</span>"
    return re.sub(r"<err t=(.*?)>(.*?)</err>", repl, text)

def generate_html_report_colored(models_dict, output_path="comparacion.html"):
    # === HTML HEADER ===
    html_out = """
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        body { font-family: Consolas, monospace; margin: 20px; background: #f7f7f7; }
        h1 { color: #333; }
        h2 { color: #663399; border-bottom: 2px solid #ccc; padding-bottom: 4px; }
        h3 { color: #444; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
        th, td { border: 1px solid #ccc; padding: 6px; }
        th { background: #eee; }
        .task-block { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 0 4px #ccc; }
        .example-block { border: 1px solid #aaa; padding: 10px; margin: 10px 0; background: #fafafa; border-radius: 6px; }
        .model-title { font-weight: bold; color: #663399; margin-top: 10px; }
        .cyan { color: #0099cc; font-weight: bold; }
        .yellow { color: #d4aa00; font-weight: bold; }
        .green { color: #009933; font-weight: bold; }
        .bold { font-weight: bold; }
    </style>
    </head>
    <body>
    <h1>Comparación de modelos</h1>
    """

    model_names = list(models_dict.keys())
    tasks = models_dict[model_names[0]].keys()

    for task in tasks:
        html_out += f"<div class='task-block'><h2>Tarea: {task.upper()}</h2>"

        # === LEYENDA DE MÉTRICAS ===
        html_out += """
        <div style='margin:10px 0; padding:12px; background:#eef7ff; border:1px solid #bcd7f0; border-radius:6px;'>
            <b style='font-size:14px;'>Interpretación de métricas:</b><br>
            <span style='color:#009933;font-weight:bold'>↑ Más alto es mejor:</span>
                BLEU, chrF, accuracy, recall, precision, F1, ttr, calidad<br>
            <span style='color:#cc3300;font-weight:bold'>↓ Más bajo es mejor:</span>
                Levenshtein, errores_totales, errores_nuevos, errores_no_corregidos,
                ngram_overlap (según tarea)
        </div>
        """

        # === MÉTRICAS ===
        html_out += "<h3>Métricas globales</h3><table><tr><th>Métrica</th>"
        for m in model_names:
            html_out += f"<th>{m}</th>"
        html_out += "</tr>"

        metrics = models_dict[model_names[0]][task]["media"].keys()
        for metric in metrics:
            html_out += f"<tr><td class='yellow'>{metric}</td>"
            for m in model_names:
                val = models_dict[m][task]["media"][metric]
                html_out += f"<td>{esc(val)}</td>"
            html_out += "</tr>"
        html_out += "</table>"

        # === EJEMPLOS ===
        html_out += "<h3>Ejemplos</h3>"
        # === LEYENDA DE COLORES ===
        if task == "ortografia":
            html_out += """
            <div style='margin:10px 0; padding:12px; background:#fff8dc; border:1px solid #e0d9b0; border-radius:6px;'>
                <b style='font-size:14px;'>Leyenda de colores:</b><br>
                <span style='color:red;font-weight:bold;text-decoration:underline'>[palabra]</span>
                    → error ortográfico (<b>ort</b>)<br>
                <span style='color:blue;font-weight:bold;text-decoration:underline'>[palabra]</span>
                    → error de reordenación (<b>reord</b>)<br>
                <span style='color:green;font-weight:bold;text-decoration:underline'>[palabra]</span>
                    → palabra añadida (<b>add</b>)
            </div>
            """

        ejemplos = {m: models_dict[m][task]["ejemplo"] for m in model_names}
        for m in model_names:
            if not isinstance(ejemplos[m], list):
                ejemplos[m] = [ejemplos[m]]

        n_ej = min(len(ejemplos[m]) for m in model_names)

        for i in range(n_ej):
            html_out += f"<div class='example-block'><h4>Ejemplo {i+1}</h4>"

            for m in model_names:
                ej = ejemplos[m][i]
                html_out += f"<div class='model-title'>{m}</div>"

                # === Render según tarea ===
                if task == "traduccion":
                    html_out += f"<span class='cyan'>source:</span> {esc(ej['source'])}<br>"
                    html_out += f"<span class='cyan'>reference:</span> {esc(ej['reference'])}<br>"
                    html_out += f"<span class='cyan'>translated:</span> {esc(ej['translated'])}<br>"

                elif task == "round_trip":
                    html_out += f"<span class='cyan'>intermediate_language:</span> {esc(ej['intermediate_language'])}<br>"
                    html_out += f"<span class='cyan'>source:</span> {esc(ej['source'])}<br>"
                    html_out += f"<span class='cyan'>intermediate:</span> {esc(ej['intermediate'])}<br>"
                    html_out += f"<span class='cyan'>return:</span> {esc(ej['return'])}<br>"

                elif task == "calidad_lengua":
                    html_out += f"<span class='cyan'>text:</span> {esc(ej['text'])}<br>"
                    for m2 in ["ttr", "entropy", "ngram_overlap", "freq_target", "freq_comparison", "calidad"]:
                        html_out += f"<span class='yellow'>{m2}:</span> {esc(ej[m2])}<br>"

                elif task == "vocabulario":
                    html_out += f"<span class='cyan'>masked_sentence:</span> {highlight_mask(ej['masked_sentence'])}<br>"
                    html_out += f"<span class='cyan'>missing_word:</span> <span class='green'>{esc(ej['missing_word'])}</span><br>"
                    r = ej["resultado"]
                    html_out += f"<span class='bold'>predicted:</span> {esc(r['predicted'])}<br>"

                elif task == "ortografia":
                    html_out += f"<span class='cyan'>original:</span> {esc(ej['original'])}<br>"
                    html_out += f"<span class='cyan'>incorrect:</span> {highlight_err(ej['annotated'])}<br>"
                    r = ej["resultado"]
                    html_out += f"<span class='bold'>corrected:</span> {esc(r['corrected'])}<br>"

            html_out += "</div>"

        html_out += "</div>"

    html_out += "</body></html>"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"HTML generado en: {output_path}")
