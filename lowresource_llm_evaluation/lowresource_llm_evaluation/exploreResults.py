import json
import html
import re
from pathlib import Path
import pandas as pd
from .constants import colours as c

# --- HELPERS DE TEXTO Y COLORES (TERMINAL) ---

def color_masked(sentence):
    """
    Resalta la etiqueta <maskGroup> o <mask> en el color amarillo de la terminal.
    
    Args:
        sentence (str): Cadena de texto que contiene la etiqueta.
    """
    return sentence.replace("<mask>", f"{c.BG_YELLOW}{c.BOLD}<mask>{c.RESET}")

def color_ort_errors(annotated):
    """
    Detecta etiquetas de error tipo <err t=ort>palabra</err> y las envuelve 
    entre corchetes con el color ANSI correspondiente al tipo de error para la consola.
    
    Args:
        annotated (str): Texto con anotaciones XML de errores.
    """
    color_map = {"ort": c.RED, "reord": c.BLUE, "add": c.GREEN}
    return re.sub(
        r"<err t=(.*?)>(.*?)</err>",
        lambda m: f"{color_map.get(m.group(1), c.MAGENTA)}[{m.group(2)}]{c.RESET}",
        annotated
    )

def _short(text, n=180):
    """Trunca de manera segura un texto si supera el límite de caracteres 'n'."""
    if text is None:
        return f"{c.RED}None{c.RESET}"
    if not isinstance(text, str):
        return text
    return text if len(text) <= n else text[:n] + "..."

# --- GESTIÓN DE ENTRADA / SALIDA ---

def load_results(path):
    """
    Carga y parsea un archivo JSON de resultados del Benchmark.
    
    Args:
        path (str|Path): Ruta del archivo a cargar.
    Returns:
        dict: Diccionario con la estructura de resultados por tarea.
    """
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)

def explorar_estructuras(resultados):
    """
    Imprime en consola un árbol visual con las tareas, métricas y sub-métricas 
    detectadas en el diccionario de resultados mapeado.
    
    Args:
        resultados (dict): Diccionario de datos cargado desde el JSON de evaluación.
    """
    print("=== MAPA DE TAREAS Y MÉTRICAS DISPONIBLES ===")
    for tarea, contenido in resultados.items():
        print(f"\nTarea: '{tarea}'")
        if "media" in contenido:
            for metrica, valor in contenido["media"].items():
                if isinstance(valor, dict):
                    print(f"    ├── '{metrica}' -> Requiere sub-métrica: {list(valor.keys())}")
                else:
                    print(f"    ├── '{metrica}'")
        else:
            print("No se encontró la clave 'media' en esta tarea.")
    print("\n" + "="*44)

# --- RENDERS DE VISUALIZACIÓN (CONSOLA, HTML, LATEX) ---

def pretty_print_results(data):
    """
    Muestra en la terminal un informe visualmente formateado con tablas de métricas 
    y bloques de ejemplos coloreados nativamente para cada tarea evaluada.
    
    Args:
        data (dict): Estructura de datos interna de resultados de un modelo.
    """
    from tabulate import tabulate
    print(f"\n{c.BOLD}{c.CYAN}==============================")
    print("      RESULTADOS DEL MODELO")
    print("==============================\n" + c.RESET)

    for task, contenido in data.items():
        print(f"\n\n{c.BOLD}{c.MAGENTA}########################################")
        print(f"### TAREA: {task.upper()}")
        print("########################################\n" + c.RESET)

        # Métricas Globales
        print(f"{c.BOLD}▶ MÉTRICAS GLOBALES{c.RESET}\n")
        table = [[k, v] for k, v in contenido["media"].items()]
        print(tabulate(table, headers=["Métrica", "Valor"], floatfmt=".4f"))
        print("\n")

        # Ejemplos
        print(f"{c.BOLD}▶ EJEMPLOS{c.RESET}\n")
        ejemplos = contenido["ejemplo"]  
        if not isinstance(ejemplos, list):
            ejemplos = [ejemplos]

        for idx, ej in enumerate(ejemplos):
            print(f"\n{c.BOLD}--- Ejemplo {idx+1} ---{c.RESET}")

            if task == "traduccion":
                print(f"{c.CYAN}source:{c.RESET}     {_short(ej['source'])}")
                print(f"{c.CYAN}reference:{c.RESET}  {_short(ej['reference'])}")
                print(f"{c.CYAN}translated:{c.RESET} {_short(ej['translated'])}")
                print(f"{c.YELLOW}BLEU:{c.RESET} {ej['BLEU']:.4f}   {c.YELLOW}chrF:{c.RESET} {ej['chrF']:.4f}")

            elif task == "round_trip":
                print(f"{c.CYAN}intermediate_language:{c.RESET} {ej['intermediate_language']}")
                print(f"{c.CYAN}source:{c.RESET}       {_short(ej['source'])}")
                print(f"{c.CYAN}intermediate:{c.RESET} {_short(ej['intermediate'])}")
                print(f"{c.CYAN}translated:{c.RESET}   {_short(ej['return'])}")
                print(f"{c.YELLOW}BLEU:{c.RESET} {ej['BLEU']:.4f}   {c.YELLOW}chrF:{c.RESET} {ej['chrF']:.4f}")

            elif task == "calidad_lengua":
                print(f"{c.CYAN}text:{c.RESET} {_short(ej['text'])}")
                for m in ["ttr", "entropy", "ngram_overlap", "freq_target", "freq_comparison", "calidad"]:
                    print(f"{c.YELLOW}{m}:{c.RESET} {ej[m]}")

            elif task == "vocabulario":
                print(f"{c.CYAN}original:{c.RESET} {_short(ej['original'])}")
                print(f"{c.CYAN}masked_sentence:{c.RESET} {color_masked(_short(ej['masked_sentence']))}")
                print(f"{c.CYAN}missing_word:{c.RESET} {c.GREEN}{ej['missing_word']}{c.RESET}")
                r = ej["resultado"]
                print(f"{c.BOLD}resultado:{c.RESET}\n   predicted: {_short(r['predicted'])}\n   accuracy: {r['accuracy']}\n   accuracy_lower: {r['accuracy_lower']}\n   levenshtein: {r['levenshtein']}")

            elif task == "ortografia":
                print(f"{c.CYAN}original:{c.RESET} {_short(ej['original'])}")
                print(f"{c.CYAN}annotated:{c.RESET} {color_ort_errors(_short(ej['annotated']))}")
                print(f"{c.CYAN}n_errors:{c.RESET} {ej['n_errors']}")
                r = ej["resultado"]
                print(f"{c.BOLD}resultado:{c.RESET}")
                print(f"   incorrect: {_short(r['incorrect'])}\n   corrected: {_short(r['corrected'])}")
                print(f"   BLEU: {r['BLEU']:.4f}   chrF: {r['chrF']:.4f}\n   Levenshtein: {r['Levenshtein']}")
                print(f"   errores_totales: {r['errores_totales']}\n   errores_corregidos: {r['errores_corregidos']}\n   errores_no_corregidos: {r['errores_no_corregidos']}\n   errores_nuevos: {r['errores_nuevos']}")
                print(f"   precision: {r['precision']:.4f}   recall: {r['recall']:.4f}   F1: {r['F1']:.4f}")
                print("   errores_detalle:")
                for tipo, palabra in r["errores_detalle"]:
                    col = c.RED if tipo == "ort" else c.BLUE if tipo == "reord" else c.GREEN
                    print(f"    - {col}{tipo}{c.RESET}: {palabra}")

            print("---------------------------")

def esc(x):
    return html.escape(str(x))

def highlight_mask(text):
    return text.replace("<mask>", "<span style='background:gold;font-weight:bold;text-decoration:underline'>&lt;mask&gt;</span>")

def highlight_err(text):
    color_map = {"ort": "red", "reord": "blue", "add": "green"}
    return re.sub(
        r"<err t=(.*?)>(.*?)</err>",
        lambda m: f"<span style='color:{color_map.get(m.group(1), 'purple')};font-weight:bold;text-decoration:underline'>[{m.group(2)}]</span>",
        text
    )

def generate_html_report(models_dict, output_path="comparacion.html"):
    """
    Construye un reporte interactivo y estructurado en HTML con hojas de estilo CSS inline,
    tablas cruzadas de métricas globales y bloques paralelos de ejemplos de inferencias.
    
    Args:
        models_dict (dict): Estructura {'NombreModelo': datos_modelo} compartida del benchmark.
        output_path (str): Destino de guardado del documento html resultante.
    """
    html_out = """<html><head><meta charset="utf-8"><style>
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
    </style></head><body><h1>Comparación de modelos</h1>"""

    model_names = list(models_dict.keys())
    tasks = models_dict[model_names[0]].keys()

    for task in tasks:
        html_out += f"<div class='task-block'><h2>Tarea: {task.upper()}</h2>"
        html_out += """<div style='margin:10px 0; padding:12px; background:#eef7ff; border:1px solid #bcd7f0; border-radius:6px;'>
            <b style='font-size:14px;'>Interpretación de métricas:</b><br>
            <span style='color:#009933;font-weight:bold'>↑ Más alto es mejor:</span> BLEU, chrF, accuracy, recall, precision, F1, ttr, calidad<br>
            <span style='color:#cc3300;font-weight:bold'>↓ Más bajo es mejor:</span> Levenshtein, errores_totales, errores_nuevos, errores_no_corregidos, ngram_overlap
        </div>"""

        # Métricas Cruzadas por Columnas de Modelos
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
        html_out += "</table><h3>Ejemplos</h3>"

        if task == "ortografia":
            html_out += """<div style='margin:10px 0; padding:12px; background:#fff8dc; border:1px solid #e0d9b0; border-radius:6px;'>
                <b style='font-size:14px;'>Leyenda de colores:</b><br>
                <span style='color:red;font-weight:bold;text-decoration:underline'>[palabra]</span> → error ortográfico (<b>ort</b>)<br>
                <span style='color:blue;font-weight:bold;text-decoration:underline'>[palabra]</span> → error de reordenación (<b>reord</b>)<br>
                <span style='color:green;font-weight:bold;text-decoration:underline'>[palabra]</span> → palabra añadida (<b>add</b>)
            </div>"""

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

                if task == "traduccion":
                    html_out += f"<span class='cyan'>source:</span> {esc(ej['source'])}<br><span class='cyan'>reference:</span> {esc(ej['reference'])}<br><span class='cyan'>translated:</span> {esc(ej['translated'])}<br>"
                elif task == "round_trip":
                    html_out += f"<span class='cyan'>intermediate_language:</span> {esc(ej['intermediate_language'])}<br><span class='cyan'>source:</span> {esc(ej['source'])}<br><span class='cyan'>intermediate:</span> {esc(ej['intermediate'])}<br><span class='cyan'>translated:</span> {esc(ej['return'])}<br>"
                elif task == "calidad_lengua":
                    html_out += f"<span class='cyan'>text:</span> {esc(ej['text'])}<br>"
                    for m2 in ["ttr", "entropy", "ngram_overlap", "freq_target", "freq_comparison", "calidad"]:
                        html_out += f"<span class='yellow'>{m2}:</span> {esc(ej[m2])}<br>"
                elif task == "vocabulario":
                    html_out += f"<span class='cyan'>masked_sentence:</span> {highlight_mask(ej['masked_sentence'])}<br><span class='cyan'>missing_word:</span> <span class='green'>{esc(ej['missing_word'])}</span><br><span class='bold'>predicted:</span> {esc(ej['resultado']['predicted'])}<br>"
                elif task == "ortografia":
                    html_out += f"<span class='cyan'>original:</span> {esc(ej['original'])}<br><span class='cyan'>incorrect:</span> {highlight_err(ej['annotated'])}<br><span class='bold'>corrected:</span> {esc(ej['resultado']['corrected'])}<br>"
            html_out += "</div>"
        html_out += "</div>"

    html_out += "</body></html>"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"HTML generado en: {output_path}")

# --- MÓDULO EXPORTADOR A REPORTE LATEX ---

def tex_escape(text):
    """Escapa de forma segura caracteres conflictivos reservados de LaTeX dentro de strings."""
    text = str(text)
    conv = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
        '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}', '<': r'{\textless}', '>': r'{\textgreater}',
        '\'': r'\textquotesingle{}'
    }
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key=lambda item: -len(item))))
    return regex.sub(lambda mo: conv[mo.group()], text)

def smart_truncate(text, limit=None, max_chars=300):
    text = str(text).replace('\n', ' ')
    eff_limit = limit if limit else max_chars
    if len(text) <= eff_limit:
        return text
    half = (eff_limit - 7) // 2
    return text[:half] + " [...] " + text[-half:]

def format_val(val, is_best=False, max_chars=300):
    if isinstance(val, (int, float)):
        formatted = f"{val:.2f}"
        return f"\\textbf{{\\underline{{{formatted}}}}}" if is_best else formatted
    if isinstance(val, dict):
        formatted_dict = {k: (round(v, 2) if isinstance(v, (int, float)) else v) for k, v in val.items()}
        val_str = str(formatted_dict)
        text_to_escape = smart_truncate(val_str, 80, max_chars) if len(val_str) > 100 else val_str
        return tex_escape(text_to_escape)
    return tex_escape(smart_truncate(str(val), 80, max_chars))

def append_latex_fields(latex_out, items, max_chars=300, bold_label=True):
    clean_items = [(l, v) for l, v in items if v is not None]
    for i, (label, value) in enumerate(clean_items):
        processed = tex_escape(smart_truncate(str(value), max_chars=max_chars))
        endline = r"\\" if i < len(clean_items) - 1 else ""
        lbl_esc = tex_escape(label)
        lbl_str = f"\\textbf{{{lbl_esc}:}}" if bold_label else f"{{{lbl_esc}:}}"
        latex_out.append(f"{lbl_str} {processed}{endline}")

def _build_metrics_list(base_media, compact):
    metrics_list = []
    for metric, val in base_media.items():
        if compact and metric == "freq_comparison" and isinstance(val, dict):
            for lang in val.keys():
                metrics_list.append((metric, lang))
        else:
            metrics_list.append((metric, None))
    return metrics_list

def _process_table_row(metric_info, model_names, models_dict, task, low_is_better):
    orig_metric, sub_metric = metric_info
    raw_values = []
    for m in model_names:
        m_val = models_dict[m][task]["media"].get(orig_metric)
        raw_values.append(m_val.get(sub_metric) if sub_metric and isinstance(m_val, dict) else m_val)
    
    best_val_rounded = None
    try:
        num_values = [v for v in raw_values if isinstance(v, (int, float))]
        if num_values:
            rounded_values = [round(v, 2) for v in num_values]
            if max(rounded_values) != min(rounded_values):
                best_val_rounded = min(rounded_values) if any(m in orig_metric.lower() for m in low_is_better) else max(rounded_values)
    except ValueError:
        pass

    formatted_cells = [format_val(v, is_best=(best_val_rounded is not None and isinstance(v, (int, float)) and round(v, 2) == best_val_rounded)) for v in raw_values]
    display_metric = f"{orig_metric}_{sub_metric}" if sub_metric else orig_metric
    return f"{tex_escape(display_metric)} & {' & '.join(formatted_cells)} \\\\"

def _extract_fields(task, ej):
    t_low = task.lower()
    if t_low == "traduccion":
        return [('source', ej.get('source')), ('reference', ej.get('reference'))], [('translated', ej.get('translated'))]
    elif t_low == "ortografia":
        return [('original', ej.get('original')), ('incorrect', ej.get('annotated'))], [('corrected', ej.get('resultado', {}).get('corrected'))]
    elif t_low == "vocabulario":
        return [('masked_sentence', ej.get('masked_sentence')), ('missing_word', ej.get('missing_word'))], [('predicted', ej.get('resultado', {}).get('predicted'))]
    elif t_low == "round_trip":
        return [('intermediate_language', ej.get('intermediate_language')), ('source', ej.get('source'))], [('intermediate', ej.get('intermediate')), ('translated', ej.get('return'))]
    elif "calidad" in t_low:
        fields = [(k, v) for k, v in ej.items() if k.lower() not in ['resultado', 'calidad']]
        if isinstance(ej.get('resultado'), dict):
            fields.extend([(k, v) for k, v in ej['resultado'].items() if k.lower() != 'calidad'])
        return [], fields
    else:
        common = [(k, v) for k, v in ej.items() if k != 'resultado']
        specific = [(k, v) for k, v in ej.get('resultado', {}).items()] if isinstance(ej.get('resultado'), dict) else []
        return common, specific

def generate_latex_results(models_dict, lengua, max_chars, output_path, compact=False):
    """
    Función núcleo encargada de procesar el JSON y estructurar secciones y tablas 
    bajo paquetes 'longtable' e inyecciones de fuentes tipográficas para LaTeX.
    """
    low_is_better = ['levenshtein', 'errores_totales', 'errores_nuevos', 'errores_no_corregidos', 'ngram_overlap']
    latex_out = [f"\\section{{{tex_escape(lengua)}}}"]
    
    model_names = list(models_dict.keys())
    tasks = models_dict[model_names[0]].keys()

    for task in tasks:
        latex_out.append(f"\n\\subsection{{Tarea: {tex_escape(task.upper())}}}\\label{{tarea-{task.lower()}}}")
        
        cols = "l" + "l" * len(model_names)
        latex_out.append(f"\\begin{{longtable}}[]{{@{{}}{cols}@{{}}}}")
        latex_out.append(r"\toprule\noalign{}")
        latex_out.append("Métrica & " + " & ".join([tex_escape(m) for m in model_names]) + r" \\")
        latex_out.append(r"\midrule\noalign{}")
        latex_out.append(r"\endhead")
        latex_out.append(r"\bottomrule\noalign{}")
        
        if compact:
            latex_out.append(f"\\caption{{Métricas de rendimiento para la tarea {tex_escape(task.replace('_', ' ').title())} ({tex_escape(lengua)}).}}")
            latex_out.append(f"\\label{{tab:metricas_{task}_{tex_escape(lengua)}}}")
            
        latex_out.append(r"\endlastfoot")

        base_media = models_dict[model_names[0]][task]["media"]
        metrics_list = _build_metrics_list(base_media, compact=compact)
        for metric_info in metrics_list:
            latex_out.append(_process_table_row(metric_info, model_names, models_dict, task, low_is_better))
        
        latex_out.append(r"\end{longtable}")
        latex_out.append(f"\n\\subsubsection{{{ 'Ejemplo:' if compact else 'Ejemplos' }}}")
        
        ejemplos_base = models_dict[model_names[0]][task]["ejemplo"]
        if not isinstance(ejemplos_base, list): ejemplos_base = [ejemplos_base]
        idx_shortest = min(range(len(ejemplos_base)), key=lambda i: len(str(ejemplos_base[i])))
        
        if compact:
            common_fields, _ = _extract_fields(task, ejemplos_base[idx_shortest])
            append_latex_fields(latex_out, common_fields, max_chars=max_chars)
            for m in model_names:
                latex_out.append(f"\n\\paragraph{{{tex_escape(m)}}}~\\\\")
                ejs_m = models_dict[m][task]["ejemplo"]
                ej_m = (ejs_m if isinstance(ejs_m, list) else [ejs_m])[idx_shortest]
                _, specific_fields = _extract_fields(task, ej_m)
                append_latex_fields(latex_out, specific_fields, max_chars=max_chars)
        else:
            for m in model_names:
                latex_out.append(f"\n\\paragraph{{{tex_escape(m)}}}~\\\\")
                ejs_m = models_dict[m][task]["ejemplo"]
                ej_m = min(ejs_m if isinstance(ejs_m, list) else [ejs_m], key=lambda x: len(str(x)))
                common, specific = _extract_fields(task, ej_m)
                append_latex_fields(latex_out, common + specific, max_chars=max_chars, bold_label=False)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(latex_out))
    return "\n".join(latex_out)

def generate_latex_snippet_completo(models_dict, lengua="LENGUA", max_chars=300, output_path=None):
    """Genera un fragmento LaTeX completo, imprimiendo bloques de ejemplos dedicados modelo por modelo."""
    return generate_latex_results(models_dict, lengua, max_chars, output_path, compact=False)

def generate_latex_snippet_compacto(models_dict, lengua="LENGUA", max_chars=300, output_path=None):
    """Genera un fragmento LaTeX con tablas reducidas agrupando contextos comunes para optimizar el espacio."""
    return generate_latex_results(models_dict, lengua, max_chars, output_path, compact=True)

def merge_latex_files(input_paths, output_path, chapter_title="Resumen General"):
    """
    Consolida una lista de fragmentos .tex individuales en un documento LaTeX unificado bajo un nodo principal \\chapter.
    """
    merged_content = [f"\\chapter{{{chapter_title}}}", "% Archivo consolidado automáticamente\n"]
    for path in input_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                merged_content.append(f.read())
                merged_content.append("\n\\vspace{1em}\n") 
        except FileNotFoundError:
            print(f"Advertencia: No se encontró el archivo en {path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(merged_content))
    print(f"Éxito: Archivo consolidado creado en {output_path}")

# --- MÓDULO DE GRAFICACIÓN (DATA VISUALIZATION) ---

def generar_grafico_barras(datos_globales, extractor, titulo="Comparativa", ylabel="Valor", 
                           ylim=None, idiomas_orden=None, output_path=None, motor="sns"):
    """
    Genera un gráfico de barras comparativo agrupado por modelos, procesando los datos 
    de manera uniforme y permitiendo salidas estáticas (Seaborn) o dinámicas (Plotly).
    
    Procesa diccionarios complejos mediante lambdas/funciones de extracción o tuplas directas.
    
    Args:
        datos_globales (dict): Diccionario jerárquico {'Lengua': {'Modelo': {'Tarea': ...}}}.
        extractor (function|tuple|str): Criterio/Ruta para extraer la métrica numérica del sub-diccionario.
        titulo (str): Título principal expuesto en la cabecera del gráfico.
        ylabel (str): Etiqueta textual asignada al eje Y.
        ylim (tuple, opcional): Límites (min, max) de escala para el eje Y.
        idiomas_orden (list, opcional): Secuencia explícita de ordenamiento para las lenguas del eje X.
        output_path (str, opcional): Ruta física de destino para guardar la gráfica (.png, .html).
        motor (str): Motor de renderizado gráfico: 'sns' (Seaborn) o 'plotly' (Plotly).
    """
    records = []
    for lengua, models_dict in datos_globales.items():
        for modelo, tareas in models_dict.items():
            val = None
            if callable(extractor):
                try: val = extractor(tareas)
                except Exception: val = None
            elif isinstance(extractor, tuple):
                tarea = extractor[0]
                metrica = extractor[1]
                sub_metrica = extractor[2] if len(extractor) > 2 else None
                if tarea in tareas and "media" in tareas[tarea]:
                    m_val = tareas[tarea]["media"].get(metrica)
                    val = m_val.get(sub_metrica) if (sub_metrica and isinstance(m_val, dict)) else m_val
            elif isinstance(extractor, str):
                tareas_encontradas = [t for t, c in tareas.items() if "media" in c and extractor in c["media"]]
                if len(tareas_encontradas) > 1:
                    raise ValueError(f"La métrica '{extractor}' es ambigua en las tareas: {tareas_encontradas}. Usa tupla.")
                elif len(tareas_encontradas) == 1:
                    val = tareas[tareas_encontradas[0]]["media"].get(extractor)
            
            if isinstance(val, (int, float)):
                records.append({"Idioma": lengua.strip().title(), "Modelo": modelo, "Valor": round(val, 4)})
                
    if not records:
        print("No se extrajeron datos numéricos para la gráfica.")
        return

    df = pd.DataFrame(records)
    orden_x = [i.strip().title() for i in idiomas_orden] if idiomas_orden else sorted(df["Idioma"].unique())
    modelos = df["Modelo"].unique()
    colores_sns_muted = ["#4878d0", "#ee854a", "#6acc64", "#d55e00", "#82169b", "#ccb974"]
    color_map = {modelo: colores_sns_muted[i % len(colores_sns_muted)] for i, modelo in enumerate(modelos)}

    if motor.lower() == "sns":
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Idioma", y="Valor", hue="Modelo", order=orden_x, palette=color_map, edgecolor="0.2", linewidth=1)
        if ylim: plt.ylim(ylim)
        plt.title(titulo, fontsize=14, pad=15)
        plt.xlabel("Lengua objetivo", fontsize=12, labelpad=10)
        plt.ylabel(ylabel, fontsize=12, labelpad=10)
        plt.legend(title="Modelos", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        if output_path: plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    elif motor.lower() == "plotly":
        import plotly.graph_objects as go
        fig = go.Figure()
        for modelo in modelos:
            df_modelo = df[df["Modelo"] == modelo]
            valores_ordenados = [df_modelo[df_modelo["Idioma"] == idm]["Valor"].values[0] if not df_modelo[df_modelo["Idioma"] == idm].empty else 0 for idm in orden_x]
            fig.add_trace(go.Bar(
                name=modelo, x=orden_x, y=valores_ordenados, marker_color=color_map[modelo],
                marker_line_color="rgb(50,50,50)", marker_line_width=1,
                hovertemplate=f"<b>{modelo}</b><br>Idioma: %{{x}}<br>{ylabel}: %{{y}}<extra></extra>"
            ))
        fig.update_layout(
            title=dict(text=titulo, font=dict(size=16, color="rgb(30,30,30)"), x=0.01),
            xaxis=dict(title=dict(text="Lengua objetivo", font=dict(size=13)), tickfont=dict(size=12)),
            yaxis=dict(title=dict(text=ylabel, font=dict(size=13)), tickfont=dict(size=12), gridcolor="rgb(235,235,235)"),
            barmode='group', plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(title="Modelos", bordercolor="rgb(220,220,220)", borderwidth=1),
            margin=dict(l=60, r=40, t=60, b=60), width=900, height=500
        )
        if ylim: fig.update_yaxes(range=[ylim[0], ylim[1]])
        if output_path: fig.write_html(output_path) if output_path.endswith('.html') else fig.write_image(output_path, scale=2)
        fig.show()