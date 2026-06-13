# -*- coding: utf-8 -*-
"""
Exporta el notebook regulatorio M07 a HTML profesional (sin código, look documento) y a PDF.
- HTML: nbconvert --no-input + inyección de estilos.css (carátula, banner, tablas).
- PDF:  desde el HTML con Microsoft Edge / Chrome en modo headless (LaTeX no se usa).
Uso:  python export_reporte.py
Requisitos previos: el notebook debe estar ya EJECUTADO (con outputs).
"""
import os, sys, subprocess, glob, shutil

AQUI = os.path.dirname(os.path.abspath(__file__))
NB   = os.path.join(AQUI, "Reporte_Modelo_Riesgo_Credito.ipynb")
CSS  = os.path.join(AQUI, "estilos.css")
SAL  = os.path.normpath(os.path.join(AQUI, "..", "salidas"))
os.makedirs(SAL, exist_ok=True)
HTML = os.path.join(SAL, "Reporte_Modelo_Riesgo_Credito.html")
PDF  = os.path.join(SAL, "Reporte_Modelo_Riesgo_Credito.pdf")

# 1) HTML sin código (look documento regulatorio)
print("→ Exportando HTML…")
subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", "--no-input",
                "--output", os.path.splitext(os.path.basename(HTML))[0], "--output-dir", SAL, NB],
               check=True)

# 2) Inyectar estilos.css en el <head>
with open(CSS, encoding="utf-8") as f: css = f.read()
with open(HTML, encoding="utf-8") as f: html = f.read()
if "/* M07 css inyectado */" not in html:
    html = html.replace("</head>", f"<style>/* M07 css inyectado */\n{css}\n</style></head>", 1)
    with open(HTML, "w", encoding="utf-8") as f: f.write(html)
print(f"   HTML listo: {HTML}")

# 3) PDF desde el HTML con Edge/Chrome headless
def find_browser():
    cands = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for c in cands:
        if os.path.exists(c): return c
    return shutil.which("msedge") or shutil.which("chrome")

browser = find_browser()
if not browser:
    print("⚠ No se encontró Edge/Chrome. Genera el PDF abriendo el HTML e imprimiendo a PDF.")
    sys.exit(0)

url = "file:///" + HTML.replace("\\", "/")
print(f"→ Generando PDF con {os.path.basename(browser)}…")
udd = os.path.join(os.environ.get("TEMP", "."), "edge_m07_profile")
subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-sandbox",
                f"--user-data-dir={udd}", "--run-all-compositor-stages-before-draw",
                "--virtual-time-budget=20000", "--no-pdf-header-footer",
                f"--print-to-pdf={PDF}", url], check=False, timeout=180)
if os.path.exists(PDF):
    print(f"   PDF listo: {PDF}  ({os.path.getsize(PDF)//1024} KB)")
else:
    print("⚠ El navegador no escribió el PDF (revisar). El HTML sí está disponible.")
