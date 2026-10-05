"""Builds notebook/jev_guardrail_study.ipynb from src/jev_study.py.

The notebook contains NO API keys. In Colab it reads them from the Secrets panel
(key icon in the left sidebar): OPENROUTER_API_KEY (required) and DEEPSEEK_API_KEY (optional).

    python notebook/make_notebook.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
script = open(os.path.join(ROOT, "src", "jev_study.py"), encoding="utf8").read()

_n = [0]
def _id():
    _n[0] += 1
    return f"cell{_n[0]:02d}"
def md(text): return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": text.splitlines(True)}
def code(text): return {"cell_type": "code", "id": _id(), "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(True)}

cells = [
    md("""# Evaluating Jev as a prompt-injection guardrail

Runs every test in this repository in one go and downloads a zip of the results.

1. **Add your keys as Colab secrets** (key icon in the left sidebar, then enable notebook access):
   * `OPENROUTER_API_KEY`: required, pays for the Jev calls (about $0.13 for the full study).
   * `DEEPSEEK_API_KEY`: optional, adds DeepSeek Flash as a comparison detector.
2. **Runtime > Change runtime type > T4 GPU** (used for the ProtectAI baseline and the translations).
3. **Runtime > Run all.** A full run takes about 10 to 15 minutes.

Every Jev call is cached as soon as it returns, so an interrupted run resumes where it stopped.
With `USE_DRIVE = True` the cache is kept in Google Drive and survives a runtime reset."""),
    code("!pip -q install httpx sentencepiece sacremoses"),
    code('''# ---- Configuration ----
import os

def secret(name):
    try:
        from google.colab import userdata
        return userdata.get(name) or ""
    except Exception:
        return os.environ.get(name, "")

OPENROUTER_API_KEY = secret("OPENROUTER_API_KEY")   # required
DEEPSEEK_API_KEY   = secret("DEEPSEEK_API_KEY")     # optional; leave unset to skip DeepSeek
DEEPSEEK_MODEL     = "deepseek-flash"

USE_DRIVE        = True    # keep the cache in Google Drive
MAX_BUDGET_USD   = 2.0     # hard stop for Jev spending
RUN_PROTECTAI    = True    # ProtectAI DeBERTa baseline (GPU, ~1 GB download)
RUN_TRANSLATION  = True    # translate 200 prompts into 6 languages
SMOKE            = False   # True = tiny test run to check the setup

assert OPENROUTER_API_KEY, "Add OPENROUTER_API_KEY in the Colab Secrets panel first."
print("OpenRouter key found; DeepSeek baseline", "enabled" if DEEPSEEK_API_KEY else "skipped")
'''),
    code("%%writefile jev_study.py\n" + script),
    code('''import jev_study
zip_path = jev_study.main(dict(
    OPENROUTER_API_KEY=OPENROUTER_API_KEY, DEEPSEEK_API_KEY=DEEPSEEK_API_KEY, DEEPSEEK_MODEL=DEEPSEEK_MODEL,
    USE_DRIVE=USE_DRIVE, MAX_BUDGET_USD=MAX_BUDGET_USD, RUN_PROTECTAI=RUN_PROTECTAI,
    RUN_TRANSLATION=RUN_TRANSLATION, SMOKE=SMOKE))'''),
    code('''# Download the results (run again if the browser blocked the automatic download)
from google.colab import files
files.download(zip_path)'''),
    code('''# Headline table
import pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
s = pd.read_csv(jev_study.P["out"] + "/summary_by_group.csv")
s[["suite", "run", "variant", "n", "catch_pct", "false_block_pct", "auc", "protectai_catch_pct", "protectai_false_block_pct", "regex_catch_pct"]].round(1)'''),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "jev_guardrail_study.ipynb")
json.dump(nb, open(out, "w", encoding="utf8"), indent=1)
print("wrote", out)
