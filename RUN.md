# How to run this project

Every command below is meant to be pasted into the **VS Code integrated terminal**
(`` Ctrl+` `` to open it). Windows PowerShell is the default on Windows; the
macOS/Linux equivalent is given wherever it differs.

> **Prefer not to install anything?** The portfolio site has a **Run this project**
> button that opens this repository in a free GitHub Codespace, installs the
> dependencies and runs the whole pipeline for you:
> <https://nikhil201716.github.io/nikhil-data-portfolio/pages/project.html?id=11>

---

## 1. Prerequisites

```powershell
python --version    # 3.11 or newer
git --version
```

### Required — Java, for PySpark

The Spark layer needs a JDK. A Java 8 JRE is often first on PATH on Windows, which Spark rejects, so set `JAVA_HOME` explicitly.

```powershell
$env:JAVA_HOME = "C:\Program Files\Microsoft\jdk-17.0.20.8-hotspot"
java -version
```

> Spark cannot **write** on Windows without `winutils.exe`. Reads and compute work. Run the Spark stages in WSL or a Codespace if you need the write path.

### Optional — the local LLM stages

This project has stages that use a local model through [Ollama](https://ollama.com). They are **optional**: without it those stages are skipped or fall back to their deterministic control arm, and every other stage runs normally.

```powershell
winget install Ollama.Ollama
ollama pull qwen2.5:0.5b
ollama list
```

---

---

## 2. One-time setup

```powershell
git clone https://github.com/Nikhil201716/11-Fair-Lending-Intelligence-Platform.git
cd 11-Fair-Lending-Intelligence-Platform
```

Create and activate a virtual environment. This keeps the project's dependencies
from colliding with anything else on your machine.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

<details>
<summary>If PowerShell refuses to run the activation script</summary>

Windows blocks unsigned scripts by default. Allow them for your own user account only:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
</details>

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the dependencies:

```powershell
pip install -r requirements.txt
```

> **Tip:** with the venv active, VS Code will offer to select it as the interpreter.
> Accept — otherwise the Run and Debug buttons use your global Python and you get
> `ModuleNotFoundError`.

---

## 3. Run it

### Everything, in one command

```powershell
python scripts/run_pipeline.py
```

That runs every stage below in order and is the normal way to use this repository.

### Or one stage at a time

Useful when you are changing a single stage and do not want to rebuild everything.

| # | Command | What it does |
|---|---|---|
| 1 | `python scripts/generate_lending_data.py` | 1/8 Generating synthetic lending data (400k applications, injected proxy) |
| 2 | `python rag/generate_policy_corpus.py` | 2/8 Generating the policy document corpus |
| 3 | `python risk_model/train_model.py` | 4/8 Training credit risk models (with vs. without the geographic proxy) |
| 4 | `python risk_model/shap_explain.py` | 5/8 SHAP explainability, validated against ground truth |
| 5 | `python fairness/audit.py` | 6/8 Fair lending audit (four-fifths, parity, equalized odds) |
| 6 | `python fairness/proxy_mechanism.py` | 6b/8 Proxy mechanism analysis with bootstrap CIs |
| 7 | `python geospatial/redlining_analysis.py` | 7/8 Geospatial redlining analysis (H3) |
| 8 | `python rag/evaluate_retrieval.py` | 8/8 Retrieval evaluation (BM25 vs dense vs hybrid, + chunking experiment) |

Each stage produces the input the next one consumes, so run them in this order.


---

## 4. Explore the results

```powershell
streamlit run dashboard/streamlit_app.py
```

Opens on <http://localhost:8501>. VS Code will offer to forward the port and open it in your browser.

The pipeline writes everything it measures into `reports/`. Those files are the
source of every number quoted on the portfolio site — nothing is typed by hand.

```powershell
ls reports
```

---

## 5. What a correct run looks like

Expect AUC 0.7114 identically with and without the proxy, and thin-file disparate impact dropping from about 0.805 to 0.761 when the proxy is removed. That drop is the finding.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | the virtual environment is not active | re-run the activate command from step 2 |
| `FileNotFoundError` on a data file | an earlier stage was skipped | run the stages in the documented order, or use the one-command runner |
| `Activate.ps1 cannot be loaded` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| Numbers differ from the README | a seed or parameter changed | check the constants at the top of the generator script |
| `command not found` | dependency missing from this environment | `pip install -r requirements.txt` with the venv active |
| VS Code runs the wrong Python | interpreter not selected | `Ctrl+Shift+P` → *Python: Select Interpreter* → pick `.venv` |

---

## 7. Finish

```powershell
deactivate
```

---

## More

- **The 60+ page technical notebook** for this project is in [`docs/`](docs/) — it
  covers the business problem, the mathematics derived from first principles, a
  guided tour of the code, worked numerical examples and exercises with solutions.
- **All fifteen projects:** <https://nikhil201716.github.io/nikhil-data-portfolio/>

*Generated from this repository's own pipeline runner, so the stage list cannot
drift from the code.*
