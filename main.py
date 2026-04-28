"""
main.py - Entry point for the Insight Quality Judge POC.

Workflow per run:
    1. Read all PDF/DOCX files in ./documents/ (abort if empty).
    2. Ask the user for: the question, Betty's answer, Copilot's answer.
    3. Send everything to the judge agent.
    4. Parse the agent's JSON verdict.
    5. Append one row to ./evaluations.xlsx (12 columns).

Run from the project root:
    python main.py
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
if _PROJECT != "acpe-dev-uc-ai":
    sys.exit(
        f"[SAFETY] Wrong GCP project detected: '{_PROJECT}'. "
        "This POC must only run against 'acpe-dev-uc-ai'. "
        "Check your .env file."
    )

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from prompt_judge.agent import root_agent

console = Console()

PROJECT_ROOT = Path(__file__).parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
EXCEL_PATH = PROJECT_ROOT / "evaluations.xlsx"

EXCEL_COLUMNS = [
    "fecha",
    "hora",
    "pregunta",
    "archivos_usados",
    "ganador",
    "margen",
    "puntaje_betty",
    "puntaje_copilot",
    "puntajes_por_dimension_betty",
    "puntajes_por_dimension_copilot",
    "dimension_mas_debil_betty",
    "dimension_mas_debil_copilot",
    "razonamiento",
    "mejora_principal",
    "respuesta_betty",
    "respuesta_copilot",
]

# Short labels used inside the per-dimension cells.
DIMENSION_SHORT_LABELS = {
    "Fidelidad y Fundamentacion": "Fidelidad",
    "Profundidad Analitica": "Profundidad",
    "Especificidad y Evidencia": "Especificidad",
    "Completitud": "Completitud",
    "Accionabilidad": "Accionabilidad",
    "Claridad y Transparencia del Razonamiento": "Claridad",
}


def format_per_dim(scores: dict | None) -> str:
    if not isinstance(scores, dict):
        return ""
    parts = []
    for full_name, short_name in DIMENSION_SHORT_LABELS.items():
        if full_name in scores:
            parts.append(f"{short_name}: {scores[full_name]}")
    return " | ".join(parts)

_SESSION_COUNTER = 0


# ---------------------------------------------------------------------------
# Document loading (PDF + DOCX only)
# ---------------------------------------------------------------------------

def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    parts = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        parts.append(f"[Page {i}]\n{text}")
    return "\n\n".join(parts)


def _read_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_source_documents() -> tuple[str, list[str]]:
    """
    Load every supported file in ./documents/ and concatenate the text.

    Returns:
        (combined_text, list_of_filenames)

    Aborts the program with a clear error if the folder is missing or empty.
    """
    if not DOCUMENTS_DIR.exists():
        sys.exit(
            f"[ERROR] Source folder not found: {DOCUMENTS_DIR}\n"
            "Create the folder and place your PDF/DOCX files inside before running."
        )

    files = sorted(
        p for p in DOCUMENTS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in (".pdf", ".docx")
    )
    if not files:
        sys.exit(
            f"[ERROR] No PDF or DOCX files found in {DOCUMENTS_DIR}\n"
            "Place the same files you sent to Betty and Copilot inside this folder, then re-run.\n"
            "Supported formats: .pdf, .docx"
        )

    sections: list[str] = []
    filenames: list[str] = []
    for path in files:
        try:
            if path.suffix.lower() == ".pdf":
                text = _read_pdf(path)
            else:
                text = _read_docx(path)
        except Exception as e:
            sys.exit(f"[ERROR] Could not read {path.name}: {e}")
        sections.append(f"===== FILE: {path.name} =====\n{text}")
        filenames.append(path.name)

    return "\n\n".join(sections), filenames


# ---------------------------------------------------------------------------
# Multiline input (END sentinel - real chatbot answers contain blank lines)
# ---------------------------------------------------------------------------

def collect_multiline(label: str) -> str:
    console.print(f"\n[bold]{label}[/bold] [dim](paste your text, then type END on its own line)[/dim]:")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)
    text = "\n".join(lines).strip()
    if not text:
        sys.exit("[ERROR] Empty input. Aborting.")
    return text


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def build_judge_input(
    question: str,
    source_excerpt: str,
    answer_betty: str,
    answer_copilot: str,
) -> str:
    return (
        f"## Question\n{question}\n\n"
        f"## Source Excerpt (from the documents both AIs received)\n"
        f"```\n{source_excerpt}\n```\n\n"
        f"## Answer from Betty\n```\n{answer_betty}\n```\n\n"
        f"## Answer from Copilot\n```\n{answer_copilot}\n```\n\n"
        "Evaluate both answers using the six insight-quality dimensions and "
        "return the JSON verdict only."
    )


async def run_judge(
    question: str,
    source_excerpt: str,
    answer_betty: str,
    answer_copilot: str,
) -> str:
    global _SESSION_COUNTER
    _SESSION_COUNTER += 1
    session_id = f"eval_session_{_SESSION_COUNTER:03d}"

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="insight_quality_judge",
        user_id="poc_user",
        session_id=session_id,
    )

    runner = Runner(
        agent=root_agent,
        app_name="insight_quality_judge",
        session_service=session_service,
    )

    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(
            text=build_judge_input(question, source_excerpt, answer_betty, answer_copilot)
        )],
    )

    final_response = ""
    async for event in runner.run_async(
        user_id="poc_user",
        session_id=session_id,
        new_message=user_message,
    ):
        if event.is_final_response():
            print(event.content.parts[0])
            final_response = event.content.parts[0].text

    return final_response


# ---------------------------------------------------------------------------
# JSON parsing (the agent returns JSON; tolerate stray code fences)
# ---------------------------------------------------------------------------

def parse_verdict(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        # strip ```json ... ``` or ``` ... ``` fences
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip("`").strip()
    # find the first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        sys.exit(f"[ERROR] Agent did not return JSON. Raw output:\n{raw}")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        sys.exit(f"[ERROR] Could not parse agent JSON: {e}\nRaw:\n{raw}")


# ---------------------------------------------------------------------------
# Excel logging (append a row to evaluations.xlsx)
# ---------------------------------------------------------------------------

def append_to_excel(row: dict) -> None:
    from openpyxl import Workbook, load_workbook

    if EXCEL_PATH.exists():
        wb = load_workbook(EXCEL_PATH)
        ws = wb.active
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "Evaluations"
        ws.append(EXCEL_COLUMNS)

    ws.append([row.get(col, "") for col in EXCEL_COLUMNS])
    wb.save(EXCEL_PATH)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def print_banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]*** Insight Quality Judge ***[/bold cyan]\n"
            "[dim]Betty vs Copilot - LLM-as-Judge for AI Engineering[/dim]\n"
            f"[dim]Project: {_PROJECT}[/dim]",
            border_style="cyan",
        )
    )


async def _async_main() -> None:
    print_banner()

    # 1. Load source documents (abort if empty)
    console.print(Rule("[bold]Step 1 - Loading source documents[/bold]"))
    source_text, filenames = load_source_documents()
    console.print(f"[green]Loaded {len(filenames)} file(s):[/green] {', '.join(filenames)}")

    # 2. Collect inputs
    console.print(Rule("[bold]Step 2 - Inputs[/bold]"))
    question = Prompt.ask("\n[bold]Question that was asked to both AIs[/bold]").strip()
    if not question:
        sys.exit("[ERROR] Empty question. Aborting.")
    answer_betty = collect_multiline("Paste Betty's answer")
    answer_copilot = collect_multiline("Paste Copilot's answer")

    # 3. Run the judge
    console.print(Rule("[bold]Step 3 - Running the judge[/bold]"))
    console.print("[yellow]Evaluating... please wait ~30 s[/yellow]")
    raw = await run_judge(question, source_text, answer_betty, answer_copilot)
    verdict = parse_verdict(raw)

    # 4. Append to Excel
    now = datetime.now()
    row = {
        "fecha": now.strftime("%Y-%m-%d"),
        "hora": now.strftime("%H:%M:%S"),
        "pregunta": question,
        "archivos_usados": ", ".join(filenames),
        "ganador": verdict.get("winner", ""),
        "margen": verdict.get("margin", ""),
        "puntaje_betty": verdict.get("score_betty", ""),
        "puntaje_copilot": verdict.get("score_copilot", ""),
        "puntajes_por_dimension_betty": format_per_dim(verdict.get("scores_betty_per_dim")),
        "puntajes_por_dimension_copilot": format_per_dim(verdict.get("scores_copilot_per_dim")),
        "dimension_mas_debil_betty": verdict.get("betty_weakest_dim", ""),
        "dimension_mas_debil_copilot": verdict.get("copilot_weakest_dim", ""),
        "razonamiento": verdict.get("reasoning", ""),
        "mejora_principal": verdict.get("top_improvement", ""),
        "respuesta_betty": answer_betty,
        "respuesta_copilot": answer_copilot,
    }
    append_to_excel(row)

    # 5. Brief console confirmation (no full report - all detail is in Excel)
    console.print(Rule("[bold green]Done[/bold green]"))
    winner = verdict.get("winner", "?")
    if winner == "Empate":
        console.print(f"[bold]Resultado:[/bold] Empate - estan aproximadamente equivalentes "
                      f"(Betty {verdict.get('score_betty')} / Copilot {verdict.get('score_copilot')})")
    else:
        console.print(f"[bold]Ganador:[/bold] {winner} "
                      f"(Betty {verdict.get('score_betty')} / Copilot {verdict.get('score_copilot')}, "
                      f"margen {verdict.get('margin')})")
    console.print(f"[dim]Fila agregada a {EXCEL_PATH}[/dim]")


if __name__ == "__main__":
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
