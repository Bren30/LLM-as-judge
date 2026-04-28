"""
tools.py - Evaluation tools used by the Insight Quality Judge agent.

Each function is registered as an ADK Tool. The agent calls these tools to
score two AI-assistant answers (Betty vs Copilot) on seven insight-quality
dimensions, then synthesises a verdict.

Inputs to every dimension tool:
    question         - the question both AIs were asked
    source_excerpt   - excerpt of the source documents (for grounding checks)
    answer_betty     - Betty's full answer
    answer_copilot   - Copilot's full answer

Each dimension returns a rubric the agent reasons over and fills in with
score_betty and score_copilot (integers 0-10).
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Dimension 1 - Faithfulness & Grounding (25%)
# ---------------------------------------------------------------------------

def evaluate_faithfulness(
    question: str,
    source_excerpt: str,
    answer_betty: str,
    answer_copilot: str,
) -> dict[str, Any]:
    """
    Score how well each answer is grounded in the source documents and free
    of hallucinated facts, numbers, or entities.

    This is a high-weight dimension (25%) because for an AI engineering
    team validating chatbot outputs, a confidently-wrong answer is worse
    than a vague one.

    CRITICAL: Answers with verified citations [n] that map to source text
    score higher than answers that make plausible-sounding but unverifiable
    claims. An unverifiable claim is a HALLUCINATION until proven otherwise.
    """
    return {
        "dimension": "Fidelidad y Fundamentacion",
        "weight": 0.25,
        "rubric": {
            "9-10": "Every factual claim is directly verifiable in the source with specific citations or quotes. No hallucinations. No contradictions. Claims are explicitly traced to source text.",
            "7-8":  "Mostly faithful with citations. One or two minor unsupported phrasings, but no fabricated facts. Most claims can be traced to the source.",
            "5-6":  "Aligned with the source in spirit but contains some unsupported claims, loose paraphrasing, or missing citations. Claims are plausible but not directly verified.",
            "3-4":  "Multiple unsupported claims or one clearly fabricated fact/number/entity. May include invented details not present in the source.",
            "0-2":  "Heavily hallucinated; key claims contradict or invent content not in the source. Confidently states facts that cannot be found anywhere in the documents.",
        },
        "question": question,
        "source_excerpt": source_excerpt[:2000],
        "answer_betty": answer_betty,
        "answer_copilot": answer_copilot,
        "instruction": (
            "Cross-check EVERY factual claim in each answer against the source_excerpt. "
            "Flag any specific claim that cannot be verified in the source text as a "
            "hallucination. Answers that explicitly cite source text (e.g., quotes, "
            "[n] references, section numbers) and can be verified score HIGHER than "
            "answers that make plausible but unverifiable claims. An answer that "
            "paraphrases loosely without citing is less faithful than one that quotes "
            "or references the source precisely. Score Betty and Copilot independently "
            "0-10 using the rubric. Return score_betty, score_copilot, and a short "
            "rationale citing specific claims."
        ),
    }


# ---------------------------------------------------------------------------
# Dimension 2 - Source Attribution (15%)
# ---------------------------------------------------------------------------

def evaluate_source_attribution(
    question: str,
    source_excerpt: str,
    answer_betty: str,
    answer_copilot: str,
) -> dict[str, Any]:
    """
    Score how explicitly and precisely each answer attributes its claims to
    the source documents using citations, references, quotes, or indicators
    like [n], page numbers, section references.

    This dimension rewards answers that make it easy for the reader to verify
    WHERE each claim comes from. An answer that lists facts without indicating
    which part of the source they come from scores LOW, even if the facts are
    correct (that is covered by Faithfulness).

    This is SEPARATE from Faithfulness: an answer can be faithful (correct
    claims) but still score low on attribution if it doesn't show the reader
    where each claim came from.
    """
    return {
        "dimension": "Atribucion de Fuentes",
        "weight": 0.20,
        "rubric": {
            "9-10": "Nearly every factual claim is explicitly attributed with citation markers [n], direct quotes in quotation marks, or precise section/page references. The reader can easily locate the exact source passage for each claim.",
            "7-8":  "Most claims are attributed with citations [n] or clear references. A few claims lack explicit attribution but the answer generally makes it clear where information comes from.",
            "5-6":  "Some attribution present but inconsistent. Several claims reference the source generally without specific citation markers or quotes. The reader must guess which part of the source supports which claim.",
            "3-4":  "Minimal or superficial attribution. The answer may mention the document name but rarely ties specific claims to specific passages. Reader cannot verify individual claims easily.",
            "0-2":  "No attribution whatsoever. The answer presents information as if from general knowledge, with no references, citations, quotes, or source indicators of any kind.",
        },
        "question": question,
        "source_excerpt": source_excerpt[:2000],
        "answer_betty": answer_betty,
        "answer_copilot": answer_copilot,
        "instruction": (
            "CRITICAL: This dimension evaluates how explicitly each answer shows the "
            "reader WHERE each claim comes from in the source. It is NOT about whether "
            "claims are correct (that is Faithfulness). It is about TRACEABILITY.\n\n"
            "A high score requires that the answer uses EXPLICIT citation markers like "
            "[1], [2], [3] or numbered references that map individual claims to specific "
            "source passages. Simply mentioning the document name or saying 'according "
            "to the contract' at the top is NOT sufficient for a high score - each "
            "individual factual claim must be traceable.\n\n"
            "Key scoring guidance:\n"
            "- [n] citation markers with per-claim references = HIGH (8-10)\n"
            "- Direct quotes in quotation marks traced to source = HIGH (8-10)\n"
            "- 'According to the document' once at the start then lists facts = LOW (3-5)\n"
            "- No references at all, facts presented as general knowledge = VERY LOW (0-2)\n"
            "- Icons/emojis/bold headers do NOT count as attribution\n\n"
            "Score Betty and Copilot 0-10 independently. If one answer uses [n] markers "
            "and the other only mentions the document name generically, the one with "
            "[n] markers MUST score significantly higher."
        ),
    }


# ---------------------------------------------------------------------------
# Dimension 3 - Analytical Depth (20%)
# ---------------------------------------------------------------------------

def evaluate_analytical_depth(
    question: str,
    source_excerpt: str,
    answer_betty: str,
    answer_copilot: str,
) -> dict[str, Any]:
    """
    Score how much each answer goes beyond surface description into causal
    reasoning, comparisons, pattern recognition, or anomaly detection.

    CRITICAL: A well-formatted list or summary (even with emojis and headers)
    scores LOW on this dimension (3-5 range). Analytical depth requires
    genuine reasoning: explaining WHY, drawing connections, surfacing
    implications that are not explicitly stated in the source text.
    """
    return {
        "dimension": "Profundidad Analitica",
        "weight": 0.15,
        "rubric": {
            "9-10": "Goes far beyond the source text: identifies root causes, draws non-obvious connections, surfaces hidden implications, or detects patterns/anomalies. The answer explains WHY things happen, not just WHAT.",
            "7-8":  "Solid analytical reasoning with at least one meaningful causal explanation, comparison, or implication drawn from the source. Adds interpretive value beyond what the source explicitly states.",
            "5-6":  "Mostly a structured summary of what the source says, with minor interpretation. Answers WHAT but not WHY. May have good formatting but lacks real analytical insight.",
            "3-4":  "Rephrased summary or well-formatted list with no substantive analysis. The answer only restates what the source says without drawing connections, causes, or implications.",
            "0-2":  "Pure paraphrase or list with no analytical content whatsoever. No attempt to explain causes, draw comparisons, or surface implications.",
        },
        "question": question,
        "source_excerpt": source_excerpt[:2000],
        "answer_betty": answer_betty,
        "answer_copilot": answer_copilot,
        "instruction": (
            "CRITICAL DISTINCTION: A well-formatted summary or list (even with "
            "emojis, bold headers, or icons) is NOT analytical depth. Analytical "
            "depth means the answer explains WHY (causes, drivers, implications), "
            "draws connections between different parts of the source, or surfaces "
            "patterns not explicitly stated. Judge whether each answer provides "
            "genuine analytical reasoning beyond what the source text already says. "
            "An answer that merely organizes or restates source content scores in "
            "the 3-5 range regardless of formatting quality. Score Betty and Copilot "
            "0-10 independently and return a rationale explaining what analysis "
            "each answer provides (or lacks)."
        ),
    }


# ---------------------------------------------------------------------------
# Dimension 4 - Specificity & Evidence (15%)
# ---------------------------------------------------------------------------

def evaluate_specificity(
    question: str,
    source_excerpt: str,
    answer_betty: str,
    answer_copilot: str,
) -> dict[str, Any]:
    """
    Score how concrete each answer is - specific entities, numbers, dates,
    direct citations - vs vague generalities.

    CRITICAL: Emojis and icons are NOT specific evidence. Specificity means
    concrete data points that can be verified in the source.
    """
    return {
        "dimension": "Especificidad y Evidencia",
        "weight": 0.15,
        "rubric": {
            "9-10": "Rich in concrete, verifiable data: specific entities, numbers, dates, and direct quotes/citations from the source. Claims are precisely traceable to source text.",
            "7-8":  "Several specific data points with clear source references. May have minor gaps in precision but the evidence is concrete and checkable.",
            "5-6":  "Mix of specific and vague claims. Some data points are present but lack precise citation, or some claims are generalities that could apply to any document.",
            "3-4":  "Mostly vague generalities despite having access to the source. Few specifics, few or no citations, claims are generic.",
            "0-2":  "Entirely abstract; no concrete entities, numbers, dates, or citations. Information could apply to any document.",
        },
        "question": question,
        "source_excerpt": source_excerpt[:2000],
        "answer_betty": answer_betty,
        "answer_copilot": answer_copilot,
        "instruction": (
            "Count concrete, verifiable data points in each answer (entities, "
            "numbers, dates, quotes). Emojis, icons, and visual formatting do "
            "NOT count as specificity. An answer with precise citations [n] "
            "and quoted text scores higher than one with decorative elements "
            "but vague claims. Score Betty and Copilot 0-10 and return a "
            "rationale."
        ),
    }


# ---------------------------------------------------------------------------
# Dimension 5 - Completeness (10%)
# ---------------------------------------------------------------------------

def evaluate_completeness(
    question: str,
    source_excerpt: str,
    answer_betty: str,
    answer_copilot: str,
) -> dict[str, Any]:
    """
    Score whether each answer addresses every part of the question without
    silent gaps or skipped sub-questions.
    """
    return {
        "dimension": "Completitud",
        "weight": 0.10,
        "rubric": {
            "9-10": "Every explicit and implicit sub-question is addressed thoroughly with specific details from the source.",
            "7-8":  "All main parts addressed with relevant detail; minor sub-aspects could be deeper.",
            "5-6":  "Core question addressed but at least one notable sub-question is missing or superficially covered.",
            "3-4":  "Multiple parts of the question unanswered or only tangentially addressed.",
            "0-2":  "Largely off-topic or addresses only a small fragment of the question.",
        },
        "question": question,
        "source_excerpt": source_excerpt[:2000],
        "answer_betty": answer_betty,
        "answer_copilot": answer_copilot,
        "instruction": (
            "Decompose the question into ALL its sub-parts. Check each answer's "
            "coverage of every sub-question. An answer that provides thorough "
            "coverage with specific details scores higher than one that mentions "
            "more topics but superficially. Score Betty and Copilot 0-10 with "
            "a short rationale naming any sub-question that was skipped."
        ),
    }


# ---------------------------------------------------------------------------
# Dimension 6 - Actionability (10%)
# ---------------------------------------------------------------------------

def evaluate_actionability(
    question: str,
    source_excerpt: str,
    answer_betty: str,
    answer_copilot: str,
) -> dict[str, Any]:
    """
    Score whether each answer lands on usable conclusions - recommendations,
    next steps, or decisions - vs only observations.

    CRITICAL: A vague offer like "let me know if you need more details" is
    NOT actionable. True actionability means concrete next steps or specific
    recommendations derived from the source material.
    """
    return {
        "dimension": "Accionabilidad",
        "weight": 0.10,
        "rubric": {
            "9-10": "Provides specific, concrete recommendations or next steps directly derived from the source material that the reader can act on immediately.",
            "7-8":  "Clear conclusions with one or two actionable items grounded in the source content.",
            "5-6":  "Conclusions present but generic or hard to act on. May include vague offers like 'I can help more' without specific next steps.",
            "3-4":  "Mostly observations; the reader must figure out what to do. Any 'actionable' items are generic and not derived from the source.",
            "0-2":  "No conclusion or recommendation at all. Purely descriptive without any guidance for the reader.",
        },
        "question": question,
        "source_excerpt": source_excerpt[:2000],
        "answer_betty": answer_betty,
        "answer_copilot": answer_copilot,
        "instruction": (
            "IMPORTANT: Generic phrases like 'if you need more details, let me "
            "know' or 'I can help you with that' are NOT actionable conclusions. "
            "They are conversational filler and should score in the 3-5 range. "
            "True actionability means the answer provides SPECIFIC next steps, "
            "concrete recommendations, or clear decisions derived from the source "
            "material. For purely descriptive questions (e.g., 'what does X say?'), "
            "both answers can score a neutral 5-6. Score Betty and Copilot 0-10 "
            "and return a rationale."
        ),
    }


# ---------------------------------------------------------------------------
# Dimension 7 - Clarity & Reasoning Transparency (5%)
# ---------------------------------------------------------------------------

def evaluate_clarity_reasoning(
    question: str,
    source_excerpt: str,
    answer_betty: str,
    answer_copilot: str,
) -> dict[str, Any]:
    """
    Score how clearly each answer distinguishes facts from inferences and
    signals confidence/uncertainty.

    CRITICAL: This dimension measures TRANSPARENCY OF REASONING, not visual
    formatting. Emojis, icons, bold headers, and visual styling are NEUTRAL
    for this score. What matters is: does the answer distinguish what it KNOWS
    (from the source) from what it INFERS? Does it signal uncertainty?
    """
    return {
        "dimension": "Claridad y Transparencia del Razonamiento",
        "weight": 0.05,
        "rubric": {
            "9-10": "Extremely transparent reasoning: explicitly distinguishes facts cited from the source versus inferences drawn. Signals uncertainty where appropriate. The reader can tell exactly what comes from the source vs what is interpretation.",
            "7-8":  "Clear reasoning that is mostly transparent about source vs interpretation. Reader can trace claims back to evidence with confidence.",
            "5-6":  "Readable answer but reasoning is partly opaque. Mixes facts with inferences without clear demarcation. May be well-formatted but lacks reasoning transparency.",
            "3-4":  "Disorganised logic or hard-to-follow chains of reasoning. Facts and inferences are mixed without distinction.",
            "0-2":  "Confusing, contradictory, or no visible reasoning structure at all.",
        },
        "question": question,
        "source_excerpt": source_excerpt[:2000],
        "answer_betty": answer_betty,
        "answer_copilot": answer_copilot,
        "instruction": (
            "IMPORTANT: This dimension measures REASONING TRANSPARENCY, not "
            "visual formatting quality. Emojis, icons, bold text, headers, and "
            "other visual decoration are NEUTRAL and must NOT increase this "
            "score. What matters is: (1) Does the answer distinguish facts from "
            "the source from its own inferences? (2) Does it signal uncertainty "
            "when appropriate? (3) Can the reader trace which claims come from "
            "the source vs which are interpretations? An answer in plain text "
            "that clearly cites sources and signals uncertainty scores HIGHER "
            "than a visually pretty answer that mixes facts and inferences "
            "without distinction. Score Betty and Copilot 0-10 and return a "
            "rationale focused on reasoning transparency."
        ),
    }


# ---------------------------------------------------------------------------
# Dimension 8 - Final verdict synthesiser
# ---------------------------------------------------------------------------

def compile_verdict(
    faithfulness_scores: dict[str, Any],
    attribution_scores: dict[str, Any],
    depth_scores: dict[str, Any],
    specificity_scores: dict[str, Any],
    completeness_scores: dict[str, Any],
    actionability_scores: dict[str, Any],
    clarity_scores: dict[str, Any],
) -> dict[str, Any]:
    """
    Aggregate the seven dimension scores into a final weighted verdict.

    Call this AFTER you have scored all seven dimensions. Each input dict must
    contain:
        - "weight"        - float (provided by the dimension tool)
        - "score_betty"   - int 0-10 (you supply this)
        - "score_copilot" - int 0-10 (you supply this)

    Tie threshold: if the absolute weighted-score gap is less than 0.30 the
    result is reported as "Tie" with the message "they're roughly equivalent".
    """
    dimensions = [
        faithfulness_scores,
        attribution_scores,
        depth_scores,
        specificity_scores,
        completeness_scores,
        actionability_scores,
        clarity_scores,
    ]

    total_betty = 0.0
    total_copilot = 0.0
    breakdown: list[dict] = []

    for dim in dimensions:
        w = float(dim.get("weight", 0.0))
        sb = float(dim.get("score_betty", 0))
        sc = float(dim.get("score_copilot", 0))
        wb = w * sb
        wc = w * sc
        total_betty += wb
        total_copilot += wc
        breakdown.append({
            "dimension": dim.get("dimension", "Unknown"),
            "weight_pct": f"{int(w * 100)}%",
            "score_betty": sb,
            "score_copilot": sc,
            "weighted_betty": round(wb, 2),
            "weighted_copilot": round(wc, 2),
            "gap_betty_minus_copilot": round(wb - wc, 2),
        })

    total_betty = round(total_betty, 2)
    total_copilot = round(total_copilot, 2)
    gap = round(total_betty - total_copilot, 2)

    if abs(gap) < 0.30:
        winner = "Empate"
        margin = 0.0
        verdict_message = "estan aproximadamente equivalentes"
    elif gap > 0:
        winner = "Betty"
        margin = abs(gap)
        verdict_message = f"Betty gana por {margin:.2f}"
    else:
        winner = "Copilot"
        margin = abs(gap)
        verdict_message = f"Copilot gana por {margin:.2f}"

    betty_weakest = min(breakdown, key=lambda d: d["score_betty"])["dimension"]
    copilot_weakest = min(breakdown, key=lambda d: d["score_copilot"])["dimension"]

    return {
        "weighted_score_betty": total_betty,
        "weighted_score_copilot": total_copilot,
        "winner": winner,
        "margin": margin,
        "verdict_message": verdict_message,
        "betty_weakest_dim": betty_weakest,
        "copilot_weakest_dim": copilot_weakest,
        "breakdown": breakdown,
        "instruction": (
            "Now produce your final JSON verdict using these calculated "
            "values. Do not recompute the totals - use them as given."
        ),
    }