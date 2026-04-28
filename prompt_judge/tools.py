"""
tools.py - Evaluation tools used by the Insight Quality Judge agent.

Each function is registered as an ADK Tool. The agent calls these tools to
score two AI-assistant answers (Betty vs Copilot) on six insight-quality
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

    This is the most heavily weighted dimension because for an AI engineering
    team validating chatbot outputs, a confidently-wrong answer is worse than
    a vague one.
    """
    return {
        "dimension": "Fidelidad y Fundamentacion",
        "weight": 0.25,
        "rubric": {
            "9-10": "Every factual claim is verifiable in the source. No hallucinations. No contradictions.",
            "7-8":  "Mostly faithful. One or two minor unsupported phrasings, but no fabricated facts.",
            "5-6":  "Generally aligned with the source but contains some unsupported claims or loose paraphrasing.",
            "3-4":  "Multiple unsupported claims or one clearly fabricated fact/number/entity.",
            "0-2":  "Heavily hallucinated; key claims contradict or invent content not in the source.",
        },
        "question": question,
        "source_excerpt": source_excerpt[:2000],
        "answer_betty": answer_betty,
        "answer_copilot": answer_copilot,
        "instruction": (
            "Cross-check each answer against the source_excerpt. Flag any "
            "specific claim that cannot be verified. Score Betty and Copilot "
            "independently 0-10 using the rubric. Return score_betty, "
            "score_copilot, and a short rationale citing specific claims."
        ),
    }


# ---------------------------------------------------------------------------
# Dimension 2 - Analytical Depth (20%)
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
    """
    return {
        "dimension": "Profundidad Analitica",
        "weight": 0.20,
        "rubric": {
            "9-10": "Identifies root causes, draws non-obvious comparisons, surfaces patterns or anomalies.",
            "7-8":  "Solid 'why' reasoning with at least one meaningful comparison or pattern.",
            "5-6":  "Some analysis but largely a structured summary of what the source says.",
            "3-4":  "Mostly rephrased summary; little to no causal or comparative reasoning.",
            "0-2":  "Pure paraphrase; no analytical content.",
        },
        "question": question,
        "source_excerpt": source_excerpt[:2000],
        "answer_betty": answer_betty,
        "answer_copilot": answer_copilot,
        "instruction": (
            "Judge whether each answer explains WHY (causes, drivers, "
            "comparisons, anomalies) or only WHAT (description). Score Betty "
            "and Copilot 0-10 independently and return a short rationale."
        ),
    }


# ---------------------------------------------------------------------------
# Dimension 3 - Specificity & Evidence (15%)
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
    """
    return {
        "dimension": "Especificidad y Evidencia",
        "weight": 0.15,
        "rubric": {
            "9-10": "Rich in concrete entities, numbers, dates, and direct quotes/citations from the source.",
            "7-8":  "Several specific data points; minor over-reliance on general phrasing.",
            "5-6":  "Mix of specific and vague; some claims lack concrete support.",
            "3-4":  "Mostly vague generalities; few specifics.",
            "0-2":  "Entirely abstract; no concrete entities, numbers, or citations.",
        },
        "question": question,
        "source_excerpt": source_excerpt[:2000],
        "answer_betty": answer_betty,
        "answer_copilot": answer_copilot,
        "instruction": (
            "Count concrete data points (entities, numbers, dates, quotes) in "
            "each answer. Score Betty and Copilot 0-10 and return a rationale."
        ),
    }


# ---------------------------------------------------------------------------
# Dimension 4 - Completeness (15%)
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
        "weight": 0.15,
        "rubric": {
            "9-10": "Every explicit and implicit sub-question is addressed thoroughly.",
            "7-8":  "All main parts addressed; minor sub-aspects glossed over.",
            "5-6":  "Core question addressed but at least one notable sub-question is missing.",
            "3-4":  "Multiple parts of the question unanswered.",
            "0-2":  "Largely off-topic or addresses only a small fragment of the question.",
        },
        "question": question,
        "source_excerpt": source_excerpt[:2000],
        "answer_betty": answer_betty,
        "answer_copilot": answer_copilot,
        "instruction": (
            "Decompose the question into sub-parts. Check each answer's "
            "coverage. Score Betty and Copilot 0-10 with a short rationale "
            "naming any sub-question that was skipped."
        ),
    }


# ---------------------------------------------------------------------------
# Dimension 5 - Actionability (15%)
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
    """
    return {
        "dimension": "Accionabilidad",
        "weight": 0.15,
        "rubric": {
            "9-10": "Concrete recommendations or next steps the reader can act on immediately.",
            "7-8":  "Useful conclusions with one or two actionable items.",
            "5-6":  "Conclusions present but generic or hard to act on.",
            "3-4":  "Mostly observations; the reader is left to figure out what to do.",
            "0-2":  "No conclusion or recommendation at all.",
        },
        "question": question,
        "source_excerpt": source_excerpt[:2000],
        "answer_betty": answer_betty,
        "answer_copilot": answer_copilot,
        "instruction": (
            "Note: if the question is purely descriptive (e.g., 'what does X "
            "say?'), both answers can score a neutral 5-6. Otherwise score "
            "Betty and Copilot 0-10 on whether the reader walks away with "
            "actionable conclusions, and return a rationale."
        ),
    }


# ---------------------------------------------------------------------------
# Dimension 6 - Clarity & Reasoning Transparency (10%)
# ---------------------------------------------------------------------------

def evaluate_clarity_reasoning(
    question: str,
    source_excerpt: str,
    answer_betty: str,
    answer_copilot: str,
) -> dict[str, Any]:
    """
    Score how readable and well-structured each answer is, and how clearly
    it distinguishes facts from inferences and signals confidence.
    """
    return {
        "dimension": "Claridad y Transparencia del Razonamiento",
        "weight": 0.10,
        "rubric": {
            "9-10": "Crisp structure, logic visible, distinguishes facts from inferences, signals uncertainty when warranted.",
            "7-8":  "Well organised and readable; reasoning mostly visible.",
            "5-6":  "Readable but reasoning is partly opaque or mixes facts with inferences.",
            "3-4":  "Disorganised or hard to follow; reasoning is hidden.",
            "0-2":  "Confusing, contradictory, or unstructured.",
        },
        "question": question,
        "source_excerpt": source_excerpt[:2000],
        "answer_betty": answer_betty,
        "answer_copilot": answer_copilot,
        "instruction": (
            "Score Betty and Copilot 0-10 on readability, structure, and "
            "whether the reasoning is transparent (facts vs inferences, "
            "confidence signals). Return a short rationale."
        ),
    }


# ---------------------------------------------------------------------------
# Dimension 7 - Final verdict synthesiser
# ---------------------------------------------------------------------------

def compile_verdict(
    faithfulness_scores: dict[str, Any],
    depth_scores: dict[str, Any],
    specificity_scores: dict[str, Any],
    completeness_scores: dict[str, Any],
    actionability_scores: dict[str, Any],
    clarity_scores: dict[str, Any],
) -> dict[str, Any]:
    """
    Aggregate the six dimension scores into a final weighted verdict.

    Call this AFTER you have scored all six dimensions. Each input dict must
    contain:
        - "weight"        - float (provided by the dimension tool)
        - "score_betty"   - int 0-10 (you supply this)
        - "score_copilot" - int 0-10 (you supply this)

    Tie threshold: if the absolute weighted-score gap is less than 0.30 the
    result is reported as "Tie" with the message "they're roughly equivalent".
    """
    dimensions = [
        faithfulness_scores,
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

    # Identify the weakest dimension for each AI (lowest absolute score).
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
