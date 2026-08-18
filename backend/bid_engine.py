"""
TenderSync Bid/No-Bid Engine

Industry-aligned hybrid evaluation:
  Phase A — Deterministic hard gates (deal killers)
  Phase B — Deterministic numeric/keyword scoring
  Phase C — LLM structured scoring for subjective dimensions
  Phase D — Weighted PWin calculation and final decision
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# Known certification aliases for fuzzy matching (RFP wording varies widely).
CERT_ALIASES: Dict[str, List[str]] = {
    "iso 27001": ["iso27001", "iso/iec 27001", "isms", "information security management"],
    "iso 9001": ["iso9001", "iso/iec 9001", "quality management"],
    "iso 14001": ["iso14001", "environmental management"],
    "soc 2": ["soc2", "soc ii", "soc-2", "service organization control"],
    "soc 1": ["soc1", "soc i"],
    "pci dss": ["pci-dss", "pci", "payment card industry"],
    "fedramp": ["fed ramp", "federal risk and authorization"],
    "hipaa": ["health insurance portability"],
    "gdpr": ["general data protection regulation"],
    "cmmi": ["capability maturity model"],
    "itil": ["information technology infrastructure library"],
    "casa": ["cloud application security assessment"],
    "nist": ["800-53", "80053", "cybersecurity framework"],
}

# Weights must sum to 100.
WEIGHTS: Dict[str, int] = {
    "capability_fit": 20,
    "compliance_readiness": 10,
    "past_performance": 15,
    "competitive_position": 15,
    "commercial_viability": 15,
    "resource_capacity": 15,
    "strategic_alignment": 10,
}

THRESHOLD_BID = 75
THRESHOLD_CONDITIONAL = 65
MIN_EXTRACTION_CONFIDENCE = 0.60


class DimensionScore(BaseModel):
    score: int = Field(ge=1, le=5, description="1=poor, 5=excellent")
    evidence: str = Field(description="One sentence citing RFP or profile facts")
    risks: List[str] = Field(default_factory=list)


class LLMScoringResult(BaseModel):
    capability_fit: DimensionScore
    past_performance: DimensionScore
    competitive_position: DimensionScore
    strategic_alignment: DimensionScore
    resource_capacity: DimensionScore


class GateResult(BaseModel):
    passed: bool
    gate_name: str
    detail: str
    severity: str = "critical"


class EvaluationResult(BaseModel):
    decision: str
    win_probability_score: int
    confidence_adjusted: bool = False
    hard_gates: List[GateResult]
    factor_scores: Dict[str, Any]
    rationale: str
    mitigations: List[str] = Field(default_factory=list)
    recommendation_summary: str


def normalize_company_profile(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing profile fields so the engine works before DB migration."""
    profile = dict(raw)
    profile.setdefault("organization_name", "")
    profile.setdefault("active_certifications", [])
    profile.setdefault("core_capabilities", [])
    profile.setdefault("past_performance_sectors", [])
    profile.setdefault("strategic_focus_areas", [])
    profile.setdefault("geographic_coverage", [])
    profile.setdefault("insurance_coverage", {})
    profile.setdefault("min_bid_lead_time_days", 14)
    profile.setdefault("team_capacity_score", 3)
    profile.setdefault("relationship_strength_score", 2)
    return profile


def _normalize(text: Any) -> str:
    if isinstance(text, dict):
        text = " ".join(str(v) for v in text.values())
    elif not isinstance(text, str):
        text = str(text or "")
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _parse_monetary_value(text: str) -> Optional[float]:
    if not text:
        return None
    cleaned = text.lower().replace(",", "")
    multipliers = {
        "crores": 10_000_000,
        "crore": 10_000_000,
        "cr": 10_000_000,
        "lakhs": 100_000,
        "lakh": 100_000,
        "lacs": 100_000,
        "lac": 100_000,
        "billion": 1_000_000_000,
        "b": 1_000_000_000,
        "million": 1_000_000,
        "m": 1_000_000,
        "thousand": 1_000,
        "k": 1_000,
    }
    for word, mult in multipliers.items():
        m = re.search(r"([\d.]+)\s*" + word, cleaned)
        if m:
            try:
                return float(m.group(1)) * mult
            except ValueError:
                pass
    amounts: List[float] = []
    for match in re.finditer(r"[\$£€₹]?\s*([\d.]+)", cleaned):
        try:
            amounts.append(float(match.group(1)))
        except ValueError:
            pass
    return max(amounts) if amounts else None


def _cert_matches(required: str, held: str) -> bool:
    req_n, held_n = _normalize(required), _normalize(held)
    if req_n in held_n or held_n in req_n:
        return True
    for _canonical, aliases in CERT_ALIASES.items():
        all_forms = [_normalize(_canonical)] + [_normalize(a) for a in aliases]
        req_hit = any(form in req_n for form in all_forms)
        held_hit = any(form in held_n for form in all_forms)
        if req_hit and held_hit:
            return True
    return False


def _keyword_overlap(required_items: List[str], capabilities: List[str]) -> float:
    if not required_items:
        return 0.5
    if not capabilities:
        return 0.3
    cap_normalized = [_normalize(c) for c in capabilities]
    hits = 0
    for item in required_items:
        item_n = _normalize(item)
        if any(cap in item_n or item_n in cap for cap in cap_normalized if cap):
            hits += 1
    return hits / len(required_items)


def evaluate_hard_gates(tender: Dict[str, Any], profile: Dict[str, Any]) -> List[GateResult]:
    gates: List[GateResult] = []

    required_certs = tender.get("mandatory_compliance_criteria") or []
    held_certs = profile.get("active_certifications") or []
    missing = [
        req for req in required_certs
        if req and not any(_cert_matches(req, held) for held in held_certs)
    ]
    if missing:
        gates.append(GateResult(
            passed=False,
            gate_name="mandatory_certification",
            detail=f"Missing required certifications: {', '.join(missing)}",
        ))
    else:
        gates.append(GateResult(
            passed=True,
            gate_name="mandatory_certification",
            detail="All identified mandatory certifications satisfied, or none were required.",
        ))

    min_val = profile.get("min_contract_value")
    budget_raw = tender.get("estimated_value_or_budget", "")
    parsed_budget = _parse_monetary_value(str(budget_raw))

    if min_val is not None and parsed_budget is not None:
        if parsed_budget < float(min_val):
            gates.append(GateResult(
                passed=False,
                gate_name="minimum_contract_value",
                detail=(
                    f"Estimated contract value (${parsed_budget:,.0f}) is below your "
                    f"minimum target (${float(min_val):,.0f})."
                ),
            ))
        else:
            gates.append(GateResult(
                passed=True,
                gate_name="minimum_contract_value",
                detail="Contract value meets your minimum threshold.",
            ))
    elif min_val is not None and parsed_budget is None:
        gates.append(GateResult(
            passed=True,
            gate_name="minimum_contract_value",
            detail="Budget could not be parsed from RFP; gate skipped (verify manually).",
            severity="warning",
        ))

    max_val = profile.get("max_contract_value")
    if max_val is not None and parsed_budget is not None and parsed_budget > float(max_val):
        gates.append(GateResult(
            passed=False,
            gate_name="maximum_contract_value",
            detail=(
                f"Contract value (${parsed_budget:,.0f}) exceeds your stated maximum "
                f"capacity (${float(max_val):,.0f})."
            ),
        ))

    confidence = tender.get("confidence_score", 1.0)
    if confidence < MIN_EXTRACTION_CONFIDENCE:
        gates.append(GateResult(
            passed=False,
            gate_name="extraction_confidence",
            detail=(
                f"AI extraction confidence ({confidence:.0%}) is too low for an "
                "automated bid decision. Human review required."
            ),
        ))
    else:
        gates.append(GateResult(
            passed=True,
            gate_name="extraction_confidence",
            detail=f"Extraction confidence acceptable ({confidence:.0%}).",
            severity="warning" if confidence < 0.85 else "info",
        ))

    return gates


def score_compliance_readiness(tender: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[int, str]:
    required = tender.get("mandatory_compliance_criteria") or []
    held = profile.get("active_certifications") or []
    if not required:
        return 4, "No mandatory compliance criteria identified in the RFP."
    matched = sum(1 for req in required if any(_cert_matches(req, h) for h in held))
    ratio = matched / len(required)
    if ratio >= 1.0:
        return 5, f"Full compliance coverage ({matched}/{len(required)} requirements met)."
    if ratio >= 0.75:
        return 3, f"Partial compliance ({matched}/{len(required)}); some gaps may be closable."
    return 1, f"Critical compliance gaps ({matched}/{len(required)} requirements met)."


def score_commercial_viability(tender: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[int, str]:
    min_val = profile.get("min_contract_value")
    budget = _parse_monetary_value(str(tender.get("estimated_value_or_budget", "")))
    if budget is None or min_val is None:
        return 3, "Budget unclear or minimum not set; neutral commercial score applied."
    ratio = budget / float(min_val)
    if ratio >= 3.0:
        return 5, "Strong margin potential — budget well above your minimum."
    if ratio >= 1.5:
        return 4, "Adequate commercial viability."
    if ratio >= 1.0:
        return 3, "Meets minimum; thin margin expected."
    return 2, "Below your target contract value."


def score_capability_fit_deterministic(tender: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[int, str]:
    deliverables = tender.get("key_deliverables") or []
    capabilities = profile.get("core_capabilities") or []
    overlap = _keyword_overlap(deliverables, capabilities)
    if not capabilities:
        return 3, "No core capabilities in profile; add them for accurate capability scoring."
    if overlap >= 0.8:
        return 5, f"Strong deliverable overlap ({overlap:.0%})."
    if overlap >= 0.5:
        return 4, f"Moderate deliverable overlap ({overlap:.0%})."
    if overlap >= 0.25:
        return 3, f"Partial overlap ({overlap:.0%}); partnering may be needed."
    return 2, f"Weak overlap ({overlap:.0%}) with your core capabilities."


def compute_pwin(factor_scores: Dict[str, int]) -> int:
    total = 0.0
    for dimension, weight in WEIGHTS.items():
        score = factor_scores.get(dimension, 3)
        total += (score / 5.0) * weight
    return round(total)


def make_decision(pwin: int, gates: List[GateResult]) -> Tuple[str, List[str]]:
    failed_critical = [
        gate for gate in gates
        if not gate.passed and gate.severity == "critical"
    ]
    if failed_critical:
        return "NO-BID", [gate.detail for gate in failed_critical]

    if pwin >= THRESHOLD_BID:
        return "BID", []
    if pwin >= THRESHOLD_CONDITIONAL:
        return "CONDITIONAL", [
            "Score is in the conditional range — pursue only if mitigations are feasible.",
            "Consider partnering, scope clarification, or pricing adjustment.",
        ]
    return "NO-BID", ["Weighted PWin score is below the minimum pursuit threshold."]


def apply_confidence_penalty(pwin: int, confidence: float) -> Tuple[int, bool]:
    if confidence >= 0.85:
        return pwin, False
    penalty = int((0.85 - confidence) * 30)
    return max(0, pwin - penalty), True


def build_llm_scoring_prompt(tender: Dict[str, Any], profile: Dict[str, Any]) -> str:
    return f"""You are a capture manager scoring an RFP opportunity. Score ONLY these dimensions on a 1-5 scale.

COMPANY PROFILE:
- Core Capabilities: {', '.join(profile.get('core_capabilities') or ['Not specified'])}
- Past Performance Sectors: {', '.join(profile.get('past_performance_sectors') or ['Not specified'])}
- Strategic Focus Areas: {', '.join(profile.get('strategic_focus_areas') or ['Not specified'])}
- Team Capacity (1-5): {profile.get('team_capacity_score', 3)}
- Relationship Strength (1-5): {profile.get('relationship_strength_score', 2)}

RFP EXTRACTED DATA:
- Issuing Authority: {tender.get('issuing_authority')}
- Budget: {tender.get('estimated_value_or_budget')}
- Deliverables: {', '.join(tender.get('key_deliverables') or [])}
- Mandatory Compliance: {', '.join(tender.get('mandatory_compliance_criteria') or [])}
- Deadline: {tender.get('submission_deadline')}

Return JSON with these keys, each containing score (1-5), evidence (one sentence), risks (list of strings):
- capability_fit: technical ability to deliver the scope of work
- past_performance: relevance of prior work to this RFP
- competitive_position: likelihood of winning vs likely competitors (assume cold bid unless profile suggests otherwise)
- strategic_alignment: fit with company growth priorities
- resource_capacity: can the team deliver on the timeline without overextension

Scoring guide: 1=very poor, 3=neutral/unknown, 5=excellent.
Be conservative. Do not inflate scores without evidence from the data above."""


def run_bid_evaluation(
    tender: Dict[str, Any],
    profile: Dict[str, Any],
    groq_client: Any = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    profile = normalize_company_profile(profile)

    gates = evaluate_hard_gates(tender, profile)
    failed_gates = [gate for gate in gates if not gate.passed]

    factor_scores: Dict[str, int] = {}
    factor_details: Dict[str, str] = {}

    score, detail = score_compliance_readiness(tender, profile)
    factor_scores["compliance_readiness"] = score
    factor_details["compliance_readiness"] = detail

    score, detail = score_commercial_viability(tender, profile)
    factor_scores["commercial_viability"] = score
    factor_details["commercial_viability"] = detail

    det_cap_score, det_cap_detail = score_capability_fit_deterministic(tender, profile)
    factor_scores["capability_fit"] = det_cap_score
    factor_details["capability_fit"] = det_cap_detail

    extraction_gate_failed = any(
        gate.gate_name == "extraction_confidence" and not gate.passed for gate in gates
    )

    if not extraction_gate_failed:
        try:
            completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Return strict JSON only. Be conservative in scoring."},
                    {"role": "user", "content": build_llm_scoring_prompt(tender, profile)},
                ],
                model=model,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            llm_scores = LLMScoringResult.model_validate_json(
                completion.choices[0].message.content
            )
            for dimension in (
                "capability_fit",
                "past_performance",
                "competitive_position",
                "strategic_alignment",
                "resource_capacity",
            ):
                dim_score = getattr(llm_scores, dimension)
                if dimension == "capability_fit":
                    factor_scores[dimension] = round(0.6 * dim_score.score + 0.4 * det_cap_score)
                    factor_details[dimension] = (
                        f"{dim_score.evidence} [Blended with keyword overlap analysis.]"
                    )
                else:
                    factor_scores[dimension] = dim_score.score
                    factor_details[dimension] = dim_score.evidence
        except Exception:
            for dimension in (
                "past_performance",
                "competitive_position",
                "strategic_alignment",
                "resource_capacity",
            ):
                factor_scores.setdefault(dimension, 3)
                factor_details.setdefault(
                    dimension,
                    "LLM scoring unavailable; neutral score of 3 applied.",
                )

    for dimension in WEIGHTS:
        factor_scores.setdefault(dimension, 3)
        factor_details.setdefault(dimension, "Not scored.")

    pwin = compute_pwin(factor_scores)
    confidence = tender.get("confidence_score", 1.0)
    pwin, adjusted = apply_confidence_penalty(pwin, confidence)

    decision, mitigations = make_decision(pwin, gates)

    ranked = sorted(
        ((dim, factor_scores[dim]) for dim in WEIGHTS),
        key=lambda item: item[1],
    )
    strengths = [dim.replace("_", " ") for dim, val in ranked if val >= 4]
    weaknesses = [dim.replace("_", " ") for dim, val in ranked if val <= 2]

    rationale_parts = [f"PWin {pwin}/100."]
    if strengths:
        rationale_parts.append(f"Strengths: {', '.join(strengths)}.")
    if weaknesses:
        rationale_parts.append(f"Weaknesses: {', '.join(weaknesses)}.")
    if failed_gates:
        rationale_parts.insert(0, f"Hard gate failed: {failed_gates[0].detail}")
    rationale = " ".join(rationale_parts)

    icon = {"BID": "BID", "CONDITIONAL": "CONDITIONAL", "NO-BID": "NO-BID"}[decision]
    result = EvaluationResult(
        decision=decision,
        win_probability_score=pwin,
        confidence_adjusted=adjusted,
        hard_gates=gates,
        factor_scores={
            "scores": factor_scores,
            "weights": WEIGHTS,
            "details": factor_details,
            "thresholds": {
                "bid": THRESHOLD_BID,
                "conditional": THRESHOLD_CONDITIONAL,
            },
        },
        rationale=rationale,
        mitigations=mitigations,
        recommendation_summary=f"{icon} — PWin {pwin}%",
    )
    return result.model_dump()
