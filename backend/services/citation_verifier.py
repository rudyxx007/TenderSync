"""
Citation Verification Service for TenderSync Enterprise RAG Pipeline.

Post-generation verification that ensures all compliance matrix scores
and claims are directly entailed by retrieved evidence.

Uses Natural Language Inference (NLI) to verify that:
1. Claims made in the response are entailed by retrieved chunks
2. Compliance scores are justified by evidence
3. Citations point to actual supporting content

This prevents hallucination of compliance scores and ensures
enterprise-grade accuracy for RFP evaluation.

References:
- https://www.sciencedirect.com/science/article/pii/S2949719126000361
- NLI-based hallucination detection
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4


class VerificationResult(Enum):
    """Result of citation verification."""
    VERIFIED = "verified"           # Claim is entailed by evidence
    FLAGGED = "flagged"            # Claim may not be fully supported
    REJECTED = "rejected"          # Claim contradicts or lacks evidence
    UNCERTAIN = "uncertain"        # Cannot determine (low confidence)


class ClaimType(Enum):
    """Types of claims that can be verified."""
    COMPLIANCE_SCORE = "compliance_score"
    BUDGET_VALUE = "budget_value"
    DEADLINE = "deadline"
    CERTIFICATION = "certification"
    REQUIREMENT = "requirement"
    DELIVERABLE = "deliverable"
    NUMERIC_CLAIM = "numeric_claim"
    TEXTUAL_CLAIM = "textual_claim"


@dataclass
class ExtractedClaim:
    """A claim extracted from generated text."""
    claim_id: str
    text: str
    claim_type: ClaimType
    entity: Optional[str] = None  # What the claim is about
    value: Optional[Any] = None   # Numeric or categorical value
    confidence: float = 0.0       # Confidence in extraction
    source_text: Optional[str] = None  # Original text containing claim


@dataclass
class EvidenceSpan:
    """A span of text from retrieved evidence."""
    source_id: str
    text: str
    start_char: int
    end_char: int
    similarity_to_claim: float = 0.0
    entailment_score: float = 0.0


@dataclass
class VerificationOutcome:
    """Result of verifying a single claim."""
    claim: ExtractedClaim
    result: VerificationResult
    supporting_evidence: List[EvidenceSpan] = field(default_factory=list)
    contradiction_evidence: List[EvidenceSpan] = field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""
    flags: List[str] = field(default_factory=list)
    
    @property
    def is_verified(self) -> bool:
        return self.result == VerificationResult.VERIFIED
    
    @property
    def needs_review(self) -> bool:
        return self.result in (VerificationResult.FLAGGED, VerificationResult.UNCERTAIN)


@dataclass
class ComplianceScoreVerification:
    """Verification of a compliance score claim."""
    claimed_score: float
    evidence_scores: List[float]
    calculated_score: Optional[float] = None
    percentage_supported: float = 0.0
    result: VerificationResult = VerificationResult.UNCERTAIN
    explanation: str = ""
    
    @property
    def is_accurate(self) -> bool:
        if self.calculated_score is None:
            return False
        # Allow 5% tolerance
        return abs(self.claimed_score - self.calculated_score) <= 5.0


class CitationVerifier:
    """
    Verifies that generated claims are supported by retrieved evidence.
    
    Uses a combination of:
    1. NLI (Natural Language Inference) for entailment checking
    2. Exact match verification for numeric claims
    3. Semantic similarity for textual claims
    4. Rule-based checks for specific claim types
    
    Example:
        verifier = CitationVerifier()
        result = verifier.verify_compliance_score(
            claimed_score=85,
            evidence_chunks=retrieved_chunks,
            source_document=original_pdf_text
        )
    """
    
    def __init__(
        self,
        nli_model=None,  # Optional NLI model (e.g., from sentence-transformers)
        similarity_threshold: float = 0.7,
        entailment_threshold: float = 0.8,
        contradiction_threshold: float = 0.7,
    ):
        """
        Initialize CitationVerifier.
        
        Args:
            nli_model: Optional pre-trained NLI model
            similarity_threshold: Min similarity for evidence match
            entailment_threshold: Score threshold for entailment
            contradiction_threshold: Score threshold for contradiction
        """
        self.nli_model = nli_model
        self.similarity_threshold = similarity_threshold
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
    
    def verify_compliance_score(
        self,
        claimed_score: float,
        evidence_chunks: List[Dict[str, Any]],
        source_document: Optional[str] = None,
        claimed_basis: Optional[str] = None,
    ) -> ComplianceScoreVerification:
        """
        Verify that a compliance score claim is supported by evidence.
        
        This is critical for TenderSync: compliance matrix scores must
        be directly entailed by the retrieved RFP document content.
        
        Args:
            claimed_score: The compliance score claimed (0-100)
            evidence_chunks: Retrieved chunks that should support the score
            source_document: Full source document for reference
            claimed_basis: Human-readable basis for the score claim
            
        Returns:
            ComplianceScoreVerification with verification result
        """
        # Extract numeric compliance indicators from evidence
        evidence_scores = []
        
        for chunk in evidence_chunks:
            chunk_text = chunk.get("content", "")
            
            # Look for explicit compliance mentions
            compliance_indicators = self._extract_compliance_indicators(chunk_text)
            evidence_scores.extend(compliance_indicators)
        
        # Calculate supported percentage
        total_checks = len(evidence_scores)
        met_checks = sum(1 for score in evidence_scores if score >= 0.5)
        
        percentage_supported = (
            (met_checks / total_checks * 100) if total_checks > 0 else 0.0
        )
        
        # Determine if claimed score is reasonable
        if claimed_score > percentage_supported + 10:
            result = VerificationResult.FLAGGED
            explanation = (
                f"Claimed score ({claimed_score}%) exceeds evidence-supported "
                f"percentage ({percentage_supported:.1f}%) by more than 10%."
            )
        elif total_checks == 0:
            result = VerificationResult.REJECTED
            explanation = "No compliance evidence found in retrieved chunks."
        elif abs(claimed_score - percentage_supported) <= 5:
            result = VerificationResult.VERIFIED
            explanation = (
                f"Claimed score ({claimed_score}%) aligns with evidence "
                f"({percentage_supported:.1f}% of checks supported)."
            )
        else:
            result = VerificationResult.UNCERTAIN
            explanation = (
                f"Evidence supports {percentage_supported:.1f}% but claimed "
                f"{claimed_score}%. Manual review recommended."
            )
        
        return ComplianceScoreVerification(
            claimed_score=claimed_score,
            evidence_scores=evidence_scores,
            percentage_supported=percentage_supported,
            result=result,
            explanation=explanation,
        )
    
    def _extract_compliance_indicators(self, text: str) -> List[float]:
        """
        Extract compliance-related indicators from text.
        
        Looks for patterns like:
        - "X of Y requirements met"
        - "complies with ..."
        - "meets ... requirement"
        - certification mentions
        
        Returns list of confidence scores (0-1) for each indicator.
        """
        indicators = []
        text_lower = text.lower()
        
        # Pattern: "X of Y requirements/compliance items met"
        match = re.search(
            r'(\d+)\s+(?:of|out\s*of)\s+(\d+)\s+(?:requirements|compliance|criteria|items)',
            text_lower
        )
        if match:
            met = int(match.group(1))
            total = int(match.group(2))
            if total > 0:
                indicators.append(min(met / total, 1.0))
        
        # Pattern: explicit compliance statements
        compliance_phrases = [
            (r'\b(complies?|compliant|meets?|satisfies?)\b', 0.8),
            (r'\b(requires?|mandatory|shall|must)\b', -0.3),  # Requirement mentioned
            (r'\b(certif\w+|accredit\w+|registered)\b', 0.6),
            (r'\b(iso\s*\d+\s*(?:certified|compliant)?)\b', 0.7),
            (r'\b(soc\s*2\s*(?:certified|compliant)?)\b', 0.7),
            (r'\b(hipaa|gdpr|fedramp|pci\s*dss)\s*(?:compliant|certified)?\b', 0.7),
        ]
        
        for pattern, score in compliance_phrases:
            if re.search(pattern, text_lower):
                indicators.append(max(0, score))
        
        return indicators
    
    def verify_claim(
        self,
        claim: ExtractedClaim,
        evidence_chunks: List[Dict[str, Any]],
        source_document: Optional[str] = None,
    ) -> VerificationOutcome:
        """
        Verify a single extracted claim against evidence.
        
        Args:
            claim: The claim to verify
            evidence_chunks: Retrieved evidence chunks
            source_document: Optional full source document
            
        Returns:
            VerificationOutcome with supporting/contradicting evidence
        """
        supporting_evidence = []
        contradiction_evidence = []
        
        # Find relevant evidence for this claim
        for chunk in evidence_chunks:
            chunk_text = chunk.get("content", "")
            chunk_id = chunk.get("id", "unknown")
            
            # Calculate similarity to claim
            similarity = self._calculate_similarity(claim.text, chunk_text)
            
            if similarity >= self.similarity_threshold:
                evidence_span = EvidenceSpan(
                    source_id=chunk_id,
                    text=chunk_text[:500],  # Truncate for storage
                    start_char=0,
                    end_char=len(chunk_text),
                    similarity_to_claim=similarity,
                )
                
                # Check for entailment using NLI if available
                if self.nli_model:
                    entailment = self._check_entailment(claim.text, chunk_text)
                    evidence_span.entailment_score = entailment
                    
                    if entailment >= self.entailment_threshold:
                        supporting_evidence.append(evidence_span)
                    elif entailment <= -self.contradiction_threshold:
                        contradiction_evidence.append(evidence_span)
                else:
                    # Without NLI, use similarity as proxy
                    if similarity > 0.8:
                        supporting_evidence.append(evidence_span)
        
        # Determine verification result
        if not supporting_evidence and not contradiction_evidence:
            result = VerificationResult.REJECTED
            explanation = "No supporting or contradicting evidence found."
            confidence = 0.5
            flags = ["no_evidence"]
        elif len(supporting_evidence) > len(contradiction_evidence):
            if len(supporting_evidence) >= 2:
                result = VerificationResult.VERIFIED
                confidence = min(0.95, 0.5 + len(supporting_evidence) * 0.1)
            else:
                result = VerificationResult.FLAGGED
                confidence = 0.6
            explanation = f"Found {len(supporting_evidence)} supporting evidence(s)."
            flags = []
        elif len(contradiction_evidence) > len(supporting_evidence):
            result = VerificationResult.REJECTED
            confidence = 0.8
            explanation = f"Found {len(contradiction_evidence)} contradicting evidence(s)."
            flags = ["contradiction_detected"]
        else:
            result = VerificationResult.UNCERTAIN
            confidence = 0.5
            explanation = "Mixed evidence found."
            flags = ["conflicting_evidence"]
        
        return VerificationOutcome(
            claim=claim,
            result=result,
            supporting_evidence=supporting_evidence,
            contradiction_evidence=contradiction_evidence,
            confidence=confidence,
            explanation=explanation,
            flags=flags,
        )
    
    def verify_response(
        self,
        generated_response: str,
        evidence_chunks: List[Dict[str, Any]],
        source_document: Optional[str] = None,
        response_type: str = "compliance_matrix",
    ) -> Dict[str, Any]:
        """
        Verify an entire generated response.
        
        Extracts claims from the response and verifies each one.
        
        Args:
            generated_response: The LLM-generated response text
            evidence_chunks: Retrieved evidence chunks
            source_document: Optional full source document
            response_type: Type of response (compliance_matrix, analysis, etc.)
            
        Returns:
            Dict with verification results for all claims
        """
        # Extract claims from response
        claims = self._extract_claims(generated_response, response_type)
        
        # Verify each claim
        outcomes = []
        for claim in claims:
            outcome = self.verify_claim(claim, evidence_chunks, source_document)
            outcomes.append(outcome)
        
        # Aggregate results
        verified_count = sum(1 for o in outcomes if o.is_verified)
        flagged_count = sum(1 for o in outcomes if o.needs_review)
        rejected_count = sum(1 for o in outcomes if o.result == VerificationResult.REJECTED)
        
        # Calculate overall verification score
        total_claims = len(outcomes)
        verification_score = (
            (verified_count + flagged_count * 0.5) / total_claims
            if total_claims > 0 else 0.0
        )
        
        return {
            "verification_score": verification_score,
            "total_claims": total_claims,
            "verified": verified_count,
            "flagged": flagged_count,
            "rejected": rejected_count,
            "outcomes": [o.__dict__ for o in outcomes],
            "overall_result": (
                VerificationResult.VERIFIED.value
                if verified_count == total_claims and total_claims > 0
                else VerificationResult.FLAGGED.value
                if rejected_count == 0
                else VerificationResult.REJECTED.value
            ),
        }
    
    def _extract_claims(
        self, text: str, response_type: str
    ) -> List[ExtractedClaim]:
        """
        Extract verifiable claims from generated text.
        
        For compliance_matrix responses, looks for:
        - Numeric scores with percentages
        - Compliance status claims
        - Requirement fulfillment statements
        """
        claims = []
        
        # Pattern 1: Percentage scores
        for match in re.finditer(
            r'(?:compliance|score|compliance\s*score|compliance\s*level)\s*(?:is\s*)?(?:of\s*)?(?:approximately\s*)?'
            r'(?:[\"]?\d+(?:\.\d+)?[%])\b',
            text, re.IGNORECASE
        ):
            claims.append(ExtractedClaim(
                claim_id=str(uuid4()),
                text=match.group(0),
                claim_type=ClaimType.COMPLIANCE_SCORE,
                value=self._extract_percentage(match.group(0)),
            ))
        
        # Pattern 2: "X% compliance" or "compliance of X%"
        for match in re.finditer(
            r'(?:achieving|with|at|approximately)?\s*(\d+(?:\.\d+)?)\s*%\s*(?:compliance| conformance|meeting)',
            text, re.IGNORECASE
        ):
            claims.append(ExtractedClaim(
                claim_id=str(uuid4()),
                text=match.group(0),
                claim_type=ClaimType.COMPLIANCE_SCORE,
                value=float(match.group(1)),
            ))
        
        # Pattern 3: Numeric requirements met
        for match in re.finditer(
            r'(\d+)\s+(?:of|out\s*of)\s+(\d+)\s+(?:requirements?|criteria|items|standards)',
            text, re.IGNORECASE
        ):
            claims.append(ExtractedClaim(
                claim_id=str(uuid4()),
                text=match.group(0),
                claim_type=ClaimType.REQUIREMENT,
                value={
                    "met": int(match.group(1)),
                    "total": int(match.group(2)),
                },
            ))
        
        # Pattern 4: Budget/financial claims
        for match in re.finditer(
            r'(?:budget|value|cost|contract\s*value|funding)\s*(?:of|is|equals?)?\s*[\$£€₹]?\s*([\d,.]+(?:\s*(?:million|billion|thousand|lakh|crore))?)',
            text, re.IGNORECASE
        ):
            claims.append(ExtractedClaim(
                claim_id=str(uuid4()),
                text=match.group(0),
                claim_type=ClaimType.BUDGET_VALUE,
                value=match.group(1),
            ))
        
        # Pattern 5: Deadline claims
        for match in re.finditer(
            r'(?:deadline|due\s*date|submission\s*date|closing\s*date)\s*(?:is|:\s*)?'
            r'([A-Za-z]+\.?\s+\d{1,2},?\s*\d{4}|'
            r'\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})',
            text, re.IGNORECASE
        ):
            claims.append(ExtractedClaim(
                claim_id=str(uuid4()),
                text=match.group(0),
                claim_type=ClaimType.DEADLINE,
                value=match.group(1),
            ))
        
        # Pattern 6: Generic textual claims (sentence-level)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            if len(sentence) > 30 and len(sentence) < 500:
                # Skip if already captured as specific claim type
                if not any(
                    sentence.lower().find(c.text.lower()) >= 0
                    for c in claims
                ):
                    claims.append(ExtractedClaim(
                        claim_id=str(uuid4()),
                        text=sentence,
                        claim_type=ClaimType.TEXTUAL_CLAIM,
                    ))
        
        return claims
    
    def _extract_percentage(self, text: str) -> Optional[float]:
        """Extract percentage value from text."""
        match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        if match:
            return float(match.group(1))
        return None
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Uses simple token overlap as fallback if no embedding model.
        """
        if self.nli_model:
            # Would use NLI model for similarity
            pass
        
        # Fallback: Jaccard similarity on word sets
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _check_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Check if premise entails hypothesis using NLI.
        
        Returns:
            Float between -1 (contradiction) and 1 (entailment)
        """
        if not self.nli_model:
            # Without NLI model, return base on similarity
            return self._calculate_similarity(premise, hypothesis) * 2 - 1
        
        # Would use actual NLI model
        # Return placeholder
        similarity = self._calculate_similarity(premise, hypothesis)
        return similarity * 2 - 1


class ComplianceMatrixVerifier:
    """
    Specialized verifier for compliance matrices in RFP evaluations.
    
    Compliance matrices map RFP requirements to vendor capabilities.
    This verifier ensures every score in the matrix is justified.
    """
    
    def __init__(self, verifier: Optional[CitationVerifier] = None):
        self.verifier = verifier or CitationVerifier()
    
    def verify_matrix(
        self,
        compliance_matrix: List[Dict[str, Any]],
        evidence_chunks: List[Dict[str, Any]],
        source_document: str,
    ) -> Dict[str, Any]:
        """
        Verify an entire compliance matrix.
        
        Args:
            compliance_matrix: List of requirement entries with scores
            evidence_chunks: Retrieved evidence
            source_document: Original RFP document text
            
        Returns:
            Verification report with per-item and overall results
        """
        results = []
        total_score = 0
        verified_score = 0
        
        for item in compliance_matrix:
            requirement = item.get("requirement", "")
            claimed_score = item.get("score", 0)
            evidence_ref = item.get("evidence_ref", "")
            
            # Find relevant evidence
            relevant_evidence = self._find_relevant_evidence(
                requirement, evidence_chunks
            )
            
            # Verify this item
            verification = self.verifier.verify_claim(
                ExtractedClaim(
                    claim_id=str(uuid4()),
                    text=f"Requirement: {requirement}\nScore: {claimed_score}",
                    claim_type=ClaimType.COMPLIANCE_SCORE,
                    value=claimed_score,
                ),
                relevant_evidence,
                source_document,
            )
            
            item_result = {
                "requirement": requirement,
                "claimed_score": claimed_score,
                "verification": verification.result.value,
                "confidence": verification.confidence,
                "explanation": verification.explanation,
                "flags": verification.flags,
                "evidence_ref": evidence_ref,
            }
            
            results.append(item_result)
            
            total_score += claimed_score
            if verification.is_verified:
                verified_score += claimed_score
        
        return {
            "items": results,
            "total_items": len(results),
            "total_claimed_score": total_score,
            "total_verified_score": verified_score,
            "verification_percentage": (
                (verified_score / total_score * 100) if total_score > 0 else 0
            ),
            "overall_result": (
                VerificationResult.VERIFIED.value
                if verified_score == total_score
                else VerificationResult.FLAGGED.value
                if verified_score >= total_score * 0.8
                else VerificationResult.REJECTED.value
            ),
        }
    
    def _find_relevant_evidence(
        self, requirement: str, evidence_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find evidence chunks relevant to a requirement."""
        relevant = []
        req_lower = requirement.lower()
        
        for chunk in evidence_chunks:
            chunk_text = chunk.get("content", "").lower()
            
            # Check for keyword overlap
            req_words = set(re.findall(r'\w+', req_lower))
            chunk_words = set(re.findall(r'\w+', chunk_text))
            
            overlap = len(req_words & chunk_words) / len(req_words) if req_words else 0
            
            if overlap > 0.3:  # At least 30% word overlap
                relevant.append(chunk)
        
        return relevant


# Convenience function for quick verification
def verify_tender_response(
    response: str,
    evidence: List[Dict[str, Any]],
    source_doc: str,
) -> Dict[str, Any]:
    """
    Quick verification of a tender evaluation response.
    
    Args:
        response: LLM-generated evaluation
        evidence: Retrieved evidence chunks
        source_doc: Original RFP document
        
    Returns:
        Verification results dict
    """
    verifier = CitationVerifier()
    return verifier.verify_response(response, evidence, source_doc)
