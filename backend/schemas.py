"""Pydantic request/response models for TenderSync API."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class CreateOrgRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)


class JoinOrgRequest(BaseModel):
    invite_code: str = Field(min_length=8, max_length=8)


class OrgResponse(BaseModel):
    id: str
    name: str
    invite_code: str
    created_at: Optional[str] = None


class OrgMemberResponse(BaseModel):
    user_id: str
    email: Optional[str] = None
    role: str
    joined_at: Optional[str] = None


class TenderSchema(BaseModel):
    tender_id: Optional[str] = Field(default=None, description="The unique reference code or ID number.")
    issuing_authority: Optional[str] = Field(default="Unknown Authority", description="The corporate or government body publishing the RFP.")
    submission_deadline: Optional[str] = Field(default="Not Specified", description="The strict closing date/time for the bid proposal.")
    estimated_value_or_budget: Optional[str] = Field(default="Not Specified", description="The financial value or budget limits defined.")
    key_deliverables: List[Union[str, Dict[str, Any]]] = Field(default_factory=list, description="Core products, services, or outcomes required.")
    mandatory_compliance_criteria: List[Union[str, Dict[str, Any]]] = Field(
        default_factory=list,
        description="Required legal, insurance, or ISO certifications."
    )
    confidence_score: float = Field(default=0.85, description="Overall confidence in extraction accuracy, 0.0 to 1.0.")


class CompanyProfileInput(BaseModel):
    """Fields the user sets during onboarding / profile edit."""

    organization_name: str = Field(min_length=2, max_length=200)
    min_contract_value: float = Field(gt=0, description="Minimum contract value worth pursuing (USD).")
    max_contract_value: Optional[float] = Field(default=None, gt=0)
    active_certifications: List[str] = Field(min_length=1)
    core_capabilities: List[str] = Field(min_length=1)
    strategic_focus_areas: List[str] = Field(default_factory=list)
    past_performance_sectors: List[str] = Field(default_factory=list)
    geographic_coverage: List[str] = Field(default_factory=list)
    insurance_coverage: Dict[str, float] = Field(default_factory=dict)
    min_bid_lead_time_days: int = Field(default=14, ge=1, le=365)
    team_capacity_score: int = Field(default=3, ge=1, le=5)
    relationship_strength_score: int = Field(default=2, ge=1, le=5)


class CompanyProfileResponse(CompanyProfileInput):
    org_id: str
    is_complete: bool
    completion_percentage: int = Field(default=0, ge=0, le=100)
    missing_fields: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ProfileStatusResponse(BaseModel):
    exists: bool
    is_complete: bool
    completion_percentage: int = Field(default=0, ge=0, le=100)
    missing_fields: List[str]
    can_use_app: bool
    profile: Optional[CompanyProfileResponse] = None


class AuthUserResponse(BaseModel):
    user_id: str
    email: Optional[str] = None


class TenderAnalysisSummary(BaseModel):
    id: str
    tender_id: Optional[str] = None
    filename: str
    issuing_authority: Optional[str] = None
    decision: Optional[str] = None
    win_probability_score: Optional[int] = None
    confidence_score: Optional[float] = None
    created_at: str


class TenderAnalysisDetail(BaseModel):
    id: str
    user_id: str
    filename: str
    extracted_data: Dict[str, Any]
    evaluation_data: Dict[str, Any]
    created_at: str

class ProposalUpdateRequest(BaseModel):
    content: str

class ProposalStatusUpdateRequest(BaseModel):
    status: str
    outcome_notes: Optional[str] = None


class GenerateProposalRequest(BaseModel):
    tender_analysis_id: Optional[str] = None
    market_tender_id: Optional[str] = None


class CompareTendersRequest(BaseModel):
    tender_ids: List[str] = Field(min_length=2, max_length=3)
