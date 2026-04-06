from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from src.backend.agentic_framework.base_agent import BaseJsonAgent
from src.backend.utils.schemas import (
    AttackExposure,
    OpinionAssessment,
    ProfileConfiguration,
    SCORE_MAX,
    SCORE_MIN,
)


class ProfileGenerationResponse(BaseModel):
    categorical_attributes: Dict[str, str] = Field(default_factory=dict)
    continuous_attributes: Dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""


class OpinionResponse(BaseModel):
    score: int
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: int) -> int:
        if value < SCORE_MIN or value > SCORE_MAX:
            raise ValueError(f"Score must be in [{SCORE_MIN}, {SCORE_MAX}]")
        return value


class ExposureResponse(BaseModel):
    exposure_text: str
    platform: str
    persuasion_strategy: str
    intensity_hint: float = Field(ge=0.0, le=1.0)


class ExposureReviewResponse(BaseModel):
    realism_score: float = Field(ge=0.0, le=1.0)
    coherence_score: float = Field(ge=0.0, le=1.0)
    rewrite_required: bool
    rewrite_feedback: str
    notes: str


class OpinionCoherenceReviewResponse(BaseModel):
    plausibility_score: float = Field(ge=0.0, le=1.0)
    consistency_score: float = Field(ge=0.0, le=1.0)
    rewrite_required: bool
    rewrite_feedback: str
    notes: str


class ProfileGenerationAgent:
    def __init__(self, base_agent: BaseJsonAgent):
        self.base = base_agent

    def generate(
        self,
        run_id: str,
        call_id: str,
        profile_id: str,
        seed: int,
        profile_leaf_nodes: List[str],
        deterministic_seed_profile: ProfileConfiguration,
    ) -> ProfileConfiguration:
        payload = {
            "profile_id": profile_id,
            "seed": seed,
            "ontology_profile_leaf_nodes": profile_leaf_nodes,
            "deterministic_seed_profile": deterministic_seed_profile.model_dump(),
        }
        response = self.base.run(
            prompt_name="profile_generation.md",
            payload=payload,
            response_model=ProfileGenerationResponse,
            run_id=run_id,
            call_id=call_id,
        )
        assert isinstance(response, ProfileGenerationResponse)
        return ProfileConfiguration(
            profile_id=profile_id,
            categorical_attributes=response.categorical_attributes,
            continuous_attributes=response.continuous_attributes,
            selected_leaf_nodes=profile_leaf_nodes,
            metadata={"generation": "llm", "reasoning": response.reasoning},
        )


class BaselineOpinionAgent:
    def __init__(self, base_agent: BaseJsonAgent, model_name: str):
        self.base = base_agent
        self.model_name = model_name

    def assess(
        self,
        run_id: str,
        call_id: str,
        scenario_id: str,
        opinion_leaf: str,
        profile: ProfileConfiguration,
        review_feedback: Optional[str] = None,
    ) -> OpinionAssessment:
        payload = {
            "scenario_id": scenario_id,
            "opinion_leaf": opinion_leaf,
            "profile": profile.model_dump(),
        }
        if review_feedback:
            payload["review_feedback"] = review_feedback
        response = self.base.run(
            prompt_name="baseline_opinion.md",
            payload=payload,
            response_model=OpinionResponse,
            run_id=run_id,
            call_id=call_id,
        )
        assert isinstance(response, OpinionResponse)
        return OpinionAssessment(
            scenario_id=scenario_id,
            phase="baseline",
            opinion_leaf=opinion_leaf,
            score=response.score,
            confidence=response.confidence,
            reasoning=response.reasoning,
            model_name=self.model_name,
        )


class AttackExposureAgent:
    def __init__(self, base_agent: BaseJsonAgent, model_name: str):
        self.base = base_agent
        self.model_name = model_name

    def generate(
        self,
        run_id: str,
        call_id: str,
        scenario_id: str,
        opinion_leaf: str,
        attack_leaf: str,
        profile: ProfileConfiguration,
        baseline_score: int,
        attack_present: bool,
        attack_context: Optional[Dict[str, Any]] = None,
        review_feedback: Optional[str] = None,
    ) -> AttackExposure:
        payload = {
            "scenario_id": scenario_id,
            "opinion_leaf": opinion_leaf,
            "attack_leaf": attack_leaf,
            "profile": profile.model_dump(),
            "baseline_score": baseline_score,
            "attack_present": attack_present,
            "attack_context": attack_context or {},
        }
        if review_feedback:
            payload["review_feedback"] = review_feedback
        response = self.base.run(
            prompt_name="attack_exposure.md",
            payload=payload,
            response_model=ExposureResponse,
            run_id=run_id,
            call_id=call_id,
        )
        assert isinstance(response, ExposureResponse)
        return AttackExposure(
            scenario_id=scenario_id,
            attack_present=attack_present,
            attack_leaf=attack_leaf if attack_present else None,
            exposure_text=response.exposure_text,
            platform=response.platform,
            persuasion_strategy=response.persuasion_strategy,
            intensity_hint=response.intensity_hint,
            model_name=self.model_name,
        )


class AttackRealismReviewerAgent:
    def __init__(self, base_agent: BaseJsonAgent, model_name: str):
        self.base = base_agent
        self.model_name = model_name

    def review(
        self,
        run_id: str,
        call_id: str,
        scenario_id: str,
        opinion_leaf: str,
        attack_leaf: str,
        baseline_score: int,
        profile_snapshot: Dict[str, Any],
        generated_exposure: AttackExposure,
    ) -> ExposureReviewResponse:
        payload = {
            "scenario_id": scenario_id,
            "opinion_leaf": opinion_leaf,
            "attack_leaf": attack_leaf,
            "baseline_score": baseline_score,
            "profile_snapshot": profile_snapshot,
            "generated_exposure": generated_exposure.model_dump(),
        }
        response = self.base.run(
            prompt_name="attack_realism_review.md",
            payload=payload,
            response_model=ExposureReviewResponse,
            run_id=run_id,
            call_id=call_id,
        )
        assert isinstance(response, ExposureReviewResponse)
        return response


class OpinionCoherenceReviewerAgent:
    def __init__(self, base_agent: BaseJsonAgent, model_name: str):
        self.base = base_agent
        self.model_name = model_name

    def review(
        self,
        run_id: str,
        call_id: str,
        phase: str,
        scenario_id: str,
        opinion_leaf: str,
        profile_snapshot: Dict[str, Any],
        generated_assessment: OpinionAssessment,
        attack_present: bool,
        adversarial_direction: int = 0,
        baseline_score: Optional[int] = None,
        exposure_text: Optional[str] = None,
        intensity_hint: Optional[float] = None,
        heuristic_checks: Optional[Dict[str, Any]] = None,
    ) -> OpinionCoherenceReviewResponse:
        payload = {
            "phase": phase,
            "scenario_id": scenario_id,
            "opinion_leaf": opinion_leaf,
            "profile_snapshot": profile_snapshot,
            "generated_assessment": generated_assessment.model_dump(),
            "attack_present": attack_present,
            "adversarial_direction": adversarial_direction,
            "baseline_score": baseline_score,
            "exposure_text": exposure_text,
            "intensity_hint": intensity_hint,
            "heuristic_checks": heuristic_checks or {},
        }
        response = self.base.run(
            prompt_name="opinion_coherence_review.md",
            payload=payload,
            response_model=OpinionCoherenceReviewResponse,
            run_id=run_id,
            call_id=call_id,
        )
        assert isinstance(response, OpinionCoherenceReviewResponse)
        return response


class PostAttackOpinionAgent:
    def __init__(self, base_agent: BaseJsonAgent, model_name: str):
        self.base = base_agent
        self.model_name = model_name

    def assess(
        self,
        run_id: str,
        call_id: str,
        scenario_id: str,
        opinion_leaf: str,
        profile: ProfileConfiguration,
        baseline_score: int,
        exposure_text: str,
        attack_present: bool,
        adversarial_direction: int = 0,
        intensity_hint: Optional[float] = None,
        review_feedback: Optional[str] = None,
    ) -> OpinionAssessment:
        payload = {
            "scenario_id": scenario_id,
            "opinion_leaf": opinion_leaf,
            "profile": profile.model_dump(),
            "baseline_score": baseline_score,
            "exposure_text": exposure_text,
            "attack_present": attack_present,
            "adversarial_direction": adversarial_direction,
            "intensity_hint": intensity_hint,
        }
        if review_feedback:
            payload["review_feedback"] = review_feedback
        response = self.base.run(
            prompt_name="post_attack_opinion.md",
            payload=payload,
            response_model=OpinionResponse,
            run_id=run_id,
            call_id=call_id,
        )
        assert isinstance(response, OpinionResponse)
        return OpinionAssessment(
            scenario_id=scenario_id,
            phase="post_attack",
            opinion_leaf=opinion_leaf,
            score=response.score,
            confidence=response.confidence,
            reasoning=response.reasoning,
            model_name=self.model_name,
        )
