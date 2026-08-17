from typing import Literal

from pydantic import BaseModel, Field


DiagnosisStatus = Literal["awaiting_description", "awaiting_answer", "complete"]
AnswerValue = Literal["yes", "no", "not_sure"]


class DiagnosisMetadata(BaseModel):
    location: str | None = None
    crop: str | None = None
    growth_stage: str | None = None


class StartResponse(BaseModel):
    session_id: str
    status: Literal["awaiting_description"]
    initial_distribution: dict[str, float]
    next_step: str = "describe"


class DescribeRequest(BaseModel):
    description: str = Field(..., min_length=1)


class Question(BaseModel):
    question_id: str
    text: str
    options: list[AnswerValue] = ["yes", "no", "not_sure"]


class DescribeResponse(BaseModel):
    session_id: str
    status: Literal["awaiting_answer"]
    observations: dict[str, bool | None]
    candidate_distribution: dict[str, float]
    question: Question


class AnswerRequest(BaseModel):
    question_id: str
    answer: AnswerValue


class DiagnosisResult(BaseModel):
    diagnosed_disease: str
    confidence_score: float
    confidence_note: str
    management: list[str]
    prevention: list[str]
    precautions: list[str]
    references: list[str]
    explanation: str


class AnswerResponse(BaseModel):
    session_id: str
    status: Literal["awaiting_answer", "complete"]
    question: Question | None = None
    result: DiagnosisResult | None = None


class StatusResponse(BaseModel):
    session_id: str
    status: DiagnosisStatus


class FollowupRequest(BaseModel):
    question: str = Field(..., min_length=1)


class FollowupResponse(BaseModel):
    answer: str
