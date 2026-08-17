from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.integrations.model_serving import get_model_serving_client
from app.integrations.reasoning_service import get_reasoning_service_client
from app.llm.symptom_extraction import extract_observations
from app.schemas.diagnosis import (
    AnswerRequest,
    AnswerResponse,
    DescribeRequest,
    DescribeResponse,
    DiagnosisMetadata,
    DiagnosisResult,
    FollowupRequest,
    FollowupResponse,
    StartResponse,
    StatusResponse,
)

router = APIRouter(prefix="/api/v1/diagnosis", tags=["diagnosis"])

_SESSIONS: dict[str, dict] = {}


def _get_session(session_id: str) -> dict:
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return session


@router.post("/start", response_model=StartResponse)
async def start_diagnosis(
    image: UploadFile = File(...),
    location: str | None = Form(default=None),
    crop: str | None = Form(default=None),
    growth_stage: str | None = Form(default=None),
) -> StartResponse:
    image_bytes = await image.read()
    image_distribution = await get_model_serving_client().predict(image_bytes)

    session_id = str(uuid4())
    metadata = DiagnosisMetadata(location=location, crop=crop, growth_stage=growth_stage)
    _SESSIONS[session_id] = {
        "status": "awaiting_description",
        "image_filename": image.filename,
        "metadata": metadata.model_dump(),
        "image_distribution": image_distribution,
        "answers": [],
    }
    return StartResponse(
        session_id=session_id,
        status="awaiting_description",
        initial_distribution=image_distribution,
    )


@router.post("/{session_id}/describe", response_model=DescribeResponse)
async def describe_symptoms(session_id: str, request: DescribeRequest) -> DescribeResponse:
    session = _get_session(session_id)
    if session["status"] != "awaiting_description":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Description is not expected now")

    observations = extract_observations(request.description)
    metadata = DiagnosisMetadata(**session["metadata"])
    reasoning_client = get_reasoning_service_client()
    candidate_distribution = await reasoning_client.expand_candidates(
        session["image_distribution"],
        observations,
        metadata,
    )
    question = await reasoning_client.next_question(candidate_distribution, [])
    session.update(
        {
            "status": "awaiting_answer" if question else "complete",
            "description": request.description,
            "observations": observations,
            "candidate_distribution": candidate_distribution,
        }
    )
    if question is None:
        result = await reasoning_client.finalize(candidate_distribution, [])
        session.update({"result": result.model_dump()})
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="No follow-up question is available")
    return DescribeResponse(
        session_id=session_id,
        status="awaiting_answer",
        observations=observations,
        candidate_distribution=candidate_distribution,
        question=question,
    )


@router.post("/{session_id}/answer", response_model=AnswerResponse)
async def answer_question(session_id: str, request: AnswerRequest) -> AnswerResponse:
    session = _get_session(session_id)
    if session["status"] != "awaiting_answer":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Answer is not expected now")

    answer = request.model_dump()
    session["answers"].append(answer)
    answers = [AnswerRequest(**item) for item in session["answers"]]
    reasoning_client = get_reasoning_service_client()
    candidate_distribution = session["candidate_distribution"]
    next_question = await reasoning_client.next_question(candidate_distribution, answers)
    if next_question is not None:
        return AnswerResponse(session_id=session_id, status="awaiting_answer", question=next_question)

    result = await reasoning_client.finalize(candidate_distribution, answers)
    session.update({"status": "complete", "result": result.model_dump()})
    return AnswerResponse(session_id=session_id, status="complete", result=result)


@router.get("/{session_id}/result", response_model=DiagnosisResult)
async def get_result(session_id: str) -> DiagnosisResult:
    session = _get_session(session_id)
    if session["status"] != "complete":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Diagnosis is not complete")
    return DiagnosisResult(**session["result"])


@router.get("/{session_id}/status", response_model=StatusResponse)
async def get_status(session_id: str) -> StatusResponse:
    session = _get_session(session_id)
    return StatusResponse(session_id=session_id, status=session["status"])


@router.post("/{session_id}/followup", response_model=FollowupResponse)
async def ask_followup(session_id: str, request: FollowupRequest) -> FollowupResponse:
    session = _get_session(session_id)
    if session["status"] != "complete":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Follow-up is available after diagnosis completion")
    disease = session["result"]["diagnosed_disease"]
    answer = await get_reasoning_service_client().answer_followup(disease, request.question)
    return FollowupResponse(answer=answer)
