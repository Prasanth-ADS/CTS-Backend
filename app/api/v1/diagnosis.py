from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.inference.ensemble import predict_ensemble
from app.integrations.team2_kb import fetch_qa_knowledge, match_symptoms
from app.llm.symptom_extraction import extract_observations
from app.reasoning.candidates import expand_candidate_distribution
from app.reasoning.fusion import fuse_predictions
from app.reasoning.info_gain import select_next_question
from app.schemas.diagnosis import (
    AnswerRequest,
    AnswerResponse,
    DescribeRequest,
    DescribeResponse,
    DiagnosisMetadata,
    DiagnosisResult,
    FollowupRequest,
    FollowupResponse,
    Question,
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
    per_model_topk = await predict_ensemble(image_bytes)
    initial_distribution = fuse_predictions(per_model_topk)

    session_id = str(uuid4())
    metadata = DiagnosisMetadata(location=location, crop=crop, growth_stage=growth_stage)
    _SESSIONS[session_id] = {
        "status": "awaiting_description",
        "image_filename": image.filename,
        "metadata": metadata.model_dump(),
        "per_model_topk": per_model_topk,
        "initial_distribution": initial_distribution,
        "answers": [],
    }
    return StartResponse(
        session_id=session_id,
        status="awaiting_description",
        initial_distribution=initial_distribution,
    )


@router.post("/{session_id}/describe", response_model=DescribeResponse)
async def describe_symptoms(session_id: str, request: DescribeRequest) -> DescribeResponse:
    session = _get_session(session_id)
    if session["status"] != "awaiting_description":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Description is not expected now")

    observations = extract_observations(request.description)
    matched_diseases = await match_symptoms(observations)
    candidate_distribution = expand_candidate_distribution(session["initial_distribution"], matched_diseases)
    qa_knowledge = await fetch_qa_knowledge(list(candidate_distribution))
    next_question = select_next_question(qa_knowledge, candidate_distribution, set())
    if next_question is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No QA knowledge available")

    session.update(
        {
            "status": "awaiting_answer",
            "description": request.description,
            "observations": observations,
            "matched_diseases": matched_diseases,
            "candidate_distribution": candidate_distribution,
            "qa_knowledge": qa_knowledge,
            "current_question_id": next_question["question_id"],
        }
    )
    return DescribeResponse(
        session_id=session_id,
        status="awaiting_answer",
        observations=observations,
        candidate_distribution=candidate_distribution,
        question=Question(question_id=next_question["question_id"], text=next_question["text"]),
    )


@router.post("/{session_id}/answer", response_model=AnswerResponse)
async def answer_question(session_id: str, request: AnswerRequest) -> AnswerResponse:
    session = _get_session(session_id)
    if session["status"] != "awaiting_answer":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Answer is not expected now")

    session["answers"].append(request.model_dump())
    answered_question_ids = {answer["question_id"] for answer in session["answers"]}
    next_question = select_next_question(
        session.get("qa_knowledge", []),
        session["candidate_distribution"],
        answered_question_ids,
    )
    if next_question is not None:
        session["current_question_id"] = next_question["question_id"]
        return AnswerResponse(
            session_id=session_id,
            status="awaiting_answer",
            question=Question(question_id=next_question["question_id"], text=next_question["text"]),
        )

    result = DiagnosisResult(
        diagnosed_disease="Potato___Late_blight",
        confidence_score=0.82,
        confidence_note="mock_weighted_average_dst_conflict_adjusted",
        management=["Remove heavily infected leaves", "Avoid overhead irrigation"],
        prevention=["Use certified disease-free seed", "Improve field air circulation"],
        precautions=["Confirm with an agricultural specialist before chemical treatment"],
        references=["https://example.org/team-2-kb/potato-late-blight"],
        explanation="Mock result: ensemble prediction plus farmer evidence currently points to Potato late blight.",
    )
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
    return FollowupResponse(answer=f"Mock SLM answer for {disease}: this should be verified against Team 2 KB guidance.")
