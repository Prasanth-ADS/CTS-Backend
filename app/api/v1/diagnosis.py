from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.inference.ensemble import predict_ensemble
from app.reasoning.fusion import fuse_predictions
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
_MOCK_CANDIDATE_DISTRIBUTION = {
    "Potato___Late_blight": 0.51,
    "Tomato___Late_blight": 0.29,
    "Alternaria_Solani": 0.20,
}
_MOCK_QUESTIONS = [
    Question(question_id="q_water_soaked", text="Do you see water-soaked patches?"),
    Question(question_id="q_yellow_halo", text="Do the spots have a yellow halo?"),
]


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
        "initial_distribution": _MOCK_INITIAL_DISTRIBUTION,
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

    observations = {"brown_patches": True, "wet_lesions": True, "yellow_halo": None}
    session.update(
        {
            "status": "awaiting_answer",
            "description": request.description,
            "observations": observations,
            "candidate_distribution": _MOCK_CANDIDATE_DISTRIBUTION,
            "current_question_index": 0,
        }
    )
    return DescribeResponse(
        session_id=session_id,
        status="awaiting_answer",
        observations=observations,
        candidate_distribution=_MOCK_CANDIDATE_DISTRIBUTION,
        question=_MOCK_QUESTIONS[0],
    )


@router.post("/{session_id}/answer", response_model=AnswerResponse)
async def answer_question(session_id: str, request: AnswerRequest) -> AnswerResponse:
    session = _get_session(session_id)
    if session["status"] != "awaiting_answer":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Answer is not expected now")

    session["answers"].append(request.model_dump())
    next_index = session.get("current_question_index", 0) + 1
    if next_index < len(_MOCK_QUESTIONS):
        session["current_question_index"] = next_index
        return AnswerResponse(session_id=session_id, status="awaiting_answer", question=_MOCK_QUESTIONS[next_index])

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
