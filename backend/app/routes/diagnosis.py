from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from backend.app.integrations.model_serving import ModelServingError
from backend.app.integrations.reasoning_service import ReasoningServiceError
from backend.app.logic.decision import decide, diagnosed_disease, displayed_confidence
from backend.app.logic.fusion_merge import expand_candidates, merge_model_predictions
from backend.app.logic.info_gain import select_next_question
from backend.app.schemas.diagnosis import (
    AnswerRequest,
    AnswerResponse,
    DescribeRequest,
    DescribeResponse,
    DiagnosisResult,
    FollowupRequest,
    FollowupResponse,
    Question,
    Recommendations,
    StartResponse,
    StatusResponse,
)

router = APIRouter(prefix="/api/v1/diagnosis", tags=["diagnosis"])


async def _load_session(request: Request, session_id: str) -> dict:
    session = await request.app.state.session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")
    return session


async def _save_session(request: Request, session_id: str, session: dict) -> None:
    await request.app.state.session_store.set(session_id, session)


@router.post("/start", response_model=StartResponse)
async def start_diagnosis(
    request: Request,
    image: UploadFile = File(...),
    location: str | None = Form(default=None),
    crop: str | None = Form(default=None),
    growth_stage: str | None = Form(default=None),
) -> StartResponse:
    try:
        per_model_topk = await request.app.state.model_serving.predict(await image.read())
    except (ModelServingError, NotImplementedError):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="model_service_unavailable")

    initial_distribution = merge_model_predictions(per_model_topk, request.app.state.settings.fusion_weights)
    session_id = str(uuid4())
    session = {
        "status": "awaiting_description",
        "metadata": {"location": location, "crop": crop, "growth_stage": growth_stage},
        "per_model_topk": per_model_topk,
        "initial_distribution": initial_distribution,
        "candidate_distribution": initial_distribution,
        "symptom_evidence": {},
        "qa_evidence": [],
    }
    await _save_session(request, session_id, session)
    return StartResponse(session_id=session_id, status="awaiting_description", initial_distribution=initial_distribution)


@router.post("/{session_id}/describe", response_model=DescribeResponse)
async def describe_symptoms(session_id: str, payload: DescribeRequest, request: Request) -> DescribeResponse:
    session = await _load_session(request, session_id)
    if session["status"] != "awaiting_description":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="description_not_expected")

    observations = await request.app.state.llm_adapter.extract_observations(payload.description)
    try:
        matched_diseases = await request.app.state.reasoning_service.symptom_match(observations)
    except (ReasoningServiceError, NotImplementedError):
        matched_diseases = []

    candidate_distribution = expand_candidates(session["candidate_distribution"], matched_diseases)
    qa_knowledge = await request.app.state.reasoning_service.qa_knowledge(list(candidate_distribution))
    questions = qa_knowledge.get("questions", [])
    next_question = select_next_question(questions, candidate_distribution, set())
    if next_question is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="qa_knowledge_unavailable")

    session.update(
        {
            "status": "awaiting_answer",
            "description": payload.description,
            "observations": observations,
            "matched_diseases": matched_diseases,
            "candidate_distribution": candidate_distribution,
            "qa_knowledge": qa_knowledge,
            "current_question_id": next_question["question_id"],
        }
    )
    await _save_session(request, session_id, session)
    return DescribeResponse(
        session_id=session_id,
        status="awaiting_answer",
        observations=observations,
        candidate_distribution=candidate_distribution,
        question=Question(question_id=next_question["question_id"], text=next_question["text"]),
    )


@router.post("/{session_id}/answer", response_model=AnswerResponse)
async def answer_question(session_id: str, payload: AnswerRequest, request: Request) -> AnswerResponse:
    session = await _load_session(request, session_id)
    if session["status"] != "awaiting_answer":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="answer_not_expected")

    session["qa_evidence"].append(payload.model_dump())
    await _save_session(request, session_id, session)

    try:
        fusion = await request.app.state.reasoning_service.fuse(
            session["initial_distribution"],
            session.get("observations", {}),
            session["qa_evidence"],
        )
    except (ReasoningServiceError, NotImplementedError):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="reasoning_service_unavailable")

    answered_ids = {answer["question_id"] for answer in session["qa_evidence"]}
    next_question = select_next_question(
        session.get("qa_knowledge", {}).get("questions", []),
        session["candidate_distribution"],
        answered_ids,
    )
    should_finish = decide(fusion, request.app.state.settings) == "confirmed" or len(session["qa_evidence"]) >= request.app.state.settings.max_turns
    if not should_finish and next_question is not None:
        session["current_question_id"] = next_question["question_id"]
        session["last_fusion"] = fusion
        await _save_session(request, session_id, session)
        return AnswerResponse(
            session_id=session_id,
            status="awaiting_answer",
            question=Question(question_id=next_question["question_id"], text=next_question["text"]),
        )

    result = await _complete_result(request, fusion)
    session.update({"status": "complete", "last_fusion": fusion, "result": result.model_dump()})
    await _save_session(request, session_id, session)
    return AnswerResponse(session_id=session_id, status="complete", result=result)


@router.get("/{session_id}/result", response_model=DiagnosisResult)
async def get_result(session_id: str, request: Request) -> DiagnosisResult:
    session = await _load_session(request, session_id)
    if session["status"] != "complete":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="result_not_ready")
    return DiagnosisResult(**session["result"])


@router.get("/{session_id}/status", response_model=StatusResponse)
async def get_status(session_id: str, request: Request) -> StatusResponse:
    session = await _load_session(request, session_id)
    return StatusResponse(session_id=session_id, status=session["status"])


@router.post("/{session_id}/followup", response_model=FollowupResponse)
async def ask_followup(session_id: str, payload: FollowupRequest, request: Request) -> FollowupResponse:
    session = await _load_session(request, session_id)
    if session["status"] != "complete":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="followup_not_available")
    answer = await request.app.state.llm_adapter.answer_followup(session, payload.question)
    return FollowupResponse(answer=answer)


async def _complete_result(request: Request, fusion: dict) -> DiagnosisResult:
    disease = diagnosed_disease(fusion)
    confidence = displayed_confidence(fusion, disease)
    diagnosis = {"diagnosed_disease": disease, "confidence_score": confidence, "fusion": fusion}
    recommendation_note = None
    try:
        remedy_payload = await request.app.state.reasoning_service.get_remedies(disease)
        recommendations = Recommendations(**remedy_payload)
    except (ReasoningServiceError, NotImplementedError):
        recommendations = None
        recommendation_note = "recommendations_unavailable"

    explanation = await request.app.state.llm_adapter.generate_explanation(diagnosis, remedy_payload if recommendations else None)
    return DiagnosisResult(
        diagnosed_disease=disease,
        confidence_score=confidence,
        confidence_note="reasoning_service_fusion_conflict_adjusted",
        recommendations=recommendations,
        recommendation_note=recommendation_note,
        explanation=explanation,
    )
