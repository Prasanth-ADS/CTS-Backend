import asyncio
from io import BytesIO
from types import SimpleNamespace

from fastapi import UploadFile

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.routes.diagnosis import answer_question, ask_followup, describe_symptoms, get_result, start_diagnosis
from backend.app.schemas.diagnosis import AnswerRequest, DescribeRequest, FollowupRequest
from backend.app.session import InMemorySessionStore


def _request(app):
    return SimpleNamespace(app=app)


def test_full_mocked_diagnosis_flow_start_describe_answer_result():
    async def run_flow():
        app = create_app(
            Settings(redis_url="redis://unused", max_turns=3),
            session_store=InMemorySessionStore(),
        )
        request = _request(app)

        start = await start_diagnosis(
            request,
            image=UploadFile(filename="leaf.jpg", file=BytesIO(b"fake-image")),
            crop="potato",
        )
        assert start.status == "awaiting_description"
        assert start.session_id
        assert start.initial_distribution

        describe = await describe_symptoms(
            start.session_id,
            DescribeRequest(description="Leaves have brown spots and water-soaked patches."),
            request,
        )
        assert describe.status == "awaiting_answer"
        assert describe.observations
        assert describe.candidate_distribution
        assert describe.question.question_id

        current_question = describe.question
        answer = None
        for _ in range(3):
            answer = await answer_question(
                start.session_id,
                AnswerRequest(question_id=current_question.question_id, answer="yes"),
                request,
            )
            if answer.status == "complete":
                break
            current_question = answer.question

        assert answer is not None
        assert answer.status == "complete"
        assert answer.result is not None
        assert answer.result.diagnosed_disease == "Potato___Late_blight"
        assert answer.result.recommendations is not None
        assert answer.result.recommendations.management[0].title == "Remove infected leaves"

        result = await get_result(start.session_id, request)
        assert result == answer.result

        followup = await ask_followup(
            start.session_id,
            FollowupRequest(question="Can this spread?"),
            request,
        )
        assert followup.answer

    asyncio.run(run_flow())
