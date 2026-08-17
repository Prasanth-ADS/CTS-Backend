from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.session import InMemorySessionStore


def test_full_mocked_diagnosis_flow_start_describe_answer_result():
    app = create_app(
        Settings(redis_url="redis://unused", max_turns=3),
        session_store=InMemorySessionStore(),
    )
    client = TestClient(app)

    start = client.post(
        "/api/v1/diagnosis/start",
        files={"image": ("leaf.jpg", b"fake-image", "image/jpeg")},
        data={"crop": "potato"},
    )
    assert start.status_code == 200
    start_payload = start.json()
    assert start_payload["status"] == "awaiting_description"
    assert start_payload["session_id"]
    assert start_payload["initial_distribution"]

    session_id = start_payload["session_id"]
    describe = client.post(
        f"/api/v1/diagnosis/{session_id}/describe",
        json={"description": "Leaves have brown spots and water-soaked patches."},
    )
    assert describe.status_code == 200
    describe_payload = describe.json()
    assert describe_payload["status"] == "awaiting_answer"
    assert describe_payload["observations"]
    assert describe_payload["candidate_distribution"]
    assert describe_payload["question"]["question_id"]

    current = describe_payload
    for _ in range(3):
        answer = client.post(
            f"/api/v1/diagnosis/{session_id}/answer",
            json={"question_id": current["question"]["question_id"], "answer": "yes"},
        )
        assert answer.status_code == 200
        current = answer.json()
        if current["status"] == "complete":
            break

    assert current["status"] == "complete"
    assert current["result"]["diagnosed_disease"] == "Potato___Late_blight"
    assert current["result"]["recommendations"]["management"][0]["title"] == "Remove infected leaves"

    result = client.get(f"/api/v1/diagnosis/{session_id}/result")
    assert result.status_code == 200
    assert result.json() == current["result"]

    followup = client.post(
        f"/api/v1/diagnosis/{session_id}/followup",
        json={"question": "Can this spread?"},
    )
    assert followup.status_code == 200
    assert followup.json()["answer"]
