from fastapi import FastAPI

from backend.app.config import Settings, get_settings
from backend.app.integrations.llm_adapter import MockLLMAdapter, SLMAdapter
from backend.app.integrations.model_serving import HTTPModelServingClient, MockModelServingClient
from backend.app.integrations.reasoning_service import HTTPReasoningServiceClient, MockReasoningServiceClient
from backend.app.routes.diagnosis import router as diagnosis_router
from backend.app.routes.models import router as models_router
from backend.app.session import InMemorySessionStore, RedisSessionStore, SessionStore


def create_app(settings: Settings | None = None, session_store: SessionStore | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Plant Disease Diagnosis Orchestrator")
    app.state.settings = settings
    app.state.session_store = session_store or RedisSessionStore(settings.redis_url)
    app.state.model_serving = _build_model_serving(settings)
    app.state.reasoning_service = _build_reasoning_service(settings)
    app.state.llm_adapter = _build_llm_adapter(settings)
    app.include_router(diagnosis_router)
    app.include_router(models_router)

    @app.get("/status")
    async def status() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _build_model_serving(settings: Settings):
    if settings.use_mock_model_serving:
        return MockModelServingClient()
    if not settings.model_serving_url:
        raise ValueError("model_serving_url is required when use_mock_model_serving=false")
    return HTTPModelServingClient(settings.model_serving_url, settings.model_serving_timeout_seconds)


def _build_reasoning_service(settings: Settings):
    if settings.use_mock_reasoning_service:
        return MockReasoningServiceClient()
    if not settings.reasoning_service_url:
        raise ValueError("reasoning_service_url is required when use_mock_reasoning_service=false")
    return HTTPReasoningServiceClient(settings.reasoning_service_url, settings.reasoning_timeout_seconds)


def _build_llm_adapter(settings: Settings):
    if settings.use_mock_llm or not settings.llm_base_url:
        return MockLLMAdapter()
    return SLMAdapter(settings.llm_base_url, settings.llm_model_name)


app = create_app()
