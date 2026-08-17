from fastapi import APIRouter, HTTPException, Request, status

from backend.app.integrations.model_serving import ModelServingError

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("/performance")
async def model_performance(request: Request) -> dict:
    try:
        return await request.app.state.model_serving.get_model_registry()
    except (ModelServingError, NotImplementedError):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="model_service_unavailable")
