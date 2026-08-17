from fastapi import FastAPI

from app.api.v1.diagnosis import router as diagnosis_router
from app.core.config import get_settings

settings = get_settings()
app = FastAPI(title=settings.app_name)

app.include_router(diagnosis_router)


@app.get("/status")
async def status() -> dict[str, str]:
    return {"status": "ok"}
