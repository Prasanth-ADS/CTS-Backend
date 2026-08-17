import json
from typing import Any, Protocol

from redis.asyncio import Redis

SessionData = dict[str, Any]


class SessionStore(Protocol):
    async def get(self, session_id: str) -> SessionData | None: ...
    async def set(self, session_id: str, data: SessionData) -> None: ...


class RedisSessionStore(SessionStore):
    def __init__(self, redis_url: str):
        self.redis = Redis.from_url(redis_url, decode_responses=True)

    async def get(self, session_id: str) -> SessionData | None:
        raw = await self.redis.get(_key(session_id))
        if raw is None:
            return None
        return json.loads(raw)

    async def set(self, session_id: str, data: SessionData) -> None:
        await self.redis.set(_key(session_id), json.dumps(data))


class InMemorySessionStore(SessionStore):
    def __init__(self):
        self.sessions: dict[str, SessionData] = {}

    async def get(self, session_id: str) -> SessionData | None:
        return self.sessions.get(session_id)

    async def set(self, session_id: str, data: SessionData) -> None:
        self.sessions[session_id] = data


def _key(session_id: str) -> str:
    return f"diagnosis:{session_id}"
