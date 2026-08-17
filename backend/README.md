# Plant Disease Diagnosis Orchestrator Backend

This backend owns the React-facing API and diagnosis session state. It does not own model weights, CNN inference, DST fusion math, symptom/question/remedy content, or KB ranking. Those external responsibilities are represented by swappable adapter protocols with mock implementations so the orchestrator can run end-to-end today.

## Running locally

```bash
docker compose up --build
```

The compose file starts the FastAPI API and Redis.

## Swapping in real integrations

1. Get the real base URL from the other team.
2. Implement the `HTTP*Client`'s methods (currently `NotImplementedError` stubs).
3. Set `use_mock_*=false` and the corresponding `*_url` in env config.
4. No other code changes needed — endpoints, schemas, and session logic are already integration-agnostic.
