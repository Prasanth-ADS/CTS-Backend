from backend.app.integrations.protocols import ModelServingClient


class ModelServingError(RuntimeError):
    pass


class MockModelServingClient(ModelServingClient):
    async def predict(self, image_bytes: bytes) -> dict[str, list[tuple[str, float]]]:
        return {
            "alexnet_v1": [("Potato___Late_blight", 0.72), ("Potato___Early_blight", 0.18), ("Tomato___Late_blight", 0.10)],
            "vgg16_v3": [("Potato___Late_blight", 0.61), ("Tomato___Late_blight", 0.24), ("Potato___Early_blight", 0.15)],
        }

    async def get_model_registry(self) -> dict:
        return {
            "active_ensemble": ["alexnet_v1", "vgg16_v3"],
            "candidates": [
                {"id": "alexnet_v1", "eval_accuracy": 0.91},
                {"id": "vgg16_v3", "eval_accuracy": 0.94},
            ],
        }


class HTTPModelServingClient(ModelServingClient):
    def __init__(self, base_url: str, timeout_seconds: float = 5.0):
        import httpx

        self.client = httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds)

    async def predict(self, image_bytes: bytes) -> dict[str, list[tuple[str, float]]]:
        raise NotImplementedError("Wire up once ML team's contract is confirmed")

    async def get_model_registry(self) -> dict:
        raise NotImplementedError("Wire up once ML team's contract is confirmed")
