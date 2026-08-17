import asyncio

PredictionDistribution = dict[str, float]


class ModelServingClientError(RuntimeError):
    pass


class ModelServingClient:
    async def predict(self, image_bytes: bytes) -> PredictionDistribution:
        """Return image-based disease probabilities from external model serving.

        This adapter boundary keeps the backend orchestrator-only: no CNN model
        artifacts are loaded in-process. The default implementation is a
        deterministic mock until the external model-serving contract is wired.
        """
        await asyncio.sleep(0)
        return {
            "Potato___Late_blight": 0.46,
            "Tomato___Late_blight": 0.31,
            "Alternaria_Solani": 0.23,
        }


def get_model_serving_client() -> ModelServingClient:
    return ModelServingClient()
