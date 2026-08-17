import asyncio
import unittest

from app.integrations.model_serving import ModelServingClient
from app.integrations.reasoning_service import ReasoningServiceClient
from app.schemas.diagnosis import DiagnosisMetadata


class OrchestratorAdapterTests(unittest.TestCase):
    def test_model_serving_client_returns_mock_distribution(self):
        distribution = asyncio.run(ModelServingClient().predict(b"image"))

        self.assertEqual(set(distribution), {"Potato___Late_blight", "Tomato___Late_blight", "Alternaria_Solani"})
        self.assertAlmostEqual(sum(distribution.values()), 1.0)

    def test_reasoning_service_expands_candidates_without_internal_kb_schema(self):
        distribution = asyncio.run(
            ReasoningServiceClient().expand_candidates(
                {"Potato___Late_blight": 0.6, "Tomato___Late_blight": 0.4},
                {"brown_patches": True},
                DiagnosisMetadata(crop="tomato"),
            )
        )

        self.assertEqual(set(distribution), {"Potato___Late_blight", "Tomato___Late_blight", "Alternaria_Solani"})
        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        self.assertGreater(distribution["Potato___Late_blight"], distribution["Alternaria_Solani"])
