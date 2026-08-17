import asyncio
import math
import unittest
from unittest.mock import AsyncMock, patch

from app.core.model_registry import ModelCandidate, ModelRegistry
from app.inference import ensemble
from app.reasoning import fusion


_TEST_REGISTRY = ModelRegistry(
    active_ensemble=("alexnet_v1", "vgg16_v3", "resnet_v2"),
    fusion_strategy="weighted_average",
    candidates=(
        ModelCandidate("alexnet_v1", "s3://bucket/alexnet.pth", "alexnet", 0.91, 0.30),
        ModelCandidate("vgg16_v3", "s3://bucket/vgg16.pth", "vgg16", 0.94, 0.40),
        ModelCandidate("resnet_v2", "s3://bucket/resnet.pth", "resnet", 0.92, 0.30),
    ),
)


class EnsembleAndFusionTests(unittest.TestCase):
    def test_predict_ensemble_runs_active_models_concurrently(self):
        started: list[str] = []
        release = asyncio.Event()

        async def fake_run_model(candidate, image_bytes):
            started.append(candidate.id)
            if len(started) == 3:
                release.set()
            await asyncio.wait_for(release.wait(), timeout=1)
            return [(candidate.id, 1.0)]

        async def exercise():
            with (
                patch.object(ensemble, "get_model_registry", return_value=_TEST_REGISTRY),
                patch.object(ensemble, "run_model", new=AsyncMock(side_effect=fake_run_model)),
            ):
                return await ensemble.predict_ensemble(b"image")

        result = asyncio.run(exercise())

        self.assertEqual(set(result), {"alexnet_v1", "vgg16_v3", "resnet_v2"})
        self.assertEqual(set(started), {"alexnet_v1", "vgg16_v3", "resnet_v2"})

    def test_weighted_average_fusion_normalizes_union_of_topk(self):
        with patch.object(fusion, "get_model_registry", return_value=_TEST_REGISTRY):
            fused = fusion.fuse_predictions(
                {
                    "alexnet_v1": [("A", 0.7), ("B", 0.3)],
                    "vgg16_v3": [("B", 0.8), ("C", 0.2)],
                    "resnet_v2": [("A", 0.4), ("C", 0.6)],
                }
            )

        self.assertEqual(set(fused), {"A", "B", "C"})
        self.assertTrue(math.isclose(sum(fused.values()), 1.0))
        self.assertGreater(fused["B"], fused["C"])
