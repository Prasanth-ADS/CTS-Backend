# Adaptive AI Disease Diagnosis — Backend Architecture (v4)

Revision for the ensemble + farmer-interaction diagram. Changes from v3 marked **[CHANGED]** / **[NEW]**. Carries over unchanged from v3: MVP-labeled uncertainty module concept (now folded into fusion, see §2), conflict-aware DST base (extended in §4), Redis session state, no-auth diagnosis endpoints.

---

## 1. Ensemble model management **[CHANGED]**

Model registry (§1 in v3) now lists an **active ensemble**, not a single active model. All listed models run per request.

```yaml
# config/models.yaml
active_ensemble: [alexnet_v1, vgg16_v3, resnet_v2]
fusion_strategy: weighted_average
candidates:
  - id: alexnet_v1
    path: s3://bucket/models/alexnet_v1.pth
    arch: alexnet
    eval_accuracy: 0.91
    fusion_weight: 0.30
  - id: vgg16_v3
    path: s3://bucket/models/vgg16_v3.pth
    arch: vgg16
    eval_accuracy: 0.94
    fusion_weight: 0.40
  - id: resnet_v2
    path: s3://bucket/models/resnet_v2.pth
    arch: resnet
    eval_accuracy: 0.92
    fusion_weight: 0.30
```

- Models load into memory once at startup.
- Run in parallel per request using `asyncio`/threadpool execution.
- The prediction fusion module is swappable by configuration.

```python
# inference/ensemble.py
async def predict_ensemble(image: PIL.Image) -> dict[str, list[tuple[str, float]]]:
    """Runs all active models concurrently, returns each model's raw top-K."""
    results = await asyncio.gather(*(
        run_model(model_id, image) for model_id in ACTIVE_ENSEMBLE
    ))
    return dict(zip(ACTIVE_ENSEMBLE, results))
```

## 2. Prediction fusion **[NEW module]**

```python
# reasoning/fusion.py
"""
FUSION STRATEGY: weighted_average (config-selectable).
Takes the union of each model's top-K, weights by configured fusion_weight
(default: proportional to eval_accuracy), normalizes to a distribution.
Alternative strategy (rank_aggregation) can be swapped in via config —
same input/output shape.
"""
def fuse_predictions(per_model_topk: dict[str, list[tuple[str, float]]]) -> dict[str, float]:
    ...
```

This output is the fused initial disease distribution and starting belief before farmer interaction.

---

## 3. Farmer free-text interaction — live SLM, once per session **[NEW]**

```python
# llm/symptom_extraction.py
"""
Live SLM call — runs once per session when the farmer submits a free-text
description. NOT part of the adaptive question loop; that loop still uses
pre-phrased questions only.
"""
def extract_observations(description: str) -> dict[str, bool | None]:
    ...
```

---

## 4. KB symptom matching + candidate expansion **[NEW — Team 2 dependency]**

Extracted observations get sent to Team 2's KB to find matching diseases, including diseases outside the ensemble's original top-K.

```http
POST {team2_kb_url}/symptom-match
Body: { "observations": { "brown_patches": true, "wet_lesions": true } }
→ { "matched_diseases": ["Potato___Late_blight", "Tomato___Late_blight", "Alternaria_Solani"] }
```

Our backend merges this with the fused ensemble distribution into an expanded, renormalized candidate set and prior distribution.

---

## 5. Information gain loop — unchanged in principle, data source clarified **[CHANGED — ownership]**

The question bank and per-disease support/weight values now come from Team 2's KB, not our own Postgres tables.

```http
GET {team2_kb_url}/qa-knowledge?diseases=Potato___Late_blight,Tomato___Late_blight
→ {
  "questions": [
    {
      "question_id": "q_water_soaked",
      "text": "Do you see water-soaked patches?",
      "support": {
        "Potato___Late_blight": {"yes": 0.90, "no": 0.10, "not_sure": 0.50}
      }
    }
  ]
}
```

Fetched once per session after candidate expansion and cached in the session object.

---

## 6. Evidence computation + DST fusion — extended output **[CHANGED]**

DST fusion combines three evidence sources:

1. Ensemble belief
2. Farmer-symptom evidence
3. Per-turn Q&A evidence accumulated across the loop

```python
# reasoning/dst_fusion.py
class FusionResult(TypedDict):
    belief: dict[str, float]
    plausibility: dict[str, float]
    uncertainty: dict[str, float]
    conflict: float

def fuse_all(ensemble_belief, symptom_evidence, qa_evidence_list) -> FusionResult:
    ...
```

`displayed_confidence()` remains the single mapping from fusion output to user-facing confidence.

---

## 7. Decision engine — explicit, configurable thresholds **[NEW, formalized]**

```python
# reasoning/decision.py
DECISION_THRESHOLDS = {
    "min_belief": 0.70,
    "min_margin": 0.20,
    "max_conflict": 0.30,
}

def decide(fusion: FusionResult) -> Literal["confirmed", "uncertain"]:
    ranked = sorted(fusion["belief"].items(), key=lambda x: -x[1])
    best, second = ranked[0], ranked[1] if len(ranked) > 1 else (None, 0)
    if (best[1] >= DECISION_THRESHOLDS["min_belief"]
        and best[1] - second[1] >= DECISION_THRESHOLDS["min_margin"]
        and fusion["conflict"] <= DECISION_THRESHOLDS["max_conflict"]):
        return "confirmed"
    return "uncertain"
```

`uncertain` loops back to the information-gain question flow. `confirmed` proceeds to remedy fetch and final response. `max_turns` caps the loop.

---

## 8. Remedy fetch — confirmed as a Team 2 API call

```http
GET {team2_kb_url}/remedies?disease=Potato___Late_blight
→ { "management": [...], "prevention": [...], "precautions": [...], "references": [...] }
```

This keeps remedy and KB ownership with Team 2.

---

## 9. Full endpoint list (frontend-facing)

| # | Endpoint | Trigger / does |
|---|---|---|
| 1 | `POST /api/v1/diagnosis/start` | image + optional metadata → ensemble + fusion → initial distribution, `status: awaiting_description` |
| 2 | `POST /api/v1/diagnosis/{id}/describe` | farmer free-text → SLM extraction + KB symptom-match + candidate expansion + first IG question → `status: awaiting_answer` |
| 3 | `POST /api/v1/diagnosis/{id}/answer` | answer → evidence + DST fusion + decision engine → next question or `status: complete` |
| 4 | `GET /api/v1/diagnosis/{id}/result` | fetch completed result |
| 5 | `GET /api/v1/diagnosis/{id}/status` | polling for loading state |
| 6 | `POST /api/v1/diagnosis/{id}/followup` | live SLM Q&A, post-diagnosis only |
| 7 | `GET /api/v1/models/performance` | researcher dashboard — ensemble registry + eval accuracy |

Outbound Team 2 KB calls: `POST /symptom-match`, `GET /qa-knowledge`, `GET /remedies`.

Open UX decision: keep `/start` and `/describe` separate or merge image and farmer description into one `/start` request.

---

## 10. Updated build order

1. Docker Compose + FastAPI skeleton, all 7 endpoints mocked to §9 shapes.
2. Ensemble inference running in parallel + fusion module.
3. Live SLM extraction wired into `/describe`.
4. Team 2 KB symptom-match integration — blocked on their contract.
5. Team 2 KB Q&A-knowledge integration — blocked on their contract.
6. Evidence computation + 3-source DST fusion.
7. Decision engine with configurable thresholds.
8. Team 2 remedies integration + live SLM explanation.
9. Follow-up Q&A endpoint.
10. Model-performance endpoint — independent after step 2.
