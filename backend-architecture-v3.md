# Adaptive AI Disease Diagnosis — Backend Architecture (v3)

Revision addressing team lead review. Changes from v2 are marked **[CHANGED]**.
Everything else carries over unchanged — see v2 doc for full detail on model
serving mechanics, session schema, and deployment.

---

## 1. Model management **[CHANGED — Point 1]**

No hardcoded model count anywhere in code or docs. Candidates are entries in a
config/registry, evaluated and swapped without touching the pipeline.

```yaml
# config/models.yaml — one entry per benchmarked model, count is whatever it is
active_model: vgg16_v3          # the one currently deployed
candidates:
  - id: alexnet_v1
    path: s3://bucket/models/alexnet_v1.pth
    arch: alexnet
    eval_accuracy: 0.91
  - id: vgg16_v3
    path: s3://bucket/models/vgg16_v3.pth
    arch: vgg16
    eval_accuracy: 0.94
  # ...as many as the current benchmark has
```
`ModelManager` reads `active_model` at startup and loads only that one into memory.
Swapping the deployed model = changing one field and redeploying, never a code change.
This same registry is what powers the model-performance endpoint in §6.

---

## 2. Uncertainty / candidate set module — explicitly labeled MVP **[CHANGED — Point 2]**

Renamed and documented so it can't be mistaken for calibrated conformal prediction
anywhere downstream (code, API response, dashboard):

```python
# reasoning/candidate_set_mvp.py
"""
MVP CANDIDATE-SET METHOD — NOT FORMAL CONFORMAL PREDICTION.

This uses simple cumulative-softmax-probability thresholding. It gives no
statistical coverage guarantee. True conformal prediction requires a held-out
calibration set and a defined non-conformity score — not implemented yet.
Replace this module (same function signature) when that's ready.
"""
def mvp_candidate_set(top3: list[tuple[str, float]], threshold=0.85) -> list[str]:
    ...
```
The API response also tags this explicitly (see §5) so the frontend/dashboard can
render a disclaimer rather than presenting it as calibrated uncertainty.

---

## 3. Dempster-Shafer fusion — conflict-tracked, confidence decoupled **[CHANGED — Point 6]**

Two additions vs. v2: (a) conflict (K) is computed and returned, not discarded;
(b) raw belief mass is never directly shown to the user as "confidence" — a
separate, explicit mapping function decides what gets displayed.

```python
# reasoning/dst_fusion.py
Evidence = dict[str, float]

class FusionResult(TypedDict):
    belief: Evidence      # raw fused belief mass per disease — internal use
    conflict: float        # K: degree of conflict between combined sources (0-1)

def combine(m1: Evidence, m2: Evidence) -> FusionResult:
    """Dempster's rule. Returns fused belief AND the conflict mass K
    (mass assigned to the empty set before renormalization). High K means
    the CNN evidence and symptom evidence substantially disagreed —
    this should visibly affect how confident the final answer looks."""
    ...

def fuse_all(cnn_evidence: Evidence, symptom_evidence: list[Evidence]) -> FusionResult:
    ...
```

```python
# reasoning/confidence.py
"""
Displayed confidence is DERIVED from DST belief + conflict — it is NOT the
raw belief mass. This function is the single place that decides what number
the user sees, so the mapping can be tuned/recalibrated independently of the
fusion math itself.
"""
def displayed_confidence(fusion: FusionResult, diagnosed_disease: str) -> float:
    raw = fusion["belief"][diagnosed_disease]
    conflict_penalty = fusion["conflict"]   # e.g. reduce displayed confidence as conflict rises
    return clamp(raw * (1 - conflict_penalty), 0.0, 1.0)
```
This keeps "what the math produces" and "what the user sees" as two explicit,
separately-testable steps — which is exactly what avoids silently overstating
confidence.

---

## 4. Knowledge Base + Recommendation Engine — owned by Team 2 **[CHANGED — Point 7, major]**

This is the one structural change. We no longer own KB tables or recommendation
ranking logic. Backend becomes a **client** of Team 2's Recommendation API.

```
┌──────────────────┐   HTTP    ┌───────────────────────────┐
│ Our backend        │ ────────► │ Team 2: Recommendation API │
│ (diagnosis result)  │ ◄──────── │ + Knowledge Base            │
└──────────────────┘           └───────────────────────────┘
```

Proposed contract to align with Team 2 (adjust once they confirm their actual API):
```
GET /recommendations?disease={diagnosed_disease}
→ {
    "management_recommendations": [ { "title": "...", "priority": 1 }, ... ],
    "prevention_strategies": [ { "title": "...", "priority": 1 }, ... ],
    "references": ["https://..."]
  }
```
Our backend calls this once the DST decision (§3) is final, merges the response
into the `/answer` result payload alongside the LLM explanation. If Team 2's API
is down, we return `503` with `error: "recommendations_unavailable"` rather than
guessing — the diagnosis itself is still valid and can be shown separately.

**Postgres usage narrows accordingly [Point 9]**: our own Postgres, if used, holds
only backend-owned data — session logs, model registry metadata, question bank
(§ from v2, since that's diagnosis-pipeline logic, not KB content). No disease/
remedy/reference tables on our side; those live in Team 2's database.

---

## 5. Diagnosis API — contract unchanged, one field added

Same endpoints and shapes as v2 (`/start`, `/answer`, `/result`, `/status`).
Two additions to reflect §2 and §4:

```json
// POST /answer — status: complete
{
  "session_id": "d3f9a2b1-...",
  "status": "complete",
  "result": {
    "diagnosed_disease": "Early Blight",
    "confidence_score": 0.83,              // from displayed_confidence(), not raw belief
    "confidence_note": "mvp_candidate_set", // flags which uncertainty method produced the candidate set
    "management_recommendations": [ ... ], // from Team 2's Recommendation API
    "prevention_strategies": [ ... ],
    "references": [ ... ],
    "explanation": "Based on the leaf pattern..."
  }
}
```

### New: follow-up Q&A **[NEW — Point 5 extended]**
```
POST /api/v1/diagnosis/{session_id}/followup
Body: { "question": "Is this contagious to nearby plants?" }
→ { "answer": "Yes, Early Blight can spread to nearby plants via..." }
```
Live SLM call, given the session's diagnosis + explanation as context. Available
only after `status: complete`. Same SLM adapter as the explanation step (§5 in v2)
— no new model, just a second live use of it.

---

## 6. Model-performance API for researcher dashboard **[NEW — Point 11]**

Separate from the diagnosis pipeline entirely — reads straight from the model
registry (§1), not from any live session:

```
GET /api/v1/models/performance
→ {
    "active_model": "vgg16_v3",
    "candidates": [
      { "id": "alexnet_v1", "arch": "alexnet", "eval_accuracy": 0.91 },
      { "id": "vgg16_v3",   "arch": "vgg16",   "eval_accuracy": 0.94 }
      // ...one entry per benchmarked model, whatever the current count is
    ]
  }
```
Static/config-backed for now (reads `config/models.yaml`); can move to a proper
table later if the dashboard needs history over time (e.g. accuracy per retrain).
No auth on the diagnosis endpoints, but this one probably wants to be gated
separately since it's internal-facing — flag that for discussion with your lead.

---

## 7. What to confirm with Team 2 before building §4

- Exact request/response shape of their Recommendation API (the contract above is
  a proposal, not confirmed)
- Whether they key recommendations by disease name (string) or a shared disease ID
  — worth agreeing on a shared enum/ID scheme now so both sides don't drift
- Their expected latency/SLA, since it's now a dependency in our critical path for
  finishing a diagnosis
- Auth between services (even with no end-user auth, service-to-service might need
  an API key)

---

## 8. Updated build order

1. Docker Compose + FastAPI skeleton, `/start`/`/answer`/`/followup` mocked exactly
   to §5 shapes — frontend unblocked immediately, including the new follow-up endpoint.
2. Model registry (§1) + real CNN inference.
3. MVP candidate-set module (§2), clearly labeled.
4. Fixed question bank + info gain (unchanged from v2).
5. Offline SLM question-phrasing batch job (unchanged from v2).
6. DST fusion with conflict tracking + separate confidence mapping (§3).
7. Integrate Team 2's Recommendation API once their contract is confirmed (§4/§7).
8. Live SLM: explanation + follow-up Q&A (§5).
9. Model-performance endpoint (§6) — can be built any time after §1, it's independent.
