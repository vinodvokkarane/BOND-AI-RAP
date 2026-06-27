from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mlflex.modeling import digital_twin_feedback, load_bundle, optimize_process, predict_coupon, predict_interface, predict_pattern
from mlflex.rap import rap_assessment
from mlflex.synthetic import default_coupon_payload, default_interface_payload, default_pattern_payload, metadata

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"
ARTIFACT_DIR = ROOT / "model_artifacts"


class Payload(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(
    title="BOND-AI-RAP Reliability Assessment Platform",
    version="2.0.0",
    description=(
        "Version 2 reliability assessment platform for virtual qualification, physics-of-failure "
        "analysis, active learning, and Digital Twin feedback on 500C alumina coupons."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_bundle: dict[str, Any] | None = None


def bundle() -> dict[str, Any]:
    global _bundle
    if _bundle is None:
        artifact = ARTIFACT_DIR / "model_bundle.joblib"
        if not artifact.exists():
            raise HTTPException(
                status_code=503,
                detail="Model artifacts are missing. Run `python scripts/train_models.py` before starting the app.",
            )
        _bundle = load_bundle(ARTIFACT_DIR)
    return _bundle


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    artifact = ARTIFACT_DIR / "model_bundle.joblib"
    return {"ok": artifact.exists(), "artifact": str(artifact.name)}


@app.get("/api/metadata")
def api_metadata() -> dict[str, Any]:
    data = metadata()
    data["portal_name"] = "BOND-AI-RAP"
    data["platform"]["title"] = "Reliability Assessment Platform"
    data["defaults"] = {
        "pattern": default_pattern_payload(),
        "interface": default_interface_payload(),
        "coupon": default_coupon_payload(),
    }
    try:
        data["metrics"] = bundle()["metrics"]
    except HTTPException:
        data["metrics"] = None
    data["research_sources"] = [
        {
            "label": "BOND-AI proposal: integrated interface and bonding coupon methodology",
            "url": "",
        },
        {
            "label": "Zone A: sheet resistance drift, crack/fatigue, adhesion, and delamination descriptors",
            "url": "",
        },
        {
            "label": "Zone B: Kelvin contact drift, die/chip attach voiding, shear, and pull strength descriptors",
            "url": "",
        },
        {
            "label": "Physics priors: Arrhenius aging, fatigue/cycling damage, void growth, and interface reaction layers",
            "url": "",
        },
        {
            "label": "RAP V2: virtual qualification, acceleration factors, Monte Carlo, sensitivity, failure maps, and report export",
            "url": "",
        },
    ]
    return data


@app.post("/api/predict/pattern")
def api_predict_pattern(payload: Payload) -> dict[str, Any]:
    return predict_pattern(bundle(), payload.values)


@app.post("/api/predict/interface")
def api_predict_interface(payload: Payload) -> dict[str, Any]:
    return predict_interface(bundle(), payload.values)


@app.post("/api/optimize")
def api_optimize(payload: Payload) -> dict[str, Any]:
    count = int(payload.values.pop("candidates", 700))
    return optimize_process(bundle(), payload.values, candidates=max(100, min(2_500, count)))


@app.post("/api/predict/coupon")
def api_predict_coupon(payload: Payload) -> dict[str, Any]:
    return predict_coupon(bundle(), payload.values)


@app.post("/api/digital-twin/feedback")
def api_digital_twin_feedback(payload: Payload) -> dict[str, Any]:
    count = int(payload.values.pop("candidates", 600))
    return digital_twin_feedback(bundle(), payload.values, candidates=max(120, min(2_500, count)))


@app.post("/api/rap/assessment")
def api_rap_assessment(payload: Payload) -> dict[str, Any]:
    values = dict(payload.values)
    count = int(values.pop("candidates", 360))
    samples = int(values.pop("samples", 260))
    seed = int(values.pop("seed", 2026))
    return rap_assessment(
        bundle(),
        values,
        candidates=max(160, min(900, count)),
        samples=max(80, min(800, samples)),
        seed=seed,
    )
