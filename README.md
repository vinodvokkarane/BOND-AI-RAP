---
title: BOND-AI-RAP
emoji: 🔬
colorFrom: teal
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: BOND-AI-RAP Reliability Assessment Platform
---

# BOND-AI-RAP Reliability Assessment Platform

BOND-AI-RAP is Version 2 of the BOND-AI portal: a reliability assessment platform for the integrated interface-and-bonding test coupon. It keeps the 500C-capable alumina coupon, Zone A interface structures, Zone B bonding structures, baseline Ag-NP ink, and ANI high-temperature conductive ink, then adds a reviewer-facing Reliability Assessment Platform layer inspired by SARA-style virtual qualification workflows.

The application generates large synthetic reliability datasets, trains tree-based surrogate models, and serves a tabbed Digital Twin that connects printed-interface degradation to bond-joint reliability, qualification confidence, physics-of-failure drivers, accelerated-test equivalence, and next-test recommendations.

## Coupon Platform

- Substrate: standardized 500C-capable alumina coupon.
- Inks: baseline Ag-NP ink and ANI 500C conductive ink.
- Zone A interface structures: straight lines, meander lines, square pads, and overlap pads.
- Zone B bonding structures: daisy-chain Kelvin structures, dummy die/chip attach sites, and shear test pads.
- Characterization: 4-point probe, Kelvin, I-V, thermal aging, thermal cycling, bending/strain, shear/pull, X-ray/CT, optical microscopy, SEM/EDS, and FIB.
- Representative conditions: 150C, 250C, 350C, and 500C aging; -40C to 125C and room-temperature to 500C cycling; static and cyclic strain; in-situ electrical monitoring.

## RAP V2 Tabs

- Virtual Qualification: PASS, MARGINAL, or DEFER decisions across candidate coupon recipes.
- Life-Cycle Profiles: package-relevant thermal, cycling, vibration, humidity, and mission exposure stages.
- Accelerated Equivalence: stress-condition equivalence to field hours with acceleration factors.
- Cycle Extractor: condensed thermal-cycle profile statistics and damage index.
- Physics Models: Arrhenius, Coffin-Manson, Black, void/delamination, shear-retention, and uncertainty model cards.
- Monte Carlo: reliability-score, shear-strength, RUL, and risk distributions.
- Failure-Site Map: coupon-level hot spots mapped to Zone A and Zone B structures.
- Sensitivity: ranked process, geometry, ink, and exposure drivers.
- Iso-Life Contours: temperature/time contour view for qualification planning.
- Test Planner: next best coupon experiments for active learning and uncertainty reduction.
- DfR What-If: design-for-reliability actions with estimated impact and cost.
- Validation: synthetic benchmark, holdout, coverage, and drift-monitor metrics.
- Qualification Report: executive-ready narrative suitable for proposal and review discussions.
- Model Plugins: extensible plugin slots for additional physics, inspection, and surrogate models.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/train_models.py --pattern-samples 80000 --interface-samples 40000 --coupon-samples 120000
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>.

The committed `model_artifacts/model_bundle.joblib` lets the app run without retraining during deployment.

## Deployment

For the full ML-enabled branch, use a Hugging Face Space with Docker.

- Space SDK: Docker
- Hardware: CPU basic
- App port: 7860
- Branch: main

The Dockerfile pins Python to `3.11-slim`, forces binary wheels for scientific packages, installs `requirements.txt`, and starts:

```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips='*'
```

## Data Notes

The default training run generates 240,000 synthetic rows: 80,000 print-characterization rows, 40,000 blind CPW validation rows, and 120,000 integrated-coupon rows. The committed `data/synthetic_preview.json`, `data/synthetic_summary.json`, and `model_artifacts/metrics.json` document generated distributions, target ranges, model benchmarks, and Digital Twin reliability metrics.

To recreate the full local parquet datasets:

```powershell
python scripts/train_models.py --write-full-data
```

Full parquet files are ignored by git to keep the deployable repository lightweight.
