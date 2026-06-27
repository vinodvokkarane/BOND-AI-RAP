from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .modeling import _bond_ai_decision, _coupon_exposure, _rul_hours, digital_twin_feedback, predict_coupon
from .synthetic import COUPON_FEATURES, COUPON_STRUCTURES, COUPON_TARGETS, complete_coupon_payload


BOLTZMANN_EV_K = 8.617333262e-5

PHYSICS_MODEL_LIBRARY: list[dict[str, Any]] = [
    {
        "id": "arrhenius_thermal_aging",
        "name": "Arrhenius Thermal Aging",
        "family": "Thermal",
        "equation": "AF = exp(Ea/k * (1/Tuse - 1/Ttest))",
        "inputs": ["activation energy", "use temperature", "test temperature", "exposure time"],
        "bond_ai_use": "Normalizes 150C to 500C aging conditions for sheet/contact drift and oxidation.",
    },
    {
        "id": "coffin_manson_fatigue",
        "name": "Coffin-Manson Fatigue",
        "family": "Thermo-mechanical",
        "equation": "Nf proportional to strain_range^-c",
        "inputs": ["temperature swing", "strain range", "cycle count", "fatigue exponent"],
        "bond_ai_use": "Converts thermal and mechanical cycling into crack/fatigue damage for meanders and overlap pads.",
    },
    {
        "id": "black_electromigration",
        "name": "Black Electromigration",
        "family": "Electrical",
        "equation": "MTTF proportional to J^-n * exp(Ea/kT)",
        "inputs": ["current density", "temperature", "activation energy", "current exponent"],
        "bond_ai_use": "Flags conductive traces where resistance drift may accelerate under in-situ monitoring.",
    },
    {
        "id": "void_delamination_growth",
        "name": "Void and Delamination Growth",
        "family": "Structural",
        "equation": "damage = f(void_fraction, thermal_dose, interface_roughness)",
        "inputs": ["CT void fraction", "thermal dose", "edge roughness", "alignment error"],
        "bond_ai_use": "Links X-ray/CT and optical morphology to bond-joint reliability.",
    },
    {
        "id": "shear_strength_retention",
        "name": "Post-Aging Shear Retention",
        "family": "Mechanical",
        "equation": "retention = f(adhesion, voids, thermal dose, cycling dose)",
        "inputs": ["adhesion strength", "void fraction", "aging profile", "thermal cycles"],
        "bond_ai_use": "Ranks dummy die attach and shear pads after aging/cycling.",
    },
    {
        "id": "conformal_uncertainty",
        "name": "Conformal Reliability Bounds",
        "family": "Uncertainty",
        "equation": "prediction +/- empirical q90 residual",
        "inputs": ["model residuals", "current prediction", "target metric"],
        "bond_ai_use": "Adds confidence-aware PASS/MARGINAL/DEFER decisions.",
    },
]


SITE_COORDINATES: dict[str, dict[str, float]] = {
    "straight_lines": {"x": 18, "y": 36},
    "meander_lines": {"x": 30, "y": 58},
    "square_pads": {"x": 43, "y": 41},
    "overlap_pads": {"x": 52, "y": 63},
    "daisy_chain_kelvin": {"x": 66, "y": 38},
    "dummy_die_attach": {"x": 78, "y": 58},
    "shear_test_pads": {"x": 90, "y": 45},
}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _arrhenius_factor(use_temp_c: float, test_temp_c: float, activation_ev: float = 0.72) -> float:
    use_k = max(use_temp_c + 273.15, 1.0)
    test_k = max(test_temp_c + 273.15, 1.0)
    return float(np.clip(math.exp(activation_ev / BOLTZMANN_EV_K * (1.0 / use_k - 1.0 / test_k)), 0.01, 1_000_000.0))


def _predict_rows(bundle: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    completed = [complete_coupon_payload(row) for row in rows]
    X = pd.DataFrame(completed, columns=COUPON_FEATURES)
    pred = bundle["coupon"]["regressor"].predict(X)
    prob = bundle["coupon"]["classifier"].predict_proba(X)
    classes = list(bundle["coupon"]["classifier"].named_steps["model"].classes_)
    results: list[dict[str, Any]] = []
    for idx, row in enumerate(completed):
        prediction = {target: float(pred[idx, col]) for col, target in enumerate(COUPON_TARGETS)}
        probabilities = {classes[col]: float(prob[idx, col]) for col in range(len(classes))}
        exposure = _coupon_exposure(row)
        decision = _bond_ai_decision(prediction, probabilities)
        results.append(
            {
                "input": row,
                "prediction": prediction,
                "probabilities": probabilities,
                "qualification_decision": decision["qualification_decision"],
                "failure_mode": max(probabilities.items(), key=lambda item: item[1])[0],
                "derived": {
                    **exposure,
                    "remaining_useful_life_hours": _rul_hours(prediction, exposure),
                    "degradation_state_index": decision["degradation_state_index"],
                    "failure_risk_index": decision["failure_risk_index"],
                    "decision_confidence_pct": decision["confidence_pct"],
                },
            }
        )
    return results


def _virtual_qualification(baseline: dict[str, Any]) -> dict[str, Any]:
    pred = baseline["prediction"]
    derived = baseline["derived"]
    intervals = baseline.get("intervals", {})
    margins = {
        "reliability_margin": float(pred["reliability_score"] - 80.0),
        "contact_drift_margin": float(48.0 - pred["contact_resistance_drift_pct"]),
        "delamination_margin": float(24.0 - pred["delamination_area_pct"]),
        "void_margin": float(8.5 - pred["void_fraction_pct"]),
        "shear_margin": float(pred["post_aging_shear_mpa"] - 14.0),
        "rul_margin_hours": float(derived["remaining_useful_life_hours"] - 250.0),
    }
    lowest = sorted(margins.items(), key=lambda item: item[1])[:3]
    return {
        "decision": baseline["qualification_decision"],
        "confidence_pct": derived["decision_confidence_pct"],
        "reliability_score": pred["reliability_score"],
        "remaining_useful_life_hours": derived["remaining_useful_life_hours"],
        "failure_mode": baseline["failure_mode"],
        "margins": margins,
        "critical_margins": [{"metric": key, "margin": value} for key, value in lowest],
        "q90_reliability": intervals.get("reliability_score", {}),
        "reviewer_readout": (
            "Virtual qualification links the chosen coupon structure, ink family, and mission profile "
            "to a confidence-qualified reliability decision before committing lab coupons."
        ),
    }


def _life_cycle_profile(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    exposure = baseline["derived"]
    segments = [
        {
            "name": "Thermal aging",
            "condition": f"{_float(row['aging_temp_c'], 0):.0f} C for {_float(row['aging_hours'], 0):.0f} h",
            "damage_index": float(exposure["thermal_exposure"]),
            "monitored_metric": "sheet/contact resistance drift",
        },
        {
            "name": "Thermal cycling",
            "condition": (
                f"{_float(row['cycle_low_temp_c'], 0):.0f} C to {_float(row['cycle_high_temp_c'], 0):.0f} C, "
                f"{_float(row['thermal_cycles'], 0):.0f} cycles"
            ),
            "damage_index": float(exposure["cycling_exposure"]),
            "monitored_metric": "crack probability and contact drift",
        },
        {
            "name": "Mechanical strain",
            "condition": f"{_float(row['strain_pct'], 0):.2f}% strain, {_float(row['strain_cycles'], 0):.0f} cycles",
            "damage_index": float(exposure["mechanical_exposure"]),
            "monitored_metric": "meander cracking and shear margin",
        },
        {
            "name": "Structural inspection",
            "condition": f"{_float(row['ct_void_fraction_pct'], 0):.1f}% CT voids, {_float(row['alignment_error_um'], 0):.0f} um alignment",
            "damage_index": float(_clip(_float(row["ct_void_fraction_pct"]) / 16.0 + _float(row["alignment_error_um"]) / 160.0, 0.0, 1.0)),
            "monitored_metric": "voiding and delamination",
        },
    ]
    cumulative = float(np.clip(sum(item["damage_index"] for item in segments) / len(segments), 0.0, 1.0))
    return {
        "segments": segments,
        "cumulative_damage_index": cumulative,
        "mission_severity": "High" if cumulative > 0.62 else "Moderate" if cumulative > 0.32 else "Low",
        "dominant_segment": max(segments, key=lambda item: item["damage_index"]),
    }


def _accelerated_equivalence(row: dict[str, Any]) -> dict[str, Any]:
    use_temp = _float(row.get("use_temp_c"), 150.0)
    test_temp = _float(row["aging_temp_c"], 350.0)
    aging_hours = _float(row["aging_hours"], 168.0)
    af = _arrhenius_factor(use_temp, test_temp)
    delta_t_test = max(_float(row["cycle_high_temp_c"]) - _float(row["cycle_low_temp_c"]), 1.0)
    delta_t_use = max(_float(row.get("use_delta_t_c"), 100.0), 1.0)
    cycle_af = float(np.clip((delta_t_test / delta_t_use) ** 2.1, 0.05, 250.0))
    equivalent_hours = float(aging_hours * af)
    equivalent_cycles = float(_float(row["thermal_cycles"]) * cycle_af)
    return {
        "use_temperature_c": use_temp,
        "test_temperature_c": test_temp,
        "thermal_acceleration_factor": af,
        "field_equivalent_hours": equivalent_hours,
        "cycle_acceleration_factor": cycle_af,
        "field_equivalent_cycles": equivalent_cycles,
        "assumption": "Same dominant failure mechanism is assumed between use and accelerated test conditions.",
        "warning": "Treat equivalence as planning guidance until calibrated with physical coupon data.",
    }


def _cycle_extractor(row: dict[str, Any]) -> dict[str, Any]:
    low = _float(row["cycle_low_temp_c"], -40.0)
    high = _float(row["cycle_high_temp_c"], 125.0)
    cycles = int(_float(row["thermal_cycles"], 250.0))
    delta = high - low
    bins = [
        {"range": f"{low:.0f} to 25 C", "count": cycles, "damage_share": 0.18 if low < 0 else 0.08},
        {"range": f"25 to {min(high, 125):.0f} C", "count": cycles, "damage_share": 0.32},
        {"range": f"125 to {high:.0f} C", "count": cycles if high > 125 else 0, "damage_share": 0.50 if high > 125 else 0.0},
    ]
    bins = [item for item in bins if item["count"] > 0]
    rainflow_index = float(np.clip(cycles * (max(delta, 1.0) / 540.0) ** 1.9 / 1000.0, 0.0, 1.0))
    return {
        "detected_cycles": cycles,
        "temperature_delta_c": delta,
        "equivalent_full_cycles": float(cycles * max(delta, 1.0) / 540.0),
        "rainflow_damage_index": rainflow_index,
        "bins": bins,
        "csv_ready_schema": ["timestamp_s", "temperature_c", "resistance_ohm", "optional_note"],
    }


def _monte_carlo(bundle: dict[str, Any], row: dict[str, Any], samples: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(samples):
        sampled = dict(row)
        sampled["aging_temp_c"] = _clip(rng.normal(_float(row["aging_temp_c"]), 18.0), 25.0, 520.0)
        sampled["aging_hours"] = _clip(rng.lognormal(math.log(max(_float(row["aging_hours"]), 1.0)), 0.18), 0.0, 1500.0)
        sampled["thermal_cycles"] = _clip(rng.normal(_float(row["thermal_cycles"]), max(25.0, _float(row["thermal_cycles"]) * 0.12)), 0.0, 1300.0)
        sampled["ct_void_fraction_pct"] = _clip(rng.normal(_float(row["ct_void_fraction_pct"]), 1.1), 0.0, 22.0)
        sampled["edge_roughness_um"] = _clip(rng.normal(_float(row["edge_roughness_um"]), 1.4), 0.5, 30.0)
        sampled["alignment_error_um"] = _clip(rng.normal(_float(row["alignment_error_um"]), 6.0), 0.0, 120.0)
        sampled["oxidation_index"] = _clip(rng.normal(_float(row["oxidation_index"]), 0.055), 0.0, 1.0)
        sampled["strain_pct"] = _clip(rng.normal(_float(row["strain_pct"]), 0.18), 0.0, 6.5)
        sampled["line_width_um"] = _clip(rng.normal(_float(row["line_width_um"]), max(1.5, _float(row["line_width_um"]) * 0.035)), 20.0, 2200.0)
        sampled["thickness_um"] = _clip(rng.normal(_float(row["thickness_um"]), 0.18), 0.45, 12.0)
        rows.append(sampled)
    results = _predict_rows(bundle, rows)
    reliability = np.array([item["prediction"]["reliability_score"] for item in results])
    rul = np.array([item["derived"]["remaining_useful_life_hours"] for item in results])
    decisions = Counter(item["qualification_decision"] for item in results)
    modes = Counter(item["failure_mode"] for item in results)
    return {
        "samples": samples,
        "reliability": {
            "p10": float(np.percentile(reliability, 10)),
            "p50": float(np.percentile(reliability, 50)),
            "p90": float(np.percentile(reliability, 90)),
            "mean": float(np.mean(reliability)),
        },
        "rul_hours": {
            "p10": float(np.percentile(rul, 10)),
            "p50": float(np.percentile(rul, 50)),
            "p90": float(np.percentile(rul, 90)),
        },
        "decision_probability": {key: float(value / samples) for key, value in decisions.items()},
        "failure_mode_probability": {key: float(value / samples) for key, value in modes.most_common()},
    }


def _failure_site_map(bundle: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    rows = [dict(row, coupon_structure=structure["id"]) for structure in COUPON_STRUCTURES]
    results = _predict_rows(bundle, rows)
    sites = []
    for result in results:
        structure_id = result["input"]["coupon_structure"]
        info = next(structure for structure in COUPON_STRUCTURES if structure["id"] == structure_id)
        sites.append(
            {
                "id": structure_id,
                "label": info["label"],
                "zone": info["zone"],
                "purpose": info.get("purpose", info["measurement_family"]),
                "risk_index": result["derived"]["failure_risk_index"],
                "reliability_score": result["prediction"]["reliability_score"],
                "rul_hours": result["derived"]["remaining_useful_life_hours"],
                "failure_mode": result["failure_mode"],
                "decision": result["qualification_decision"],
                "coordinates": SITE_COORDINATES.get(structure_id, {"x": 50, "y": 50}),
            }
        )
    sites.sort(key=lambda item: item["risk_index"], reverse=True)
    return {"sites": sites, "highest_risk": sites[0], "lowest_risk": sites[-1]}


def _sensitivity(bundle: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    baseline = predict_coupon(bundle, row)
    baseline_rel = baseline["prediction"]["reliability_score"]
    variables = [
        ("aging_temp_c", "Aging temperature", -75.0, 75.0, 25.0, 500.0),
        ("aging_hours", "Aging hours", -96.0, 240.0, 0.0, 1200.0),
        ("thermal_cycles", "Thermal cycles", -150.0, 350.0, 0.0, 1200.0),
        ("ct_void_fraction_pct", "CT void fraction", -2.5, 3.5, 0.0, 18.0),
        ("edge_roughness_um", "Edge roughness", -3.0, 4.5, 0.8, 24.0),
        ("alignment_error_um", "Alignment error", -12.0, 18.0, 0.0, 90.0),
        ("strain_pct", "Strain", -0.35, 0.75, 0.0, 6.5),
        ("cure_peak_temp_c", "Cure peak temperature", -70.0, 70.0, 140.0, 520.0),
    ]
    rows = []
    labels = []
    for key, label, low_delta, high_delta, lo, hi in variables:
        low = dict(row)
        high = dict(row)
        low[key] = _clip(_float(row[key]) + low_delta, lo, hi)
        high[key] = _clip(_float(row[key]) + high_delta, lo, hi)
        labels.append((key, label, low[key], high[key]))
        rows.extend([low, high])
    results = _predict_rows(bundle, rows)
    entries = []
    for idx, (key, label, low_value, high_value) in enumerate(labels):
        low_result = results[idx * 2]
        high_result = results[idx * 2 + 1]
        low_rel = low_result["prediction"]["reliability_score"]
        high_rel = high_result["prediction"]["reliability_score"]
        entries.append(
            {
                "key": key,
                "label": label,
                "low_value": low_value,
                "high_value": high_value,
                "low_reliability": low_rel,
                "high_reliability": high_rel,
                "impact": float(abs(high_rel - low_rel)),
                "direction": "higher is worse" if high_rel < low_rel else "higher is better",
                "baseline_delta": float(max(abs(low_rel - baseline_rel), abs(high_rel - baseline_rel))),
            }
        )
    entries.sort(key=lambda item: item["impact"], reverse=True)
    return {"baseline_reliability": baseline_rel, "drivers": entries}


def _iso_life(bundle: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    temps = [150.0, 250.0, 350.0, 500.0]
    hours = [24.0, 72.0, 168.0, 500.0, 1000.0]
    temp_hour_rows = []
    for temp in temps:
        for hour in hours:
            temp_hour_rows.append(dict(row, aging_temp_c=temp, aging_hours=hour, cycle_high_temp_c=500.0 if temp >= 500.0 else 125.0))
    temp_hour_results = _predict_rows(bundle, temp_hour_rows)
    matrix = []
    cursor = 0
    for temp in temps:
        values = []
        for hour in hours:
            result = temp_hour_results[cursor]
            cursor += 1
            values.append(
                {
                    "aging_hours": hour,
                    "reliability_score": result["prediction"]["reliability_score"],
                    "rul_hours": result["derived"]["remaining_useful_life_hours"],
                    "decision": result["qualification_decision"],
                }
            )
        matrix.append({"aging_temp_c": temp, "values": values})

    voids = [1.0, 3.0, 6.0, 10.0, 14.0]
    strains = [0.2, 0.65, 1.5, 3.0, 5.0]
    morph_rows = [dict(row, ct_void_fraction_pct=void, strain_pct=strain) for void in voids for strain in strains]
    morph_results = _predict_rows(bundle, morph_rows)
    morphology = []
    cursor = 0
    for void in voids:
        values = []
        for strain in strains:
            result = morph_results[cursor]
            cursor += 1
            values.append({"strain_pct": strain, "reliability_score": result["prediction"]["reliability_score"]})
        morphology.append({"ct_void_fraction_pct": void, "values": values})

    return {"temperature_axis": temps, "hour_axis": hours, "temp_hour_matrix": matrix, "void_axis": voids, "strain_axis": strains, "morphology_matrix": morphology}


def _what_if(bundle: dict[str, Any], row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    scenarios = [
        ("current", "Current recipe", {}),
        ("ani_hardened", "ANI high-temp hardening", {"ink_family": "high_temp_500c", "cure_peak_temp_c": 455.0, "cure_time_min": 70.0}),
        ("low_void", "Low-void attach process", {"ct_void_fraction_pct": 2.0, "alignment_error_um": 10.0}),
        ("polished_edge", "Tighter print edge", {"edge_roughness_um": 2.4, "oxidation_index": max(0.06, _float(row["oxidation_index"]) - 0.08)}),
        ("soft_cycle", "Reduced cycling severity", {"thermal_cycles": max(50.0, _float(row["thermal_cycles"]) * 0.45), "cycle_high_temp_c": 125.0}),
        ("proof_500c", "500C proof stress", {"aging_temp_c": 500.0, "aging_hours": 168.0, "cycle_high_temp_c": 500.0}),
    ]
    rows = [dict(row, **patch) for _, _, patch in scenarios]
    results = _predict_rows(bundle, rows)
    baseline_rel = baseline["prediction"]["reliability_score"]
    baseline_rul = baseline["derived"]["remaining_useful_life_hours"]
    items = []
    for idx, (scenario_id, label, _) in enumerate(scenarios):
        result = results[idx]
        items.append(
            {
                "id": scenario_id,
                "label": label,
                "decision": result["qualification_decision"],
                "failure_mode": result["failure_mode"],
                "reliability_score": result["prediction"]["reliability_score"],
                "rul_hours": result["derived"]["remaining_useful_life_hours"],
                "reliability_delta": result["prediction"]["reliability_score"] - baseline_rel,
                "rul_delta_hours": result["derived"]["remaining_useful_life_hours"] - baseline_rul,
            }
        )
    return {"scenarios": items, "best": max(items, key=lambda item: item["reliability_score"])}


def _validation_dashboard(bundle: dict[str, Any], failure_map: dict[str, Any]) -> dict[str, Any]:
    metrics = bundle.get("metrics", {})
    coupon_metrics = metrics.get("coupon", {})
    mae = coupon_metrics.get("coupon_mae", {})
    r2 = coupon_metrics.get("coupon_r2", {})
    q90 = coupon_metrics.get("conformal_q90", {})
    points = []
    for index, site in enumerate(failure_map["sites"]):
        synthetic_measured = site["reliability_score"] + (-1) ** index * (1.5 + index * 0.18)
        points.append(
            {
                "structure": site["label"],
                "predicted_reliability": site["reliability_score"],
                "synthetic_measured_reliability": float(_clip(synthetic_measured, 0.0, 100.0)),
                "absolute_error": float(abs(site["reliability_score"] - synthetic_measured)),
            }
        )
    return {
        "holdout_metrics": {"mae": mae, "r2": r2, "q90": q90},
        "calibration_points": points,
        "coverage_target_pct": 90.0,
        "calibration_message": "Replace synthetic measured points with lab coupon data to continuously recalibrate BOND-AI-RAP.",
    }


def _qualification_report(row: dict[str, Any], baseline: dict[str, Any], acceleration: dict[str, Any], failure_map: dict[str, Any], test_plan: dict[str, Any]) -> dict[str, Any]:
    pred = baseline["prediction"]
    derived = baseline["derived"]
    return {
        "title": "BOND-AI-RAP Reliability Assessment Report",
        "executive_decision": baseline["qualification_decision"],
        "coupon_configuration": {
            "structure": row["coupon_structure"],
            "zone": row["coupon_zone"],
            "ink": row["ink_family"],
            "test_method": row["test_method"],
        },
        "key_results": [
            f"Reliability score: {pred['reliability_score']:.1f} / 100",
            f"Remaining useful life estimate: {derived['remaining_useful_life_hours']:.0f} h",
            f"Dominant mechanism: {baseline['failure_mode'].replace('_', ' ')}",
            f"Field-equivalent aging exposure: {acceleration['field_equivalent_hours']:.0f} h",
            f"Highest-risk coupon site: {failure_map['highest_risk']['label']}",
        ],
        "recommended_next_action": test_plan["recommended_experiments"][0] if test_plan["recommended_experiments"] else None,
        "decision_reasons": baseline["decision_reasons"],
    }


def _test_planner(bundle: dict[str, Any], row: dict[str, Any], candidates: int) -> dict[str, Any]:
    feedback = digital_twin_feedback(bundle, row, candidates=max(160, min(900, candidates)), seed=911)
    experiments = feedback.get("next_experiments", [])
    recipes = feedback.get("top", [])
    return {
        "recommended_experiments": experiments,
        "recipe_shortlist": recipes[:4],
        "actions": feedback.get("actions", []),
        "coupon_reduction_target_pct": 35.0,
    }


def _plugin_registry() -> dict[str, Any]:
    return {
        "models": PHYSICS_MODEL_LIBRARY,
        "plugin_contract": {
            "inputs": ["coupon payload", "prediction", "uncertainty", "mission profile"],
            "outputs": ["damage index", "life estimate", "failure probability", "explanation"],
            "status": "ready for lab-calibrated physics modules",
        },
        "candidate_plugins": [
            "SEM/EDS reaction-layer thickness extractor",
            "CT void morphology descriptor",
            "FE stress hot-spot importer",
            "Arrhenius parameter fitter from aging coupons",
            "Conformal calibration updater from blind CPW validation",
        ],
    }


def rap_assessment(bundle: dict[str, Any], payload: dict[str, Any], candidates: int = 360, samples: int = 260, seed: int = 2026) -> dict[str, Any]:
    row = complete_coupon_payload(payload)
    baseline = predict_coupon(bundle, row)
    lifecycle = _life_cycle_profile(row, baseline)
    acceleration = _accelerated_equivalence(row)
    cycle_extractor = _cycle_extractor(row)
    failure_map = _failure_site_map(bundle, row)
    test_plan = _test_planner(bundle, row, candidates)
    assessment = {
        "baseline": baseline,
        "virtual_qualification": _virtual_qualification(baseline),
        "life_cycle_profile": lifecycle,
        "accelerated_equivalence": acceleration,
        "cycle_extractor": cycle_extractor,
        "physics_models": {"models": PHYSICS_MODEL_LIBRARY},
        "monte_carlo": _monte_carlo(bundle, row, max(80, min(800, samples)), seed),
        "failure_site_map": failure_map,
        "sensitivity": _sensitivity(bundle, row),
        "iso_life": _iso_life(bundle, row),
        "test_planner": test_plan,
        "what_if": _what_if(bundle, row, baseline),
        "validation": _validation_dashboard(bundle, failure_map),
        "plugin_registry": _plugin_registry(),
    }
    assessment["qualification_report"] = _qualification_report(row, baseline, acceleration, failure_map, test_plan)
    return assessment
