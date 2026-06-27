from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MaterialSet:
    set_id: str
    label: str
    ink: str
    substrate: str
    stack: str
    conductivity_s_m: float
    viscosity_cp: float
    surface_energy_mn_m: float
    dielectric_constant: float
    wetting_bias: float


MATERIAL_SETS: tuple[MaterialSet, ...] = (
    MaterialSet(
        "ag_np_alumina",
        "Baseline Ag-NP",
        "Baseline Ag-NP ink",
        "500C-capable alumina",
        "Baseline Ag-NP ink on alumina",
        1.05e7,
        31.0,
        41.0,
        9.8,
        0.02,
    ),
    MaterialSet(
        "ani_alumina",
        "ANI 500C",
        "ANI 500C conductive ink",
        "500C-capable alumina",
        "ANI 500C conductive ink on alumina",
        7.7e6,
        41.0,
        41.0,
        9.8,
        0.09,
    ),
)


PATTERN_TYPES: tuple[dict[str, Any], ...] = (
    {
        "id": "square_pad",
        "label": "Square pads",
        "nominal_width_um": 850.0,
        "nominal_pitch_um": 1200.0,
        "nominal_thickness_um": 3.8,
        "path_length_mm": 7.5,
    },
    {
        "id": "dogbone",
        "label": "Dog bones",
        "nominal_width_um": 145.0,
        "nominal_pitch_um": 420.0,
        "nominal_thickness_um": 3.2,
        "path_length_mm": 26.0,
    },
    {
        "id": "meander",
        "label": "Meander lines",
        "nominal_width_um": 82.0,
        "nominal_pitch_um": 245.0,
        "nominal_thickness_um": 2.7,
        "path_length_mm": 82.0,
    },
)


DEVICE_TYPES: tuple[dict[str, Any], ...] = (
    {
        "id": "cpw",
        "label": "Blind CPW validation structure",
        "nominal_width_um": 360.0,
        "nominal_gap_um": 180.0,
        "trace_length_mm": 31.0,
        "substrate_thickness_mm": 0.508,
    },
)

COUPON_STRUCTURES: tuple[dict[str, Any], ...] = (
    {
        "id": "straight_lines",
        "label": "Straight lines",
        "zone": "interface",
        "measurement_family": "sheet_resistance",
        "purpose": "Resistance drift",
        "nominal_width_um": 95.0,
        "path_length_mm": 64.0,
        "overlap_area_mm2": 0.0,
        "bond_area_mm2": 0.0,
    },
    {
        "id": "meander_lines",
        "label": "Meander lines",
        "zone": "interface",
        "measurement_family": "crack_fatigue",
        "purpose": "Crack and fatigue",
        "nominal_width_um": 78.0,
        "path_length_mm": 135.0,
        "overlap_area_mm2": 0.0,
        "bond_area_mm2": 0.0,
    },
    {
        "id": "square_pads",
        "label": "Square pads",
        "zone": "interface",
        "measurement_family": "adhesion_delamination",
        "purpose": "Adhesion and delamination",
        "nominal_width_um": 520.0,
        "path_length_mm": 6.0,
        "overlap_area_mm2": 0.0,
        "bond_area_mm2": 0.30,
    },
    {
        "id": "overlap_pads",
        "label": "Overlap pads",
        "zone": "interface",
        "measurement_family": "interface_adhesion",
        "purpose": "Interface adhesion",
        "nominal_width_um": 740.0,
        "path_length_mm": 10.0,
        "overlap_area_mm2": 1.15,
        "bond_area_mm2": 0.0,
    },
    {
        "id": "daisy_chain_kelvin",
        "label": "Daisy-chain Kelvin",
        "zone": "bonding",
        "measurement_family": "contact_resistance",
        "purpose": "Contact resistance",
        "nominal_width_um": 135.0,
        "path_length_mm": 92.0,
        "overlap_area_mm2": 0.22,
        "bond_area_mm2": 0.0,
    },
    {
        "id": "dummy_die_attach",
        "label": "Dummy die / chip attach",
        "zone": "bonding",
        "measurement_family": "bond_reliability",
        "purpose": "Bond reliability",
        "nominal_width_um": 1100.0,
        "path_length_mm": 4.5,
        "overlap_area_mm2": 0.0,
        "bond_area_mm2": 1.85,
    },
    {
        "id": "shear_test_pads",
        "label": "Shear test pads",
        "zone": "bonding",
        "measurement_family": "shear_strength",
        "purpose": "Adhesion strength",
        "nominal_width_um": 880.0,
        "path_length_mm": 7.2,
        "overlap_area_mm2": 0.58,
        "bond_area_mm2": 1.10,
    },
)

COUPON_INKS: tuple[dict[str, Any], ...] = (
    {
        "id": "baseline_ag",
        "label": "Baseline Ag-NP ink",
        "nominal_conductivity_s_m": 1.05e7,
        "temp_rating_c": 250.0,
        "oxidation_bias": 0.20,
        "adhesion_bias": 0.00,
    },
    {
        "id": "high_temp_500c",
        "label": "ANI 500C conductive ink",
        "nominal_conductivity_s_m": 7.8e6,
        "temp_rating_c": 500.0,
        "oxidation_bias": -0.12,
        "adhesion_bias": 0.12,
    },
)

COUPON_TEST_METHODS: tuple[dict[str, Any], ...] = (
    {"id": "electrical_4pt", "label": "Electrical 4-point / Kelvin", "stress_bias": 0.02},
    {"id": "thermal_aging", "label": "Thermal aging up to 500C", "stress_bias": 0.18},
    {"id": "thermal_cycling", "label": "Thermal cycling -40C to 500C", "stress_bias": 0.24},
    {"id": "mechanical_bending", "label": "Mechanical bending / strain", "stress_bias": 0.22},
    {"id": "shear_pull", "label": "Shear / pull testing", "stress_bias": 0.20},
    {"id": "xray_ct", "label": "X-ray / CT", "stress_bias": 0.05},
    {"id": "optical_sem_eds_fib", "label": "Optical / SEM / EDS / FIB", "stress_bias": 0.04},
)

REPRESENTATIVE_CONDITIONS: tuple[str, ...] = (
    "Thermal aging at 150C, 250C, 350C, and 500C",
    "Thermal cycling from -40C to 125C and room temperature to 500C",
    "Static bend, cyclic bend, and strain cycling",
    "In-situ electrical monitoring during aging and cycling",
)

QUALIFICATION_DECISIONS: tuple[str, ...] = ("PASS", "MARGINAL", "DEFER_TO_INSPECTION")


PATTERN_FEATURES = [
    "material_set",
    "ink",
    "substrate",
    "pattern_type",
    "nominal_width_um",
    "nominal_pitch_um",
    "nominal_thickness_um",
    "path_length_mm",
    "print_speed_mm_s",
    "atomizer_voltage_v",
    "carrier_flow_sccm",
    "sheath_flow_sccm",
    "substrate_temp_c",
    "standoff_mm",
    "humidity_pct",
    "deposition_rate_ug_s",
    "d_ideal_um",
    "d_min_um",
    "d_max_um",
    "blob_count",
    "blob_area_ratio",
    "boundary_angle_deg",
    "edge_roughness_um",
    "camera_noise",
    "cure_energy_j_cm2",
    "viscosity_cp",
    "surface_energy_mn_m",
]

PATTERN_TARGETS = [
    "line_width_um",
    "thickness_um",
    "resistance_ohm",
    "quality_score",
]

INTERFACE_FEATURES = [
    "material_set",
    "ink",
    "substrate",
    "device_type",
    "nominal_width_um",
    "nominal_gap_um",
    "trace_length_mm",
    "substrate_thickness_mm",
    "dielectric_constant",
    "line_width_um",
    "thickness_um",
    "resistance_ohm",
    "quality_score",
    "overspray_ratio",
    "clog_ratio",
    "edge_roughness_um",
    "print_speed_mm_s",
    "deposition_rate_ug_s",
    "atomizer_voltage_v",
    "carrier_flow_sccm",
    "sheath_flow_sccm",
]

INTERFACE_TARGETS = [
    "resonance_frequency_ghz",
    "return_loss_db",
    "insertion_loss_db",
    "impedance_ohm",
    "yield_probability",
]

COUPON_FEATURES = [
    "coupon_zone",
    "coupon_structure",
    "measurement_family",
    "ink_family",
    "substrate",
    "test_method",
    "nominal_width_um",
    "path_length_mm",
    "overlap_area_mm2",
    "bond_area_mm2",
    "print_speed_mm_s",
    "atomizer_voltage_v",
    "carrier_flow_sccm",
    "sheath_flow_sccm",
    "substrate_temp_c",
    "cure_peak_temp_c",
    "cure_time_min",
    "aging_temp_c",
    "aging_hours",
    "cycle_low_temp_c",
    "cycle_high_temp_c",
    "thermal_cycles",
    "bend_radius_mm",
    "strain_pct",
    "strain_cycles",
    "ct_void_fraction_pct",
    "oxidation_index",
    "edge_roughness_um",
    "alignment_error_um",
    "line_width_um",
    "thickness_um",
    "initial_resistance_ohm",
]

COUPON_TARGETS = [
    "sheet_resistance_drift_pct",
    "contact_resistance_drift_pct",
    "crack_probability",
    "delamination_area_pct",
    "adhesion_strength_mpa",
    "void_fraction_pct",
    "post_aging_shear_mpa",
    "reliability_score",
]


def metadata() -> dict[str, Any]:
    return {
        "portal_name": "BOND-AI-RAP",
        "platform": {
            "title": "Integrated Interface-and-Bonding Test Platform",
            "substrate": "Single 500C-capable alumina coupon",
            "objective": (
                "Link printed-interface degradation in Zone A to bond-joint reliability "
                "in Zone B using common inks, process settings, and characterization data."
            ),
            "outcome": (
                "Unified reliability methodology for baseline Ag-NP and ANI high-temperature "
                "conductive inks on alumina."
            ),
        },
        "material_sets": [m.__dict__ for m in MATERIAL_SETS],
        "pattern_types": list(PATTERN_TYPES),
        "device_types": list(DEVICE_TYPES),
        "coupon_structures": list(COUPON_STRUCTURES),
        "coupon_inks": list(COUPON_INKS),
        "coupon_test_methods": list(COUPON_TEST_METHODS),
        "representative_conditions": list(REPRESENTATIVE_CONDITIONS),
        "qualification_decisions": list(QUALIFICATION_DECISIONS),
        "pattern_features": PATTERN_FEATURES,
        "pattern_targets": PATTERN_TARGETS,
        "interface_features": INTERFACE_FEATURES,
        "interface_targets": INTERFACE_TARGETS,
        "coupon_features": COUPON_FEATURES,
        "coupon_targets": COUPON_TARGETS,
    }


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _choice_index(rng: np.random.Generator, size: int, count: int) -> np.ndarray:
    return rng.integers(0, count, size=size)


def _clip(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(a, lo, hi)


def _material_arrays(material_idx: np.ndarray) -> dict[str, np.ndarray]:
    sets = np.array([m.set_id for m in MATERIAL_SETS], dtype=object)[material_idx]
    inks = np.array([m.ink for m in MATERIAL_SETS], dtype=object)[material_idx]
    substrates = np.array([m.substrate for m in MATERIAL_SETS], dtype=object)[material_idx]
    conductivity = np.array([m.conductivity_s_m for m in MATERIAL_SETS])[material_idx]
    viscosity = np.array([m.viscosity_cp for m in MATERIAL_SETS])[material_idx]
    surface = np.array([m.surface_energy_mn_m for m in MATERIAL_SETS])[material_idx]
    dielectric = np.array([m.dielectric_constant for m in MATERIAL_SETS])[material_idx]
    wetting = np.array([m.wetting_bias for m in MATERIAL_SETS])[material_idx]
    return {
        "material_set": sets,
        "ink": inks,
        "substrate": substrates,
        "conductivity_s_m": conductivity,
        "viscosity_cp": viscosity,
        "surface_energy_mn_m": surface,
        "dielectric_constant": dielectric,
        "wetting_bias": wetting,
    }


def _pattern_arrays(pattern_idx: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    pattern_type = np.array([p["id"] for p in PATTERN_TYPES], dtype=object)[pattern_idx]
    width = np.array([p["nominal_width_um"] for p in PATTERN_TYPES])[pattern_idx]
    pitch = np.array([p["nominal_pitch_um"] for p in PATTERN_TYPES])[pattern_idx]
    thickness = np.array([p["nominal_thickness_um"] for p in PATTERN_TYPES])[pattern_idx]
    length = np.array([p["path_length_mm"] for p in PATTERN_TYPES])[pattern_idx]

    width = _clip(width * rng.normal(1.0, 0.045, size=len(pattern_idx)), 38.0, 1400.0)
    pitch = _clip(pitch * rng.normal(1.0, 0.05, size=len(pattern_idx)), 90.0, 1800.0)
    thickness = _clip(thickness * rng.normal(1.0, 0.08, size=len(pattern_idx)), 0.9, 8.5)
    length = _clip(length * rng.normal(1.0, 0.04, size=len(pattern_idx)), 4.0, 130.0)

    return {
        "pattern_type": pattern_type,
        "nominal_width_um": width,
        "nominal_pitch_um": pitch,
        "nominal_thickness_um": thickness,
        "path_length_mm": length,
    }


def generate_pattern_dataset(n: int = 80_000, seed: int | None = 7) -> pd.DataFrame:
    rng = _rng(seed)
    material_idx = _choice_index(rng, n, len(MATERIAL_SETS))
    pattern_idx = _choice_index(rng, n, len(PATTERN_TYPES))
    mat = _material_arrays(material_idx)
    pat = _pattern_arrays(pattern_idx, rng)

    atomizer = rng.uniform(27.0, 43.5, n)
    carrier = rng.uniform(17.0, 42.0, n)
    sheath = rng.uniform(38.0, 82.0, n)
    speed = rng.uniform(5.5, 25.0, n)
    substrate_temp = rng.uniform(24.0, 96.0, n)
    standoff = rng.uniform(1.8, 5.2, n)
    humidity = rng.uniform(16.0, 62.0, n)
    cure_energy = rng.uniform(0.55, 2.6, n)
    camera_noise = np.abs(rng.normal(0.0, 0.045, n))

    viscosity = mat["viscosity_cp"]
    surface = mat["surface_energy_mn_m"]
    wetting = mat["wetting_bias"]
    nominal_width = pat["nominal_width_um"]
    nominal_thickness = pat["nominal_thickness_um"]

    flow_ratio = carrier / np.maximum(sheath, 1.0)
    deposition_rate = (
        20.0
        + 0.82 * (atomizer - 35.0)
        + 0.42 * (carrier - 28.0)
        - 0.22 * (sheath - 60.0)
        - 0.13 * (viscosity - 34.0)
        + rng.normal(0.0, 2.1, n)
    )
    deposition_rate = _clip(deposition_rate, 4.5, 55.0)

    d_ideal = nominal_width * rng.normal(1.12, 0.035, n)
    overspray_drive = (
        0.95 * (flow_ratio - 0.48)
        + 0.020 * (atomizer - 35.0)
        - 0.010 * (speed - 15.0)
        + 0.004 * (humidity - 35.0)
        + wetting
        + rng.normal(0.0, 0.065, n)
    )
    clog_drive = (
        0.028 * (viscosity - 34.0)
        + 0.018 * (speed - 15.0)
        - 0.034 * (atomizer - 35.0)
        - 0.010 * (carrier - 28.0)
        + 0.11 * (standoff - 3.2)
        + rng.normal(0.0, 0.055, n)
    )

    overspray_ratio = _clip(overspray_drive, -0.08, 0.72)
    clog_ratio = _clip(clog_drive, -0.08, 0.58)
    d_max = d_ideal * (1.0 + np.maximum(0, overspray_ratio))
    d_min = d_ideal * (1.0 - np.maximum(0, clog_ratio))

    instability = np.abs(flow_ratio - 0.48) + camera_noise * 2.2 + np.maximum(0, clog_ratio) * 0.55
    blob_lambda = _clip(0.12 + np.maximum(0, clog_ratio) * 6.2 + np.maximum(0, instability - 0.23) * 2.4, 0.02, 5.5)
    blob_count = rng.poisson(blob_lambda)
    blob_area_ratio = _clip(blob_count * rng.uniform(0.002, 0.019, n) + np.maximum(0, clog_ratio) * 0.035, 0, 0.22)
    boundary_angle = rng.normal(0.0, 2.5, n) + overspray_ratio * 11.0 - clog_ratio * 7.5
    edge_roughness = _clip(
        2.0
        + 13.5 * np.maximum(0, overspray_ratio)
        + 17.5 * np.maximum(0, clog_ratio)
        + 18.0 * blob_area_ratio
        + 9.0 * camera_noise
        + np.abs(rng.normal(0, 1.2, n)),
        0.7,
        42.0,
    )

    target_dep = 18.0 + 4.5 * (nominal_width / 120.0) ** 0.25 + 0.08 * (viscosity - 34.0)
    dep_balance = deposition_rate / target_dep
    line_width = nominal_width * (
        1.0
        + 0.42 * np.maximum(0, overspray_ratio)
        - 0.23 * np.maximum(0, clog_ratio)
        + 0.085 * (dep_balance - 1.0)
        + 0.025 * ((surface - 39.0) / 10.0)
        + rng.normal(0.0, 0.025, n)
    )
    line_width = _clip(line_width, 20.0, 2200.0)

    thickness = nominal_thickness * (
        0.78
        + 0.36 * dep_balance
        - 0.010 * (speed - 14.0)
        - 0.12 * np.maximum(0, overspray_ratio)
        + 0.07 * np.maximum(0, clog_ratio)
        + rng.normal(0.0, 0.045, n)
    )
    thickness = _clip(thickness, 0.45, 12.0)

    cross_section_m2 = (line_width * 1e-6) * (thickness * 1e-6)
    length_m = pat["path_length_mm"] * 1e-3
    resistance = length_m / np.maximum(cross_section_m2 * mat["conductivity_s_m"], 1e-12)
    resistance *= 1.0 + edge_roughness / 95.0 + np.maximum(0, clog_ratio) * 0.35 + rng.normal(0.0, 0.035, n)
    resistance = _clip(resistance, 0.001, 5_000.0)

    width_error = np.abs(line_width - nominal_width) / nominal_width
    thickness_error = np.abs(thickness - nominal_thickness) / nominal_thickness
    quality_score = (
        100
        - 95 * width_error
        - 70 * thickness_error
        - 0.78 * edge_roughness
        - 34 * blob_area_ratio
        - 8 * np.maximum(0, overspray_ratio - 0.12)
        - 11 * np.maximum(0, clog_ratio - 0.10)
    )
    quality_score = _clip(quality_score, 0, 100)

    conditions = [
        np.maximum(0, clog_ratio) > 0.22,
        blob_count >= 4,
        np.maximum(0, overspray_ratio) > 0.24,
        dep_balance < 0.78,
        dep_balance > 1.28,
        (edge_roughness > 17.0) | (instability > 0.52),
    ]
    choices = ["clog", "clog", "overspray", "under_deposit", "over_deposit", "unstable"]
    defect_class = np.select(conditions, choices, default="nominal")

    df = pd.DataFrame(
        {
            "material_set": mat["material_set"],
            "ink": mat["ink"],
            "substrate": mat["substrate"],
            "pattern_type": pat["pattern_type"],
            "nominal_width_um": nominal_width,
            "nominal_pitch_um": pat["nominal_pitch_um"],
            "nominal_thickness_um": nominal_thickness,
            "path_length_mm": pat["path_length_mm"],
            "print_speed_mm_s": speed,
            "atomizer_voltage_v": atomizer,
            "carrier_flow_sccm": carrier,
            "sheath_flow_sccm": sheath,
            "substrate_temp_c": substrate_temp,
            "standoff_mm": standoff,
            "humidity_pct": humidity,
            "deposition_rate_ug_s": deposition_rate,
            "d_ideal_um": d_ideal,
            "d_min_um": d_min,
            "d_max_um": d_max,
            "blob_count": blob_count,
            "blob_area_ratio": blob_area_ratio,
            "boundary_angle_deg": boundary_angle,
            "edge_roughness_um": edge_roughness,
            "camera_noise": camera_noise,
            "cure_energy_j_cm2": cure_energy,
            "viscosity_cp": viscosity,
            "surface_energy_mn_m": surface,
            "overspray_ratio": np.maximum(0, overspray_ratio),
            "clog_ratio": np.maximum(0, clog_ratio),
            "line_width_um": line_width,
            "thickness_um": thickness,
            "resistance_ohm": resistance,
            "quality_score": quality_score,
            "defect_class": defect_class,
        }
    )
    return df


def _device_arrays(device_idx: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    device_type = np.array([d["id"] for d in DEVICE_TYPES], dtype=object)[device_idx]
    width = np.array([d["nominal_width_um"] for d in DEVICE_TYPES])[device_idx]
    gap = np.array([d["nominal_gap_um"] for d in DEVICE_TYPES])[device_idx]
    length = np.array([d["trace_length_mm"] for d in DEVICE_TYPES])[device_idx]
    height = np.array([d["substrate_thickness_mm"] for d in DEVICE_TYPES])[device_idx]
    return {
        "device_type": device_type,
        "nominal_width_um": width * rng.normal(1.0, 0.035, len(device_idx)),
        "nominal_gap_um": gap * rng.normal(1.0, 0.045, len(device_idx)),
        "trace_length_mm": length * rng.normal(1.0, 0.035, len(device_idx)),
        "substrate_thickness_mm": height * rng.normal(1.0, 0.035, len(device_idx)),
    }


def _coupon_structure_arrays(structure_idx: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    structures = np.array([s["id"] for s in COUPON_STRUCTURES], dtype=object)[structure_idx]
    zones = np.array([s["zone"] for s in COUPON_STRUCTURES], dtype=object)[structure_idx]
    families = np.array([s["measurement_family"] for s in COUPON_STRUCTURES], dtype=object)[structure_idx]
    width = np.array([s["nominal_width_um"] for s in COUPON_STRUCTURES])[structure_idx]
    length = np.array([s["path_length_mm"] for s in COUPON_STRUCTURES])[structure_idx]
    overlap = np.array([s["overlap_area_mm2"] for s in COUPON_STRUCTURES])[structure_idx]
    bond = np.array([s["bond_area_mm2"] for s in COUPON_STRUCTURES])[structure_idx]
    return {
        "coupon_structure": structures,
        "coupon_zone": zones,
        "measurement_family": families,
        "nominal_width_um": _clip(width * rng.normal(1.0, 0.04, len(structure_idx)), 45.0, 1400.0),
        "path_length_mm": _clip(length * rng.normal(1.0, 0.05, len(structure_idx)), 3.5, 165.0),
        "overlap_area_mm2": _clip(overlap * rng.normal(1.0, 0.08, len(structure_idx)), 0.0, 1.8),
        "bond_area_mm2": _clip(bond * rng.normal(1.0, 0.08, len(structure_idx)), 0.0, 2.4),
    }


def _coupon_ink_arrays(ink_idx: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ink_family": np.array([i["id"] for i in COUPON_INKS], dtype=object)[ink_idx],
        "nominal_conductivity_s_m": np.array([i["nominal_conductivity_s_m"] for i in COUPON_INKS])[ink_idx],
        "temp_rating_c": np.array([i["temp_rating_c"] for i in COUPON_INKS])[ink_idx],
        "oxidation_bias": np.array([i["oxidation_bias"] for i in COUPON_INKS])[ink_idx],
        "adhesion_bias": np.array([i["adhesion_bias"] for i in COUPON_INKS])[ink_idx],
    }


def generate_coupon_dataset(n: int = 120_000, seed: int | None = 31) -> pd.DataFrame:
    rng = _rng(seed)
    structure_idx = _choice_index(rng, n, len(COUPON_STRUCTURES))
    ink_idx = _choice_index(rng, n, len(COUPON_INKS))
    method_idx = _choice_index(rng, n, len(COUPON_TEST_METHODS))
    struct = _coupon_structure_arrays(structure_idx, rng)
    ink = _coupon_ink_arrays(ink_idx)
    methods = np.array([m["id"] for m in COUPON_TEST_METHODS], dtype=object)[method_idx]
    method_stress = np.array([m["stress_bias"] for m in COUPON_TEST_METHODS])[method_idx]

    width = struct["nominal_width_um"]
    speed = rng.uniform(5.5, 24.0, n)
    atomizer = rng.uniform(28.0, 42.0, n)
    carrier = rng.uniform(18.0, 40.0, n)
    sheath = rng.uniform(42.0, 80.0, n)
    substrate_temp = rng.uniform(35.0, 95.0, n)
    cure_peak = rng.uniform(140.0, 520.0, n)
    cure_time = rng.uniform(10.0, 120.0, n)

    aging_choices = np.array([25.0, 150.0, 250.0, 350.0, 500.0])
    aging_temp = rng.choice(aging_choices, n, p=[0.15, 0.18, 0.22, 0.22, 0.23])
    aging_hours = rng.choice(np.array([0.0, 24.0, 72.0, 168.0, 500.0, 1000.0]), n, p=[0.12, 0.18, 0.22, 0.24, 0.16, 0.08])
    cycle_low = rng.choice(np.array([-40.0, 25.0]), n, p=[0.72, 0.28])
    cycle_high = rng.choice(np.array([125.0, 250.0, 350.0, 500.0]), n, p=[0.28, 0.25, 0.24, 0.23])
    thermal_cycles = rng.choice(np.array([0, 50, 100, 250, 500, 1000]), n, p=[0.16, 0.18, 0.22, 0.20, 0.16, 0.08])
    bend_radius = rng.uniform(3.0, 80.0, n)
    strain = _clip(18.0 / bend_radius + rng.normal(0.0, 0.035, n), 0.02, 6.5)
    strain_cycles = rng.choice(np.array([0, 50, 200, 1000, 5000, 10000]), n, p=[0.20, 0.17, 0.22, 0.20, 0.14, 0.07])

    flow_ratio = carrier / np.maximum(sheath, 1.0)
    deposition_rate = _clip(
        19.5
        + 0.76 * (atomizer - 35.0)
        + 0.38 * (carrier - 28.0)
        - 0.19 * (sheath - 60.0)
        - 0.08 * (speed - 14.0)
        + rng.normal(0.0, 2.0, n),
        5.0,
        52.0,
    )
    spread = _clip(0.82 * (flow_ratio - 0.49) + 0.016 * (atomizer - 35.0) - 0.010 * (speed - 14.0), -0.04, 0.46)
    narrowing = _clip(0.012 * (speed - 14.0) - 0.024 * (atomizer - 35.0) - 0.008 * (carrier - 28.0), -0.04, 0.38)
    edge_roughness = _clip(2.1 + 16.0 * np.maximum(0, spread) + 19.0 * np.maximum(0, narrowing) + np.abs(rng.normal(0, 1.4, n)), 0.7, 36.0)
    alignment = _clip(np.abs(rng.normal(18.0, 14.0, n)) + np.maximum(0, spread) * 32.0, 0.8, 115.0)

    dep_target = 19.0 + 4.2 * (width / 120.0) ** 0.25
    dep_balance = deposition_rate / dep_target
    line_width = _clip(width * (1.0 + 0.34 * np.maximum(0, spread) - 0.20 * np.maximum(0, narrowing) + 0.07 * (dep_balance - 1.0) + rng.normal(0, 0.025, n)), 22.0, 2100.0)
    thickness = _clip(3.1 * (0.72 + 0.34 * dep_balance - 0.011 * (speed - 14.0) + rng.normal(0, 0.055, n)), 0.45, 12.0)
    area = np.maximum((line_width * 1e-6) * (thickness * 1e-6), 1e-12)
    initial_resistance = _clip((struct["path_length_mm"] * 1e-3) / (area * ink["nominal_conductivity_s_m"]) * (1 + edge_roughness / 92.0), 0.001, 5500.0)

    over_temp = np.maximum(0.0, aging_temp - ink["temp_rating_c"]) / 250.0
    temp_fraction = np.maximum(0.0, aging_temp - 25.0) / 475.0
    thermal_dose = np.log1p(aging_hours) / np.log1p(1000.0) * temp_fraction
    cycling_dose = np.log1p(thermal_cycles) / np.log1p(1000.0) * np.maximum(0, cycle_high - cycle_low) / 540.0
    mechanical_dose = np.log1p(strain_cycles) / np.log1p(10000.0) * strain / 6.5
    zone_bonding = (struct["coupon_zone"] == "bonding").astype(float)
    long_line = (struct["coupon_structure"] == "meander_lines").astype(float)
    overlap_sensitive = ((struct["overlap_area_mm2"] > 0) | (struct["bond_area_mm2"] > 0)).astype(float)

    oxidation_index = _clip(
        0.10
        + 0.75 * thermal_dose
        + 0.55 * over_temp
        + ink["oxidation_bias"]
        + 0.15 * np.maximum(0, edge_roughness - 8.0) / 28.0
        + rng.normal(0.0, 0.045, n),
        0.0,
        1.0,
    )
    ct_void_fraction = _clip(
        1.2
        + 6.5 * np.maximum(0, narrowing)
        + 2.4 * np.maximum(0, 1.0 - dep_balance)
        + 3.2 * zone_bonding
        + 1.8 * overlap_sensitive
        + rng.normal(0.0, 0.75, n),
        0.0,
        23.0,
    )

    crack_drive = -2.25 + 2.6 * mechanical_dose + 1.9 * cycling_dose + 0.045 * edge_roughness + 0.85 * long_line + 0.85 * over_temp
    crack_probability = _clip(1.0 / (1.0 + np.exp(-crack_drive)) + rng.normal(0.0, 0.025, n), 0.0, 1.0)
    delamination_area = _clip(
        2.0
        + 28.0 * over_temp
        + 18.0 * thermal_dose
        + 14.0 * cycling_dose
        + 1.4 * ct_void_fraction
        + 0.19 * alignment
        - 8.0 * ink["adhesion_bias"]
        + rng.normal(0.0, 2.2, n),
        0.0,
        100.0,
    )
    sheet_drift = _clip(
        1.2
        + 38.0 * oxidation_index
        + 22.0 * crack_probability
        + 6.0 * thermal_dose
        + 4.5 * np.maximum(0, narrowing)
        + rng.normal(0.0, 2.0, n),
        -3.0,
        180.0,
    )
    contact_drift = _clip(
        1.8
        + 25.0 * oxidation_index
        + 1.8 * ct_void_fraction
        + 0.52 * delamination_area
        + 16.0 * zone_bonding
        + 3.0 * method_stress
        + rng.normal(0.0, 2.6, n),
        -2.0,
        220.0,
    )
    adhesion = _clip(
        31.0
        + 4.8 * ink["adhesion_bias"]
        - 0.19 * delamination_area
        - 0.26 * ct_void_fraction
        - 5.0 * over_temp
        - 0.035 * alignment
        + rng.normal(0.0, 1.5, n),
        1.5,
        42.0,
    )
    shear = _clip(
        24.0
        + 5.2 * ink["adhesion_bias"]
        - 0.16 * delamination_area
        - 0.40 * ct_void_fraction
        - 3.2 * thermal_dose
        - 2.6 * cycling_dose
        + rng.normal(0.0, 1.3, n),
        0.8,
        36.0,
    )
    reliability = _clip(
        100.0
        - 0.30 * sheet_drift
        - 0.27 * contact_drift
        - 36.0 * crack_probability
        - 0.42 * delamination_area
        - 1.18 * ct_void_fraction
        - 1.2 * np.maximum(0, 12.0 - shear)
        + rng.normal(0.0, 2.0, n),
        0.0,
        100.0,
    )

    conditions = [
        reliability >= 82.0,
        crack_probability > 0.58,
        delamination_area > 34.0,
        ct_void_fraction > 11.0,
        sheet_drift > 46.0,
        contact_drift > 58.0,
        shear < 12.0,
    ]
    choices = [
        "pass",
        "crack_fatigue",
        "delamination",
        "voiding",
        "resistance_drift",
        "contact_degradation",
        "shear_failure",
    ]
    failure_mode = np.select(conditions, choices, default="marginal")

    return pd.DataFrame(
        {
            "coupon_zone": struct["coupon_zone"],
            "coupon_structure": struct["coupon_structure"],
            "measurement_family": struct["measurement_family"],
            "ink_family": ink["ink_family"],
            "substrate": "single_alumina",
            "test_method": methods,
            "nominal_width_um": width,
            "path_length_mm": struct["path_length_mm"],
            "overlap_area_mm2": struct["overlap_area_mm2"],
            "bond_area_mm2": struct["bond_area_mm2"],
            "print_speed_mm_s": speed,
            "atomizer_voltage_v": atomizer,
            "carrier_flow_sccm": carrier,
            "sheath_flow_sccm": sheath,
            "substrate_temp_c": substrate_temp,
            "cure_peak_temp_c": cure_peak,
            "cure_time_min": cure_time,
            "aging_temp_c": aging_temp,
            "aging_hours": aging_hours,
            "cycle_low_temp_c": cycle_low,
            "cycle_high_temp_c": cycle_high,
            "thermal_cycles": thermal_cycles,
            "bend_radius_mm": bend_radius,
            "strain_pct": strain,
            "strain_cycles": strain_cycles,
            "ct_void_fraction_pct": ct_void_fraction,
            "oxidation_index": oxidation_index,
            "edge_roughness_um": edge_roughness,
            "alignment_error_um": alignment,
            "line_width_um": line_width,
            "thickness_um": thickness,
            "initial_resistance_ohm": initial_resistance,
            "sheet_resistance_drift_pct": sheet_drift,
            "contact_resistance_drift_pct": contact_drift,
            "crack_probability": crack_probability,
            "delamination_area_pct": delamination_area,
            "adhesion_strength_mpa": adhesion,
            "void_fraction_pct": ct_void_fraction,
            "post_aging_shear_mpa": shear,
            "reliability_score": reliability,
            "failure_mode": failure_mode,
        }
    )


def generate_interface_dataset(n: int = 40_000, seed: int | None = 17) -> pd.DataFrame:
    rng = _rng(seed)
    material_idx = _choice_index(rng, n, len(MATERIAL_SETS))
    device_idx = _choice_index(rng, n, len(DEVICE_TYPES))
    mat = _material_arrays(material_idx)
    dev = _device_arrays(device_idx, rng)

    # Reuse process physics from the printed witness structures, then project the
    # same geometry/roughness descriptors onto the blind CPW validation witness.
    pseudo_pattern_idx = np.full(n, 2)
    pattern_base = generate_pattern_dataset(n, seed=(seed or 0) + 101)
    pattern_base["pattern_type"] = np.array([PATTERN_TYPES[i]["id"] for i in pseudo_pattern_idx], dtype=object)

    width_scale = np.ones(n)
    line_width = dev["nominal_width_um"] * (
        1.0
        + (pattern_base["line_width_um"].to_numpy() / pattern_base["nominal_width_um"].to_numpy() - 1.0) * width_scale
        + rng.normal(0.0, 0.012, n)
    )
    thickness = pattern_base["thickness_um"].to_numpy() * 0.96
    resistance = (
        dev["trace_length_mm"] * 1e-3
        / np.maximum((line_width * 1e-6) * (thickness * 1e-6) * mat["conductivity_s_m"], 1e-12)
    )
    resistance *= 1.0 + pattern_base["edge_roughness_um"].to_numpy() / 105.0
    quality = _clip(
        pattern_base["quality_score"].to_numpy()
        - 22.0 * np.abs((line_width - dev["nominal_width_um"]) / dev["nominal_width_um"])
        - 3.5 * pattern_base["overspray_ratio"].to_numpy(),
        0,
        100,
    )

    eps_eff = (mat["dielectric_constant"] + 1.0) / 2.0
    c = 299_792_458.0
    cpw_len_m = _clip(dev["trace_length_mm"], 7.2, 31.0) * 1e-3
    nominal_resonance = c / (2 * cpw_len_m * np.sqrt(eps_eff)) / 1e9
    nominal_resonance = _clip(nominal_resonance, 7.0, 13.5)

    width_error = (line_width - dev["nominal_width_um"]) / dev["nominal_width_um"]
    gap_error = (dev["nominal_gap_um"] - np.where(dev["device_type"] == "cpw", 180.0, 600.0)) / np.maximum(dev["nominal_gap_um"], 1.0)
    rough = pattern_base["edge_roughness_um"].to_numpy()
    overspray = pattern_base["overspray_ratio"].to_numpy()
    clog = pattern_base["clog_ratio"].to_numpy()

    resonance = nominal_resonance * (
        1.0
        - 0.14 * width_error
        + 0.045 * gap_error
        + 0.018 * (mat["dielectric_constant"] - 3.5)
        + rng.normal(0.0, 0.006, n)
    )
    impedance = np.where(
        dev["device_type"] == "cpw",
        50.0
        + 16.0 * ((dev["nominal_gap_um"] / np.maximum(line_width, 1.0)) - 0.50)
        + 8.5 * width_error
        + 0.18 * rough
        + rng.normal(0.0, 1.4, n),
        50.0 + 11.0 * width_error + 0.10 * rough + rng.normal(0.0, 1.8, n),
    )
    impedance = _clip(impedance, 25.0, 88.0)

    return_loss = (
        -31.0
        + 0.34 * np.abs(impedance - 50.0)
        + 19.0 * np.abs(width_error)
        + 7.0 * overspray
        + 8.0 * clog
        + 0.12 * rough
        + rng.normal(0.0, 0.9, n)
    )
    return_loss = _clip(return_loss, -36.0, -4.0)

    insertion_loss = np.where(
        dev["device_type"] == "cpw",
        0.18
        + 0.024 * dev["trace_length_mm"]
        + 0.12 * resistance
        + 0.020 * rough
        + 0.80 * overspray
        + 1.05 * clog
        + rng.normal(0.0, 0.06, n),
        0.05 + 0.016 * resistance + 0.008 * rough + 0.21 * overspray + rng.normal(0.0, 0.035, n),
    )
    insertion_loss = _clip(insertion_loss, 0.02, 8.0)

    resonance_error = np.abs(resonance - nominal_resonance) / nominal_resonance
    yield_probability = (
        0.99
        - 0.0075 * np.maximum(0, -return_loss - 18.0)
        - 0.11 * insertion_loss
        - 1.55 * resonance_error
        - 0.004 * (100.0 - quality)
        - 0.18 * overspray
        - 0.23 * clog
    )
    yield_probability = _clip(yield_probability, 0.02, 0.995)

    return pd.DataFrame(
        {
            "material_set": mat["material_set"],
            "ink": mat["ink"],
            "substrate": mat["substrate"],
            "device_type": dev["device_type"],
            "nominal_width_um": dev["nominal_width_um"],
            "nominal_gap_um": dev["nominal_gap_um"],
            "trace_length_mm": dev["trace_length_mm"],
            "substrate_thickness_mm": dev["substrate_thickness_mm"],
            "dielectric_constant": mat["dielectric_constant"],
            "line_width_um": line_width,
            "thickness_um": thickness,
            "resistance_ohm": resistance,
            "quality_score": quality,
            "overspray_ratio": overspray,
            "clog_ratio": clog,
            "edge_roughness_um": rough,
            "print_speed_mm_s": pattern_base["print_speed_mm_s"].to_numpy(),
            "deposition_rate_ug_s": pattern_base["deposition_rate_ug_s"].to_numpy(),
            "atomizer_voltage_v": pattern_base["atomizer_voltage_v"].to_numpy(),
            "carrier_flow_sccm": pattern_base["carrier_flow_sccm"].to_numpy(),
            "sheath_flow_sccm": pattern_base["sheath_flow_sccm"].to_numpy(),
            "resonance_frequency_ghz": resonance,
            "return_loss_db": return_loss,
            "insertion_loss_db": insertion_loss,
            "impedance_ohm": impedance,
            "yield_probability": yield_probability,
        }
    )


def default_pattern_payload() -> dict[str, Any]:
    material = MATERIAL_SETS[0]
    pattern = PATTERN_TYPES[2]
    nominal_width = pattern["nominal_width_um"]
    d_ideal = nominal_width * 1.12
    return {
        "material_set": material.set_id,
        "ink": material.ink,
        "substrate": material.substrate,
        "pattern_type": pattern["id"],
        "nominal_width_um": nominal_width,
        "nominal_pitch_um": pattern["nominal_pitch_um"],
        "nominal_thickness_um": pattern["nominal_thickness_um"],
        "path_length_mm": pattern["path_length_mm"],
        "print_speed_mm_s": 14.5,
        "atomizer_voltage_v": 35.0,
        "carrier_flow_sccm": 28.0,
        "sheath_flow_sccm": 60.0,
        "substrate_temp_c": 55.0,
        "standoff_mm": 3.2,
        "humidity_pct": 34.0,
        "deposition_rate_ug_s": 22.0,
        "d_ideal_um": d_ideal,
        "d_min_um": d_ideal * 0.96,
        "d_max_um": d_ideal * 1.06,
        "blob_count": 0,
        "blob_area_ratio": 0.0,
        "boundary_angle_deg": 0.0,
        "edge_roughness_um": 4.5,
        "camera_noise": 0.02,
        "cure_energy_j_cm2": 1.4,
        "viscosity_cp": material.viscosity_cp,
        "surface_energy_mn_m": material.surface_energy_mn_m,
    }


def default_interface_payload(pattern_prediction: dict[str, float] | None = None) -> dict[str, Any]:
    material = MATERIAL_SETS[0]
    device = DEVICE_TYPES[0]
    if pattern_prediction is None:
        pattern_prediction = {
            "line_width_um": device["nominal_width_um"],
            "thickness_um": 3.4,
            "resistance_ohm": 0.28,
            "quality_score": 91.0,
            "overspray_ratio": 0.06,
            "clog_ratio": 0.02,
            "edge_roughness_um": 5.5,
            "print_speed_mm_s": 14.5,
            "deposition_rate_ug_s": 22.0,
            "atomizer_voltage_v": 35.0,
            "carrier_flow_sccm": 28.0,
            "sheath_flow_sccm": 60.0,
        }
    return {
        "material_set": material.set_id,
        "ink": material.ink,
        "substrate": material.substrate,
        "device_type": device["id"],
        "nominal_width_um": device["nominal_width_um"],
        "nominal_gap_um": device["nominal_gap_um"],
        "trace_length_mm": device["trace_length_mm"],
        "substrate_thickness_mm": device["substrate_thickness_mm"],
        "dielectric_constant": material.dielectric_constant,
        **pattern_prediction,
    }


def default_coupon_payload() -> dict[str, Any]:
    structure = COUPON_STRUCTURES[4]
    ink = COUPON_INKS[1]
    return {
        "coupon_zone": structure["zone"],
        "coupon_structure": structure["id"],
        "measurement_family": structure["measurement_family"],
        "ink_family": ink["id"],
        "substrate": "single_alumina",
        "test_method": "thermal_cycling",
        "nominal_width_um": structure["nominal_width_um"],
        "path_length_mm": structure["path_length_mm"],
        "overlap_area_mm2": structure["overlap_area_mm2"],
        "bond_area_mm2": structure["bond_area_mm2"],
        "print_speed_mm_s": 14.5,
        "atomizer_voltage_v": 35.0,
        "carrier_flow_sccm": 28.0,
        "sheath_flow_sccm": 60.0,
        "substrate_temp_c": 55.0,
        "cure_peak_temp_c": 430.0,
        "cure_time_min": 45.0,
        "aging_temp_c": 350.0,
        "aging_hours": 168.0,
        "cycle_low_temp_c": -40.0,
        "cycle_high_temp_c": 125.0,
        "thermal_cycles": 250,
        "bend_radius_mm": 28.0,
        "strain_pct": 0.65,
        "strain_cycles": 1000,
        "ct_void_fraction_pct": 4.0,
        "oxidation_index": 0.24,
        "edge_roughness_um": 5.8,
        "alignment_error_um": 18.0,
        "line_width_um": structure["nominal_width_um"],
        "thickness_um": 3.2,
        "initial_resistance_ohm": 0.35,
    }


def complete_pattern_payload(payload: dict[str, Any]) -> dict[str, Any]:
    base = default_pattern_payload()
    base.update({k: v for k, v in payload.items() if v is not None})
    material = next((m for m in MATERIAL_SETS if m.set_id == base["material_set"]), MATERIAL_SETS[0])
    pattern = next((p for p in PATTERN_TYPES if p["id"] == base["pattern_type"]), PATTERN_TYPES[2])
    base["ink"] = material.ink
    base["substrate"] = material.substrate
    base["viscosity_cp"] = material.viscosity_cp
    base["surface_energy_mn_m"] = material.surface_energy_mn_m
    for key in ("nominal_width_um", "nominal_pitch_um", "nominal_thickness_um", "path_length_mm"):
        if key not in payload:
            base[key] = pattern[key]
    if "d_ideal_um" not in payload:
        base["d_ideal_um"] = base["nominal_width_um"] * 1.12
    if "d_max_um" not in payload:
        base["d_max_um"] = base["d_ideal_um"] * 1.06
    if "d_min_um" not in payload:
        base["d_min_um"] = base["d_ideal_um"] * 0.96
    return {k: base[k] for k in PATTERN_FEATURES}


def complete_interface_payload(payload: dict[str, Any]) -> dict[str, Any]:
    base = default_interface_payload()
    base.update({k: v for k, v in payload.items() if v is not None})
    material = next((m for m in MATERIAL_SETS if m.set_id == base["material_set"]), MATERIAL_SETS[0])
    device = next((d for d in DEVICE_TYPES if d["id"] == base["device_type"]), DEVICE_TYPES[0])
    base["ink"] = material.ink
    base["substrate"] = material.substrate
    base["dielectric_constant"] = material.dielectric_constant
    for key in ("nominal_width_um", "nominal_gap_um", "trace_length_mm", "substrate_thickness_mm"):
        if key not in payload:
            base[key] = device[key]
    return {k: base[k] for k in INTERFACE_FEATURES}


def complete_coupon_payload(payload: dict[str, Any]) -> dict[str, Any]:
    base = default_coupon_payload()
    base.update({k: v for k, v in payload.items() if v is not None})
    structure = next((s for s in COUPON_STRUCTURES if s["id"] == base["coupon_structure"]), COUPON_STRUCTURES[4])
    ink = next((i for i in COUPON_INKS if i["id"] == base["ink_family"]), COUPON_INKS[1])
    base["coupon_zone"] = structure["zone"]
    base["measurement_family"] = structure["measurement_family"]
    base["substrate"] = "single_alumina"
    for key in ("nominal_width_um", "path_length_mm", "overlap_area_mm2", "bond_area_mm2"):
        if key not in payload:
            base[key] = structure[key]
    if "line_width_um" not in payload:
        base["line_width_um"] = base["nominal_width_um"]
    if "cure_peak_temp_c" not in payload:
        base["cure_peak_temp_c"] = min(500.0, max(180.0, ink["temp_rating_c"] - 70.0))
    return {k: base[k] for k in COUPON_FEATURES}
