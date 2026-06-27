from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlflex.modeling import (  # noqa: E402
    benchmark_regressors,
    save_bundle,
    train_coupon_model,
    train_interface_model,
    train_pattern_models,
)
from mlflex.synthetic import (  # noqa: E402
    COUPON_FEATURES,
    COUPON_TARGETS,
    INTERFACE_FEATURES,
    INTERFACE_TARGETS,
    PATTERN_FEATURES,
    PATTERN_TARGETS,
    generate_coupon_dataset,
    generate_interface_dataset,
    generate_pattern_dataset,
    metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train synthetic BOND-AI interface and bonding coupon models.")
    parser.add_argument("--pattern-samples", type=int, default=80_000)
    parser.add_argument("--interface-samples", type=int, default=40_000)
    parser.add_argument("--coupon-samples", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "model_artifacts")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--write-full-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.pattern_samples:,} synthetic pattern rows...")
    pattern_df = generate_pattern_dataset(args.pattern_samples, seed=args.seed)
    print(f"Generating {args.interface_samples:,} synthetic interface rows...")
    interface_df = generate_interface_dataset(args.interface_samples, seed=args.seed + 10)
    print(f"Generating {args.coupon_samples:,} integrated coupon reliability rows...")
    coupon_df = generate_coupon_dataset(args.coupon_samples, seed=args.seed + 15)

    print("Training pattern defect + trace models...")
    pattern = train_pattern_models(pattern_df, seed=args.seed + 20)
    print("Training interface performance model...")
    interface = train_interface_model(interface_df, seed=args.seed + 30)
    print("Training integrated coupon reliability + failure models...")
    coupon = train_coupon_model(coupon_df, seed=args.seed + 35)

    print("Benchmarking model families on sampled folds...")
    benchmarks = {
        "pattern_trace": benchmark_regressors(pattern_df, PATTERN_FEATURES, PATTERN_TARGETS, seed=args.seed + 40),
        "cpw_validation": benchmark_regressors(interface_df, INTERFACE_FEATURES, INTERFACE_TARGETS, seed=args.seed + 50),
        "coupon_reliability": benchmark_regressors(coupon_df, COUPON_FEATURES, COUPON_TARGETS, seed=args.seed + 60),
    }

    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_rows": {
            "pattern": int(len(pattern_df)),
            "interface": int(len(interface_df)),
            "coupon": int(len(coupon_df)),
            "total": int(len(pattern_df) + len(interface_df) + len(coupon_df)),
        },
        "pattern": pattern["metrics"],
        "interface": interface["metrics"],
        "coupon": coupon["metrics"],
        "benchmarks": benchmarks,
        "model_card": {
            "served_models": {
                "print_process_classifier": "HistGradientBoostingClassifier for process anomaly states that feed coupon geometry descriptors",
                "trace_regressor": "HistGradientBoostingRegressor multi-output wrapper for coupon line width, thickness, resistance, and quality",
                "cpw_validation_regressor": "HistGradientBoostingRegressor surrogate for blind CPW validation metrics on alumina",
                "coupon_digital_twin": "Tree ensemble reliability surrogate for Zone A interface and Zone B bonding structures",
            },
            "research_upgrade_path": [
                "Physics-informed tabular learning after coupon data collection",
                "Visual encoders for optical, SEM, EDS, FIB, and CT inspection imagery",
                "Temporal models for in-situ electrical drift during aging and thermal cycling",
                "Uncertainty-guided active learning for next coupon and condition selection",
            ],
        },
    }

    bundle = {
        "pattern": {
            "classifier": pattern["classifier"],
            "regressor": pattern["regressor"],
            "metrics": pattern["metrics"],
            "features": pattern["features"],
            "targets": pattern["targets"],
        },
        "interface": {
            "model": interface["model"],
            "metrics": interface["metrics"],
            "features": interface["features"],
            "targets": interface["targets"],
        },
        "coupon": {
            "regressor": coupon["regressor"],
            "classifier": coupon["classifier"],
            "metrics": coupon["metrics"],
            "features": coupon["features"],
            "targets": coupon["targets"],
        },
        "metrics": metrics,
        "metadata": metadata(),
    }
    save_bundle(bundle, args.artifact_dir)

    preview = {
            "pattern": pattern_df.sample(min(750, len(pattern_df)), random_state=args.seed).to_dict(orient="records"),
            "interface": interface_df.sample(min(500, len(interface_df)), random_state=args.seed + 1).to_dict(orient="records"),
            "coupon": coupon_df.sample(min(900, len(coupon_df)), random_state=args.seed + 2).to_dict(orient="records"),
        }
    (args.data_dir / "synthetic_preview.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")
    summary = {
        "rows": metrics["synthetic_rows"],
        "pattern_defect_distribution": pattern_df["defect_class"].value_counts().to_dict(),
        "pattern_type_distribution": pattern_df["pattern_type"].value_counts().to_dict(),
        "material_set_distribution": pattern_df["material_set"].value_counts().to_dict(),
        "device_type_distribution": interface_df["device_type"].value_counts().to_dict(),
        "coupon_structure_distribution": coupon_df["coupon_structure"].value_counts().to_dict(),
        "coupon_failure_distribution": coupon_df["failure_mode"].value_counts().to_dict(),
        "coupon_ink_distribution": coupon_df["ink_family"].value_counts().to_dict(),
        "target_ranges": {
            "pattern": pattern_df[PATTERN_TARGETS].agg(["min", "mean", "max"]).to_dict(),
            "interface": interface_df[INTERFACE_TARGETS].agg(["min", "mean", "max"]).to_dict(),
            "coupon": coupon_df[COUPON_TARGETS].agg(["min", "mean", "max"]).to_dict(),
        },
    }
    (args.data_dir / "synthetic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.write_full_data:
        pattern_df.to_parquet(args.data_dir / "synthetic_patterns.parquet", index=False)
        interface_df.to_parquet(args.data_dir / "synthetic_interfaces.parquet", index=False)
        coupon_df.to_parquet(args.data_dir / "synthetic_coupons.parquet", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
