from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    import xgboost as xgb
    from sklearn.metrics import f1_score, precision_score, recall_score
except ImportError as exc:
    raise SystemExit(
        "Install dependencies first: pip install geopandas xgboost scikit-learn pandas"
    ) from exc

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.45
SURFACE_FEATURES = [
    "elevation_m", "slope_deg", "ndvi", "iron_oxide_ratio",
    "fe_mn_swir_ratio", "alteration_index", "bare_ground_index",
]
SUBSURFACE_FEATURES = [
    "depth_to_ore_m", "ore_thickness_m", "bulk_density_t_m3", "mn_grade_pct",
]
FEATURES = SURFACE_FEATURES + SUBSURFACE_FEATURES


def save_metadata(**kwargs):
    with open(OUTPUT_DIR / "prospectivity_model_metadata.json", "w") as f:
        json.dump(kwargs, f, indent=2)


def main():
    enhanced_path = BASE_DIR / "synthetic_reserve_grid_enhanced.csv"
    integrated_path = OUTPUT_DIR / "integrated_reserve_grid_predictions.csv"
    training_path = BASE_DIR / "prospectivity_training.csv"
    drill_path = BASE_DIR / "synthetic_drill_holes.csv"

    if not enhanced_path.exists():
        raise FileNotFoundError(f"Missing {enhanced_path}")

    grid = pd.read_csv(enhanced_path)

    # Best case: true trained deployment model using drill labels + enhanced grid.
    if training_path.exists() and drill_path.exists():
        train = pd.read_csv(training_path)
        drill = pd.read_csv(drill_path)

        drill_gdf = gpd.GeoDataFrame(
            drill,
            geometry=gpd.points_from_xy(drill["lon"], drill["lat"]),
            crs="EPSG:4326",
        ).to_crs("EPSG:32644")
        grid_gdf = gpd.GeoDataFrame(
            grid,
            geometry=gpd.points_from_xy(grid["lon"], grid["lat"]),
            crs="EPSG:4326",
        ).to_crs("EPSG:32644")
        joined = gpd.sjoin_nearest(
            drill_gdf, grid_gdf, how="left", distance_col="distance_to_grid_m"
        )

        available = [c for c in FEATURES if c in joined.columns]
        train_ml = joined[available + ["mn_present"]].dropna().copy()
        if len(train_ml) >= 20 and train_ml["mn_present"].nunique() == 2:
            model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=(train_ml["mn_present"] == 0).sum() /
                                 max((train_ml["mn_present"] == 1).sum(), 1),
                eval_metric="aucpr",
                random_state=42,
            )
            # Simple geographic holdout for an honest prototype check.
            joined["spatial_block"] = (
                (joined["lon_right"] > joined["lon_right"].median()).astype(int)
                + 2 * (joined["lat_right"] > joined["lat_right"].median()).astype(int)
            )
            tr = joined[joined["spatial_block"] != 3].dropna(subset=available + ["mn_present"])
            te = joined[joined["spatial_block"] == 3].dropna(subset=available + ["mn_present"])
            if len(tr) >= 10 and len(te) >= 5:
                model.fit(tr[available], tr["mn_present"].astype(int))
                p = model.predict_proba(te[available])[:, 1]
                pred = (p >= THRESHOLD).astype(int)
                print("Spatial holdout:")
                print("precision:", precision_score(te["mn_present"], pred, zero_division=0))
                print("recall   :", recall_score(te["mn_present"], pred, zero_division=0))
                print("f1       :", f1_score(te["mn_present"], pred, zero_division=0))

            model.fit(train_ml[available], train_ml["mn_present"].astype(int))
            grid["probability"] = model.predict_proba(grid[available])[:, 1]
            grid["model_source"] = "XGBoost trained from drill-hole labels"
            model.save_model(str(OUTPUT_DIR / "prospectivity_model.json"))
            print(f"Saved trained prospectivity model using {len(train_ml)} labeled points.")
        else:
            print("Not enough valid drill-linked training rows; using integrated/evidence fallback.")
    elif integrated_path.exists():
        existing = pd.read_csv(integrated_path)
        keep = [c for c in [
            "lon", "lat", "zone", "probability", "classification", "confidence_tier",
            "potential_ore_tonnes", "potential_mn_tonnes",
            "weighted_potential_ore_tonnes", "weighted_potential_mn_tonnes",
            "mn_grade_pct", "ore_thickness_m", "depth_to_ore_m",
        ] if c in existing.columns]
        grid = existing.merge(existing[keep], on=["lon", "lat"], how="left", suffixes=("", "_dup"))
        grid = grid[[c for c in grid.columns if not c.endswith("_dup")]]
        grid["model_source"] = "integrated notebook output"
        print("Using existing integrated reserve-grid predictions.")
    else:
        surface = grid["surface_evidence_score"].astype(float).clip(0, 1)
        subsurface = grid["subsurface_evidence_score"].astype(float).clip(0, 1)
        grid["probability"] = (0.55 * surface + 0.45 * subsurface).clip(0, 1)
        grid["model_source"] = "surface+sub-surface evidence score"
        print("No trained model/integrated output found. Using transparent evidence score.")

    grid["classification"] = np.where(
        grid["probability"] >= THRESHOLD, "probable", "improbable"
    )
    if "confidence_tier" not in grid.columns:
        grid["confidence_tier"] = pd.cut(
            grid["probability"],
            bins=[-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf],
            labels=["very_low", "low", "medium", "high", "very_high"],
        ).astype(str)

    # Potential ore calculation is retained/created when the necessary fields exist.
    if "potential_ore_tonnes" in grid.columns:
        grid["weighted_potential_ore_tonnes"] = (
            grid["potential_ore_tonnes"].astype(float) * grid["probability"]
        )
    if "potential_mn_tonnes" in grid.columns:
        grid["weighted_potential_mn_tonnes"] = (
            grid["potential_mn_tonnes"].astype(float) * grid["probability"]
        )

    out_cols = [c for c in [
        "lon", "lat", "zone", "probability", "classification", "confidence_tier",
        "potential_ore_tonnes", "potential_mn_tonnes",
        "weighted_potential_ore_tonnes", "weighted_potential_mn_tonnes",
        "mn_grade_pct", "ore_thickness_m", "depth_to_ore_m", "model_source",
    ] if c in grid.columns]
    grid[out_cols].to_csv(OUTPUT_DIR / "prospectivity_grid_predictions.csv", index=False)
    grid[out_cols].to_csv(OUTPUT_DIR / "integrated_reserve_grid_predictions.csv", index=False)

    save_metadata(
        threshold=THRESHOLD,
        probability_semantics="prototype prospectivity score; not a certified mineral-reserve probability",
        features=SURFACE_FEATURES + SUBSURFACE_FEATURES,
    )
    print(f"Saved {len(grid):,} grid points with threshold {THRESHOLD}.")


if __name__ == "__main__":
    main()
