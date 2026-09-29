from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import xgboost as xgb
except Exception:
    xgb = None

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

SHORTFALL_PATH = OUTPUT_DIR / "shortfall_predictions.csv"
RECOMMENDATIONS_PATH = OUTPUT_DIR / "preventive_recommendations.csv"
ENV_PATH = OUTPUT_DIR / "environment_zone_summary.csv"
EQUIPMENT_PATH = OUTPUT_DIR / "equipment_zone_summary.csv"
BLASTING_PATH = OUTPUT_DIR / "blasting_zone_summary.csv"
RESERVE_SUMMARY_PATH = OUTPUT_DIR / "reserve_estimates_by_zone.csv"
RESERVE_GRID_PATH = OUTPUT_DIR / "integrated_reserve_grid_predictions.csv"
HISTORY_PATH = OUTPUT_DIR / "production_history_dashboard.csv"
PROD_MODEL_PATH = OUTPUT_DIR / "production_model.json"
PROD_METRICS_PATH = OUTPUT_DIR / "production_model_metrics.json"
PROD_IMPORTANCE_PATH = OUTPUT_DIR / "production_model_feature_importance.csv"
AREA_BOUNDS_PATH = OUTPUT_DIR / "study_area_bounds.json"

ENV_INPUT = BASE_DIR / "synthetic_environmental.csv"
EQUIPMENT_INPUT = BASE_DIR / "synthetic_equipment.csv"
BLASTING_INPUT = BASE_DIR / "synthetic_blasting.csv"
TARGET_INPUT = BASE_DIR / "synthetic_production_targets.csv"
DRILL_INPUT = BASE_DIR / "synthetic_drill_holes.csv"
OVERRIDES_PATH = OUTPUT_DIR / "manual_overrides.csv"

PROSPECTIVITY_THRESHOLD = 0.45
VALID_HORIZONS = (30, 60, 90)

app = FastAPI(
    title="Manganum AI — SIH26009 Decision Support API",
    description=(
        "Integrated manganese reserve/prospectivity, environmental, equipment, "
        "blasting, production shortfall and preventive-action backend."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA: Dict[str, Any] = {}


class OverrideSubmission(BaseModel):
    lon: float
    lat: float
    mn_present: int = Field(ge=0, le=1)
    note: Optional[str] = None
    submitted_by: str = "geologist"


class ForecastResponse(BaseModel):
    zone: str
    horizon_days: int
    forecast: List[Dict[str, Any]]


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _zone_or_404(zone: str) -> None:
    if zone not in DATA["zones"]:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown zone '{zone}'. Valid zones: {DATA['zones']}",
        )


def _latest_row(df: pd.DataFrame, zone: str) -> pd.Series:
    z = df[df["zone"] == zone].sort_values("date")
    if z.empty:
        raise HTTPException(status_code=404, detail=f"No data available for {zone}")
    return z.iloc[-1]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _top_equipment(zone: str) -> Optional[Dict[str, Any]]:
    df = DATA.get("equipment_detail")
    if df is None or df.empty:
        return None
    z = df[df["zone"] == zone].copy()
    if z.empty:
        return None
    g = (
        z.groupby(["equipment_id", "equipment_type"], as_index=False)
        .agg(
            downtime_hours=("downtime_hours", "sum"),
            availability=("availability_pct", "mean"),
            utilization=("utilization_pct", "mean"),
        )
        .sort_values("downtime_hours", ascending=False)
    )
    if g.empty:
        return None
    row = g.iloc[0].to_dict()
    return {
        "equipment_id": row["equipment_id"],
        "equipment_type": row["equipment_type"],
        "downtime_hours": round(_safe_float(row["downtime_hours"]), 2),
        "availability_pct": round(_safe_float(row["availability"]) * 100, 2),
        "utilization_pct": round(_safe_float(row["utilization"]) * 100, 2),
    }


def _driver_analysis(zone: str) -> Dict[str, Any]:
    env = DATA["environment"].set_index("zone").loc[zone]
    eq = DATA["equipment"].set_index("zone").loc[zone]
    bl = DATA["blasting"].set_index("zone").loc[zone]

    scores = {
        "equipment": _safe_float(eq.get("equipment_risk_score")),
        "blasting": _safe_float(bl.get("blast_risk_score")),
        "environment": _safe_float(env.get("environment_risk_score")),
    }
    total = sum(max(v, 0.0) for v in scores.values()) or 1.0
    contributions = {
        k: round(max(v, 0.0) / total * 100.0, 1) for k, v in scores.items()
    }
    return {
        "scores": scores,
        "contributions_pct": contributions,
        "environment_reasons": str(env.get("environment_reasons", "")),
        "equipment_reasons": str(eq.get("equipment_reasons", "")),
        "blast_reasons": str(bl.get("blast_reasons", "")),
    }


def _actions(zone: str) -> List[str]:
    rec = DATA["recommendations"]
    row = rec[rec["zone"] == zone]
    if not row.empty:
        raw = str(row.iloc[0].get("recommended_actions", ""))
        actions = [x.strip() for x in raw.split("|") if x.strip()]
        if actions:
            return actions

    short = DATA["shortfall"].set_index("zone").loc[zone]
    env = DATA["environment"].set_index("zone").loc[zone]
    eq = DATA["equipment"].set_index("zone").loc[zone]
    bl = DATA["blasting"].set_index("zone").loc[zone]
    actions: List[str] = []
    if _safe_float(short.get("shortfall_pct")) >= 15:
        actions.append("Prioritize high-prospectivity ore blocks in the near-term mine schedule.")
    elif _safe_float(short.get("shortfall_pct")) >= 5:
        actions.append("Review the next planning cycle and keep a production buffer.")
    if str(eq.get("equipment_risk")) == "HIGH":
        top = _top_equipment(zone)
        if top:
            actions.append(
                f"Schedule preventive maintenance for {top['equipment_id']} and assess redeployment of available backup equipment."
            )
        else:
            actions.append("Schedule preventive maintenance and review equipment allocation.")
    if str(bl.get("blast_risk")) == "HIGH":
        actions.append("Re-plan blasting windows and verify drill/haulage readiness before the blast.")
    elif str(bl.get("blast_risk")) == "MEDIUM":
        actions.append("Review recent blast delays and prepare schedule slack for the next cycle.")
    if str(env.get("environment_risk")) == "HIGH":
        actions.append("Adjust mine/haulage scheduling for the environmental risk window and maintain contingency capacity.")
    elif str(env.get("environment_risk")) == "MEDIUM":
        actions.append("Monitor environmental conditions daily and keep a schedule buffer.")
    return actions or ["Continue normal operations and monitor the next forecast update."]


def _build_history(zone: str, days: int = 90) -> pd.DataFrame:
    h = DATA["history"]
    z = h[h["zone"] == zone].sort_values("date").tail(days).copy()
    return z


def _recompute_model_prediction(zone: str) -> Optional[Dict[str, Any]]:
    model = DATA.get("production_model")
    if model is None:
        return None
    h = DATA["history"]
    z = h[h["zone"] == zone].sort_values("date").copy()
    if z.empty:
        return None

    feature_names = DATA["production_metrics"].get("prediction_features", [])
    if not feature_names:
        return None
    row = z.iloc[-1]
    missing = [c for c in feature_names if c not in row.index]
    if missing:
        return None
    X = pd.DataFrame([{c: _safe_float(row[c]) for c in feature_names}])
    predicted = float(model.predict(X)[0])
    target = _safe_float(row.get("planned_production_tonnes"))
    minimum = _safe_float(row.get("minimum_required_tonnes"))
    shortfall = max(target - predicted, 0.0)
    shortfall_pct = (shortfall / target * 100.0) if target else 0.0
    risk = "HIGH" if shortfall_pct >= 15 else "MEDIUM" if shortfall_pct >= 5 else "LOW"
    return {
        "zone": zone,
        "feature_date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
        "predicted_next_day_tonnes": round(max(predicted, 0.0), 2),
        "planned_tonnes": round(target, 2),
        "minimum_required_tonnes": round(minimum, 2),
        "shortfall_tonnes": round(shortfall, 2),
        "shortfall_pct": round(shortfall_pct, 2),
        "shortfall_risk": risk,
        "prediction_source": "saved XGBoost production_model.json",
    }


@app.on_event("startup")
def load_data() -> None:
    required = [
        SHORTFALL_PATH, RECOMMENDATIONS_PATH, ENV_PATH, EQUIPMENT_PATH,
        BLASTING_PATH, RESERVE_SUMMARY_PATH, RESERVE_GRID_PATH, HISTORY_PATH
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "Missing notebook outputs. Execute code_SIH26009_final.ipynb artifact cell first.\n"
            + "\n".join(missing)
        )

    DATA["shortfall"] = read_csv(SHORTFALL_PATH)
    DATA["recommendations"] = read_csv(RECOMMENDATIONS_PATH)
    DATA["environment"] = read_csv(ENV_PATH)
    DATA["equipment"] = read_csv(EQUIPMENT_PATH)
    DATA["blasting"] = read_csv(BLASTING_PATH)
    DATA["reserve_summary"] = read_csv(RESERVE_SUMMARY_PATH)
    DATA["reserve_grid"] = read_csv(RESERVE_GRID_PATH)
    DATA["history"] = read_csv(HISTORY_PATH, parse_dates=["date"])

    if PROD_METRICS_PATH.exists():
        DATA["production_metrics"] = json.loads(PROD_METRICS_PATH.read_text())
    else:
        DATA["production_metrics"] = {}

    if PROD_IMPORTANCE_PATH.exists():
        DATA["production_importance"] = read_csv(PROD_IMPORTANCE_PATH)
    else:
        DATA["production_importance"] = pd.DataFrame()

    if AREA_BOUNDS_PATH.exists():
        DATA["area_bounds"] = json.loads(AREA_BOUNDS_PATH.read_text())
    else:
        grid_df = DATA["reserve_grid"]
        DATA["area_bounds"] = {
            "lon_min": float(grid_df["lon"].min()),
            "lon_max": float(grid_df["lon"].max()),
            "lat_min": float(grid_df["lat"].min()),
            "lat_max": float(grid_df["lat"].max()),
        }

    DATA["zones"] = sorted(DATA["shortfall"]["zone"].astype(str).unique().tolist())
    for key in ["environment", "equipment", "blasting", "recommendations", "reserve_summary", "reserve_grid"]:
        DATA[key]["zone"] = DATA[key]["zone"].astype(str)

    DATA["environment_detail"] = read_csv(ENV_INPUT, parse_dates=["date"]) if ENV_INPUT.exists() else pd.DataFrame()
    DATA["equipment_detail"] = read_csv(EQUIPMENT_INPUT, parse_dates=["date"]) if EQUIPMENT_INPUT.exists() else pd.DataFrame()
    DATA["blasting_detail"] = read_csv(BLASTING_INPUT, parse_dates=["date"]) if BLASTING_INPUT.exists() else pd.DataFrame()
    DATA["target_detail"] = read_csv(TARGET_INPUT, parse_dates=["date"]) if TARGET_INPUT.exists() else pd.DataFrame()
    DATA["drill_holes"] = read_csv(DRILL_INPUT) if DRILL_INPUT.exists() else pd.DataFrame()

    DATA["production_model"] = None
    if xgb is not None and PROD_MODEL_PATH.exists():
        model = xgb.XGBRegressor()
        model.load_model(str(PROD_MODEL_PATH))
        DATA["production_model"] = model

    print(f"SIH26009 API loaded. Zones: {DATA['zones']}")
    print(f"Production model loaded: {DATA['production_model'] is not None}")


@app.get("/", tags=["status"])
def root() -> Dict[str, Any]:
    return {
        "message": "Manganum AI — SIH26009 Decision Support API",
        "zones": DATA.get("zones", []),
        "model_loaded": DATA.get("production_model") is not None,
        "prospectivity_threshold": PROSPECTIVITY_THRESHOLD,
        "docs": "/docs",
    }


@app.get("/health", tags=["status"])
def health() -> Dict[str, Any]:
    required_keys = [
        "shortfall", "recommendations", "environment", "equipment",
        "blasting", "reserve_summary", "reserve_grid", "history"
    ]
    return {
        "status": "ok",
        "artifacts_loaded": all(k in DATA for k in required_keys),
        "production_model_loaded": DATA.get("production_model") is not None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@app.get("/area", tags=["area"])
def get_area() -> Dict[str, Any]:
    grid = DATA["reserve_grid"]
    return {
        "name": "SIH26009 prototype study area",
        "bounds": DATA["area_bounds"],
        "grid_points": int(len(grid)),
        "zones": DATA["zones"],
        "source": "synthetic reserve grid for prototype demonstration",
    }


@app.get("/environment", tags=["environment"])
def get_environment(zone: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    df = DATA["environment"]
    if zone:
        _zone_or_404(zone)
        return {"zone": zone, "summary": _records(df[df.zone == zone])}
    return {"zones": _records(df)}


@app.get("/equipment", tags=["equipment"])
def get_equipment(zone: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    df = DATA["equipment"]
    if zone:
        _zone_or_404(zone)
        return {
            "zone": zone,
            "summary": _records(df[df.zone == zone]),
            "top_unit": _top_equipment(zone),
        }
    return {"zones": _records(df)}


@app.get("/blasting", tags=["blasting"])
def get_blasting(zone: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    df = DATA["blasting"]
    if zone:
        _zone_or_404(zone)
        recent_events = DATA["blasting_detail"]
        if not recent_events.empty:
            recent_events = recent_events[recent_events["zone"] == zone].sort_values("date").tail(12)
        return {
            "zone": zone,
            "summary": _records(df[df.zone == zone]),
            "recent_events": _records(recent_events),
        }
    return {"zones": _records(df)}


@app.get("/production-target/{zone}", tags=["production"])
def get_target(zone: str) -> Dict[str, Any]:
    _zone_or_404(zone)
    latest = _latest_row(DATA["target_detail"], zone)
    return {
        "zone": zone,
        "date": pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"),
        "planned_production_tonnes": round(_safe_float(latest["planned_production_tonnes"]), 2),
        "minimum_required_tonnes": round(_safe_float(latest["minimum_required_tonnes"]), 2),
    }


@app.get("/production-history/{zone}", tags=["production"])
def get_history(zone: str, days: int = Query(default=90, ge=7, le=365)) -> Dict[str, Any]:
    _zone_or_404(zone)
    return {"zone": zone, "history": _records(_build_history(zone, days))}


@app.get("/prediction/{zone}", tags=["prediction"])
def get_prediction(zone: str) -> Dict[str, Any]:
    _zone_or_404(zone)
    recomputed = _recompute_model_prediction(zone)
    stored = DATA["shortfall"]
    stored_row = stored[stored.zone == zone]
    stored_payload = _records(stored_row)[0] if not stored_row.empty else None
    return {
        "zone": zone,
        "model_prediction": recomputed,
        "stored_notebook_prediction": stored_payload,
        "model_loaded": DATA["production_model"] is not None,
    }


@app.get("/shortfall/{zone}", tags=["prediction"])
def get_shortfall(zone: str) -> Dict[str, Any]:
    _zone_or_404(zone)
    row = DATA["shortfall"][DATA["shortfall"].zone == zone]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"No shortfall prediction for {zone}")
    payload = _records(row)[0]
    payload["driver_analysis"] = _driver_analysis(zone)
    return payload


@app.get("/forecast/{zone}", response_model=ForecastResponse, tags=["prediction"])
def get_forecast(zone: str, days: int = Query(default=1, ge=1, le=90)) -> ForecastResponse:
    _zone_or_404(zone)
    h = _build_history(zone, 90)
    model_pred = _recompute_model_prediction(zone)
    base_date = pd.Timestamp(h.iloc[-1]["date"])
    if model_pred is None:
        raise HTTPException(status_code=503, detail="Saved production model is not available.")
    if days == 1:
        pts = [{
            "date": (base_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "forecast_tonnes": model_pred["predicted_next_day_tonnes"],
        }]
    else:
        pts = []
        for i in range(1, days + 1):
            decay = 1.0 - min(i - 1, 30) * 0.002
            pts.append({
                "date": (base_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                "forecast_tonnes": round(model_pred["predicted_next_day_tonnes"] * decay, 2),
            })
    return ForecastResponse(zone=zone, horizon_days=days, forecast=pts)


@app.get("/recommendation/{zone}", tags=["decision-support"])
def get_recommendation(zone: str) -> Dict[str, Any]:
    _zone_or_404(zone)
    short = DATA["shortfall"].set_index("zone").loc[zone].to_dict()
    env = DATA["environment"].set_index("zone").loc[zone].to_dict()
    eq = DATA["equipment"].set_index("zone").loc[zone].to_dict()
    bl = DATA["blasting"].set_index("zone").loc[zone].to_dict()
    rec_rows = DATA["recommendations"][DATA["recommendations"].zone == zone]
    stored = rec_rows.iloc[0].to_dict() if not rec_rows.empty else {}
    return {
        "zone": zone,
        "risk": short.get("shortfall_risk"),
        "shortfall_pct": _safe_float(short.get("shortfall_pct")),
        "predicted_next_day_tonnes": _safe_float(short.get("predicted_next_day_tonnes")),
        "planned_tonnes": _safe_float(short.get("planned_tonnes")),
        "environment": env,
        "equipment": eq,
        "blasting": bl,
        "driver_analysis": _driver_analysis(zone),
        "top_equipment": _top_equipment(zone),
        "recommended_actions": _actions(zone),
        "notebook_recommendation": stored.get("recommended_actions"),
    }


@app.get("/reserve-summary", tags=["reserve"])
def reserve_summary(zone: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    df = DATA["reserve_summary"]
    if zone:
        _zone_or_404(zone)
        row = df[df.zone == zone]
        if row.empty:
            raise HTTPException(status_code=404, detail=f"No reserve estimate for {zone}")
        return {"zone": zone, "estimate": _records(row)[0], "threshold": PROSPECTIVITY_THRESHOLD}
    return {"zones": _records(df), "threshold": PROSPECTIVITY_THRESHOLD}


@app.get("/reserve-map", tags=["reserve"])
def reserve_map(zone: Optional[str] = Query(default=None), limit: int = Query(default=14400, ge=100, le=14400)) -> Dict[str, Any]:
    df = DATA["reserve_grid"]
    if zone:
        _zone_or_404(zone)
        df = df[df.zone == zone]
    return {
        "zone": zone or "ALL",
        "threshold": PROSPECTIVITY_THRESHOLD,
        "points": _records(df.head(limit)),
    }


@app.get("/drill-holes", tags=["reserve"])
def drill_holes(zone: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    df = DATA["drill_holes"]
    if df.empty:
        return {"available": False, "holes": []}
    if zone and "zone" in df.columns:
        df = df[df.zone == zone]
    return {"available": True, "holes": _records(df)}


@app.get("/model-info", tags=["models"])
def model_info() -> Dict[str, Any]:
    return {
        "production_model_loaded": DATA["production_model"] is not None,
        "production_metrics": DATA["production_metrics"],
        "production_feature_importance": _records(DATA["production_importance"]),
        "prospectivity_threshold": PROSPECTIVITY_THRESHOLD,
    }


def _load_overrides() -> pd.DataFrame:
    cols = ["override_id", "lon", "lat", "mn_present", "note", "submitted_by", "submitted_at"]
    if not OVERRIDES_PATH.exists():
        return pd.DataFrame(columns=cols)
    return read_csv(OVERRIDES_PATH)


@app.get("/overrides", tags=["human-in-the-loop"])
def get_overrides() -> Dict[str, Any]:
    df = _load_overrides()
    if not df.empty:
        df = df.sort_values("submitted_at", ascending=False)
    return {"total": len(df), "overrides": _records(df)}


@app.post("/overrides", status_code=201, tags=["human-in-the-loop"])
def post_override(item: OverrideSubmission) -> Dict[str, Any]:
    b = DATA["area_bounds"]
    if not (
        b["lon_min"] <= item.lon <= b["lon_max"]
        and b["lat_min"] <= item.lat <= b["lat_max"]
    ):
        raise HTTPException(status_code=422, detail="Coordinates outside study area.")

    df = _load_overrides()
    next_id = int(df["override_id"].max()) + 1 if not df.empty else 1
    record = {
        "override_id": next_id,
        "lon": item.lon,
        "lat": item.lat,
        "mn_present": item.mn_present,
        "note": item.note,
        "submitted_by": item.submitted_by,
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    new_df = pd.DataFrame([record])
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(OVERRIDES_PATH, mode="a", header=not OVERRIDES_PATH.exists(), index=False)
    return record
