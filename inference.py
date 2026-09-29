from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pandas as pd
import scipy.io
from scipy.ndimage import median_filter


BASE_DIR = Path(__file__).resolve().parent
MODEL_PACKAGE_FILE = BASE_DIR / "models" / "SPAD_V4_Final_Model_Package.pkl"

if not MODEL_PACKAGE_FILE.exists():
    raise RuntimeError(f"Model package not found: {MODEL_PACKAGE_FILE}")

with MODEL_PACKAGE_FILE.open("rb") as f:
    MODEL_PACKAGE = pickle.load(f)

if MODEL_PACKAGE.get("package_version") != "SPAD_V4":
    raise RuntimeError("Unsupported SPAD model package version.")

MODULE_A = MODEL_PACKAGE["module_a"]
MODULE_B = MODEL_PACKAGE["module_b"]
MODULE_C = MODEL_PACKAGE["module_c"]
TARGET_CONDITION = tuple(MODEL_PACKAGE["metadata"]["operating_condition"])
PROVISIONAL_MIN_CURRENT_A = float(MODULE_C["provisional_min_current_A"])


def canonical_component_id(test_id: int | str) -> str:
    return f"TEST-{int(test_id):02d}"


def _as_finite_float(value: Any, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def validate_engineering_limits(engineering_limits: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if engineering_limits is None:
        return {}
    if not isinstance(engineering_limits, dict):
        raise ValueError("engineeringLimits must be an object.")

    resolved: dict[str, dict[str, Any]] = {}
    for parameter, spec in engineering_limits.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Engineering limit for '{parameter}' must be an object.")
        if "limitValue" not in spec:
            raise ValueError(f"Engineering limit for '{parameter}' is missing limitValue.")
        value = _as_finite_float(spec["limitValue"], f"engineeringLimits.{parameter}.limitValue")
        direction = str(spec.get("direction", "UPPER")).upper()
        if direction not in {"UPPER", "LOWER"}:
            raise ValueError(f"Unsupported limit direction for '{parameter}': {direction}")
        resolved[parameter] = {
            "limitValue": value,
            "unit": spec.get("unit"),
            "direction": direction,
        }
    return resolved


def _safe_extract_zip(uploaded_bytes: bytes, destination: Path) -> None:
    if len(uploaded_bytes) > 250 * 1024 * 1024:
        raise ValueError("Uploaded dataset ZIP exceeds the 250 MB service limit.")
    try:
        with zipfile.ZipFile(io_bytes := __import__("io").BytesIO(uploaded_bytes)) as zf:
            bad = zf.testzip()
            if bad:
                raise ValueError(f"Corrupt ZIP member: {bad}")
            for info in zf.infolist():
                member = Path(info.filename)
                if member.is_absolute() or ".." in member.parts:
                    raise ValueError("Unsafe ZIP path detected.")
                target = destination / member
                target.parent.mkdir(parents=True, exist_ok=True)
                if not info.is_dir():
                    with zf.open(info, "r") as src, target.open("wb") as dst:
                        dst.write(src.read())
    except zipfile.BadZipFile as exc:
        raise ValueError("Uploaded dataset must be a valid ZIP archive.") from exc


def _find_runlevel_csv(root: Path) -> Path:
    candidates = list(root.rglob("MOSFET_199_200C_V2_Electrical_Features_RunLevel.csv"))
    if not candidates:
        candidates = list(root.rglob("*RunLevel.csv"))
    if not candidates:
        raise ValueError("Uploaded dataset ZIP does not contain a run-level CSV.")
    return candidates[0]


def _load_checkpoint_csv(root: Path) -> pd.DataFrame | None:
    candidates = list(root.rglob("MOSFET_Normalized_4Checkpoints.csv"))
    if not candidates:
        return None
    return pd.read_csv(candidates[0])


def _build_checkpoint_values(runlevel: pd.DataFrame) -> pd.DataFrame:
    checkpoints = [
        (1, 0.000000, "0h_equivalent"),
        (2, 0.333333, "24h_equivalent"),
        (3, 0.666667, "96h_equivalent"),
        (4, 1.000000, "168h_equivalent"),
    ]
    rows: list[dict[str, Any]] = []
    for _, group in runlevel.groupby("Test_ID"):
        group = group.sort_values("Cumulative_Aging_Hours").reset_index(drop=True)
        ages = group["Cumulative_Aging_Hours"].to_numpy(dtype=float)
        rds = group["RDSon_Median_Ohm"].to_numpy(dtype=float)
        if len(group) < 2:
            continue
        span = float(ages.max() - ages.min())
        if span <= 0:
            continue
        for checkpoint_id, progress, label in checkpoints:
            target_age = float(ages.min() + progress * span)
            nearest_idx = int(np.argmin(np.abs(ages - target_age)))
            rows.append({
                "Test_ID": int(group["Test_ID"].iloc[0]),
                "Checkpoint": checkpoint_id,
                "Normalized_Progress": progress,
                "Challenge_Equivalent_Label": label,
                "Observed_Age_Min_Hours": float(ages.min()),
                "Observed_Age_Max_Hours": float(ages.max()),
                "Checkpoint_Age_Equivalent_Hours": target_age,
                "RDSon_Interpolated_Ohm": float(np.interp(target_age, ages, rds)),
                "Nearest_Actual_Run_ID": int(group.loc[nearest_idx, "Run_ID"]),
                "Nearest_Actual_Age_Hours": float(ages[nearest_idx]),
                "Number_of_Actual_Runs": int(len(group)),
            })
    if not rows:
        raise ValueError("Could not construct the 4-checkpoint trajectory from the run-level data.")
    return pd.DataFrame(rows)


def _build_wide(runlevel: pd.DataFrame, checkpoints_df: pd.DataFrame) -> pd.DataFrame:
    required_run = {
        "Test_ID", "Run_ID", "Cumulative_Aging_Hours", "RDSon_Median_Ohm", "ID_ON_Median_A"
    }
    missing = sorted(required_run - set(runlevel.columns))
    if missing:
        raise ValueError("Run-level CSV is missing required columns: " + ", ".join(missing))

    records: list[dict[str, Any]] = []
    for test_id, cp_group in checkpoints_df.groupby("Test_ID"):
        runs = runlevel[runlevel["Test_ID"] == test_id].sort_values("Cumulative_Aging_Hours")
        if len(runs) < 2:
            continue
        age = runs["Cumulative_Aging_Hours"].to_numpy(dtype=float)
        ids = runs["ID_ON_Median_A"].to_numpy(dtype=float)
        if not np.isfinite(age).all() or not np.isfinite(ids).all():
            raise ValueError(f"Non-finite run-level inputs for Test_ID {test_id}.")
        for _, cp in cp_group.iterrows():
            target_age = float(cp["Checkpoint_Age_Equivalent_Hours"])
            records.append({
                "Test_ID": int(test_id),
                "Checkpoint": int(cp["Checkpoint"]),
                "Normalized_Progress": float(cp["Normalized_Progress"]),
                "Checkpoint_Age_Equivalent_Hours": target_age,
                "RDSon_Ohm": float(cp["RDSon_Interpolated_Ohm"]),
                "ID_ON_A": float(np.interp(target_age, age, ids)),
            })

    aligned = pd.DataFrame(records)
    if aligned.empty:
        raise ValueError("No valid component trajectories could be constructed.")

    wide = aligned.pivot(index="Test_ID", columns="Checkpoint", values=["RDSon_Ohm", "ID_ON_A"])
    wide.columns = [f"{feature}_{checkpoint}" for feature, checkpoint in wide.columns]
    wide = wide.reset_index().rename(columns={
        "RDSon_Ohm_1": "RDS0",
        "RDSon_Ohm_2": "RDS33",
        "RDSon_Ohm_3": "RDS66",
        "RDSon_Ohm_4": "RDS100",
        "ID_ON_A_1": "ID0",
        "ID_ON_A_2": "ID33",
    })
    wide["Delta_RDS_0_33"] = wide["RDS33"] - wide["RDS0"]
    wide["Delta_RDS_33_66"] = wide["RDS66"] - wide["RDS33"]
    wide["Delta_RDS_66_100"] = wide["RDS100"] - wide["RDS66"]
    wide["ID33_ID0_Ratio"] = wide["ID33"] / wide["ID0"].replace(0, np.nan)

    required_wide = [
        "Test_ID", "RDS0", "RDS33", "RDS66", "RDS100",
        "ID0", "ID33", "Delta_RDS_0_33", "Delta_RDS_33_66",
        "Delta_RDS_66_100", "ID33_ID0_Ratio",
    ]
    missing = [c for c in required_wide if c not in wide.columns]
    if missing:
        raise ValueError("Trajectory construction missing columns: " + ", ".join(missing))
    return wide


def _module_a_infer(wide: pd.DataFrame) -> None:
    for stage in (0, 33, 66, 100):
        features = MODULE_A["features_by_stage"][stage]
        model = MODULE_A["models"][stage]
        X = wide[features].to_numpy(dtype=float)
        if not np.isfinite(X).all():
            raise ValueError(f"Non-finite Module A input at stage {stage}%.")
        scores = model.decision_function(X)
        flags = np.where(model.predict(X) == -1, "FLAGGED", "NOT FLAGGED")
        reference_scores = np.asarray(MODULE_A["reference_scores"][str(stage)], dtype=float)
        novelty = np.array([100.0 * np.mean(reference_scores >= score) for score in scores])
        wide[f"Module_A_IF_Score_{stage}"] = scores
        wide[f"Module_A_Novelty_Percentile_{stage}"] = novelty
        wide[f"Module_A_Flag_{stage}"] = flags

    stage_flag_cols = [f"Module_A_Flag_{s}" for s in (0, 33, 66, 100)]
    wide["Progressive_ML_Flag"] = np.where(
        wide[stage_flag_cols].eq("FLAGGED").any(axis=1), "FLAGGED", "NOT FLAGGED"
    )
    wide["First_Anomaly_Stage"] = np.nan
    for stage in (0, 33, 66, 100):
        mask = wide["First_Anomaly_Stage"].isna() & wide[f"Module_A_Flag_{stage}"].eq("FLAGGED")
        wide.loc[mask, "First_Anomaly_Stage"] = stage


def _module_b_infer(wide: pd.DataFrame) -> None:
    features = MODULE_B["features"]
    if not all(c in wide.columns for c in features):
        missing = [c for c in features if c not in wide.columns]
        raise ValueError("Missing Module B features: " + ", ".join(missing))
    X = wide[features].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite Module B input.")
    wide["Predicted_RDS100"] = MODULE_B["model"].predict(X)
    wide["Forecast_Residual"] = wide["RDS100"] - wide["Predicted_RDS100"]
    wide["Absolute_Forecast_Error"] = wide["Forecast_Residual"].abs()
    actual_abs = wide["RDS100"].abs()
    wide["Relative_Error_Percent"] = np.where(
        actual_abs > 0,
        wide["Absolute_Forecast_Error"] / actual_abs * 100.0,
        np.nan,
    )


def get_pwm_condition(p: Any) -> tuple[int, int, int, int, int, int] | None:
    values = []
    fields = ("lowTemp", "highTemp", "gateVoltage", "supplyVoltage", "switchingFrequency", "dutyCycle")
    for field in fields:
        value = getattr(p, field, np.nan)
        try:
            value = float(value)
        except Exception:
            value = np.nan
        if not np.isfinite(value):
            return None
        values.append(int(round(value)))
    return tuple(values)  # type: ignore[return-value]


def build_pwm_timeline(measurement: Any) -> tuple[np.ndarray, list[tuple[int, int, int, int, int, int]]]:
    if not hasattr(measurement, "pwmTempControllerState"):
        return np.array([], dtype=float), []
    pwm = np.atleast_1d(measurement.pwmTempControllerState)
    rows = []
    for p in pwm:
        try:
            t = float(getattr(p, "timeEpoch"))
            condition = get_pwm_condition(p)
            if np.isfinite(t) and condition is not None:
                rows.append((t, condition))
        except Exception:
            continue
    rows.sort(key=lambda x: x[0])
    if not rows:
        return np.array([], dtype=float), []
    return np.array([x[0] for x in rows], dtype=float), [x[1] for x in rows]


def detect_on_window(gate: np.ndarray) -> tuple[int, int] | None:
    gate = np.asarray(gate, dtype=float).squeeze()
    valid = np.isfinite(gate)
    if valid.sum() < 10:
        return None
    gate_clean = gate.copy()
    gate_clean[~valid] = np.nanmedian(gate_clean[valid])
    gate_filtered = median_filter(gate_clean, size=5, mode="nearest")
    low = np.percentile(gate_filtered, 5)
    high = np.percentile(gate_filtered, 95)
    amplitude = high - low
    if amplitude <= 0:
        return None
    on_threshold = low + 0.70 * amplitude
    off_threshold = low + 0.30 * amplitude
    state = np.zeros(len(gate_filtered), dtype=bool)
    is_on = False
    for i, value in enumerate(gate_filtered):
        if not is_on and value >= on_threshold:
            is_on = True
        elif is_on and value <= off_threshold:
            is_on = False
        state[i] = is_on
    changes = np.diff(np.concatenate(([False], state, [False])).astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    regions = [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]
    return max(regions, key=lambda x: x[1] - x[0]) if regions else None


def scan_run_hard_limit(mat_file: Path, engineering_limit_ohm: float) -> dict[str, Any]:
    data = scipy.io.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
    measurement = data["measurement"]
    pwm_times, pwm_conditions = build_pwm_timeline(measurement)
    transients = np.atleast_1d(measurement.transient)

    max_rds = -np.inf
    max_transient = np.nan
    max_time_us = np.nan
    exceedance_count = 0
    first_exceedance_time_us = np.nan
    first_exceedance_transient_id = np.nan
    valid_samples = 0
    target_transients = 0

    for transient_id, tr in enumerate(transients):
        try:
            transient_time = float(getattr(tr, "timeEpoch"))
        except Exception:
            continue
        if not np.isfinite(transient_time) or len(pwm_times) == 0:
            continue
        pwm_idx = np.searchsorted(pwm_times, transient_time, side="right") - 1
        if pwm_idx < 0 or pwm_conditions[pwm_idx] != TARGET_CONDITION:
            continue
        target_transients += 1

        td = tr.timeDomain
        gate = np.asarray(td.gateSignalVoltage, dtype=float).squeeze()
        vds = np.asarray(td.drainSourceVoltage, dtype=float).squeeze()
        ids = np.asarray(td.drainCurrent, dtype=float).squeeze()
        dt = float(np.asarray(td.dt).squeeze())
        n = min(len(gate), len(vds), len(ids))
        gate, vds, ids = gate[:n], vds[:n], ids[:n]
        detected = detect_on_window(gate)
        if detected is None:
            continue
        start, end = detected
        on_vds = vds[start:end]
        on_ids = ids[start:end]
        idx = np.arange(start, end)
        time_us = idx * dt * 1e6
        valid = np.isfinite(on_vds) & np.isfinite(on_ids) & (np.abs(on_ids) >= PROVISIONAL_MIN_CURRENT_A)
        if not np.any(valid):
            continue
        rds = on_vds[valid] / on_ids[valid]
        times = time_us[valid]
        valid_rds = np.isfinite(rds) & (rds > 0)
        rds, times = rds[valid_rds], times[valid_rds]
        if len(rds) == 0:
            continue
        valid_samples += len(rds)
        local_max_idx = int(np.argmax(rds))
        local_max = float(rds[local_max_idx])
        if local_max > max_rds:
            max_rds = local_max
            max_transient = int(transient_id)
            max_time_us = float(times[local_max_idx])
        exceed = rds > engineering_limit_ohm
        count = int(np.sum(exceed))
        exceedance_count += count
        if count > 0:
            first_idx = int(np.where(exceed)[0][0])
            t_first = float(times[first_idx])
            if np.isnan(first_exceedance_time_us) or t_first < first_exceedance_time_us:
                first_exceedance_time_us = t_first
                first_exceedance_transient_id = int(transient_id)

    if target_transients == 0:
        raise ValueError("No target-condition transients were found in the uploaded run.")

    return {
        "Engineering_Limit_Ohm": float(engineering_limit_ohm),
        "Max_RDS_Instantaneous_Ohm": float(max_rds) if np.isfinite(max_rds) else np.nan,
        "Max_RDS_Evidence_Transient_ID": max_transient,
        "Max_RDS_Evidence_Time_us": max_time_us,
        "Limit_Exceedance_Count": int(exceedance_count),
        "First_Limit_Exceedance_Time_us": first_exceedance_time_us,
        "First_Limit_Exceedance_Transient_ID": first_exceedance_transient_id,
        "Limit_Exceedance_Margin_Ohm": float(max_rds - engineering_limit_ohm) if np.isfinite(max_rds) else np.nan,
        "Limit_Exceedance_Flag": "FLAGGED" if exceedance_count > 0 else "NOT FLAGGED",
        "Valid_RDS_Sample_Count": int(valid_samples),
        "Target_Condition_Transient_Count": int(target_transients),
        "Provisional_Min_Current_A": PROVISIONAL_MIN_CURRENT_A,
    }


def _module_c_infer(root: Path, wide: pd.DataFrame, engineering_limits: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if "rdson" not in engineering_limits:
        # Current Python pipeline has no other supported measured parameter.
        out = pd.DataFrame({"Test_ID": wide["Test_ID"].astype(int)})
        out["Limit_Scan_Status"] = "NOT_EVALUATED"
        out["Limit_Scan_Reason"] = "No rdson engineering limit was supplied."
        out["Engineering_Limit_Ohm"] = np.nan
        out["Provisional_Min_Current_A"] = PROVISIONAL_MIN_CURRENT_A
        out["Limit_Exceedance_Flag"] = "NOT_EVALUATED"
        out["Max_RDS_Instantaneous_Ohm"] = np.nan
        out["Limit_Exceedance_Count"] = 0
        return out

    limit = engineering_limits["rdson"]
    engineering_limit_ohm = float(limit["limitValue"])

    records = []
    for _, row in wide[["Test_ID"]].drop_duplicates().iterrows():
        test_id = int(row["Test_ID"])
        mat_files = sorted(root.rglob(f"Test_{test_id}_run_*.mat"))
        if not mat_files:
            raise ValueError(f"No raw MAT files found for Test_ID {test_id}.")

        per_run = []
        for mat_file in mat_files:
            # Preserve existing V4 file naming and condition-selection behavior.
            # Run ID is taken from the filename for reporting only.
            result = scan_run_hard_limit(mat_file, engineering_limit_ohm)
            run_id = int(mat_file.stem.rsplit("_", 1)[-1])
            per_run.append({"Run_ID": run_id, **result})

        df = pd.DataFrame(per_run)
        max_idx = int(df["Max_RDS_Instantaneous_Ohm"].idxmax())
        max_row = df.loc[max_idx]
        records.append({
            "Test_ID": test_id,
            "Engineering_Limit_Ohm": engineering_limit_ohm,
            "Max_RDS_Instantaneous_Ohm": float(df["Max_RDS_Instantaneous_Ohm"].max()),
            "Limit_Exceedance_Count": int(df["Limit_Exceedance_Count"].sum()),
            "Limit_Exceedance_Runs": int((df["Limit_Exceedance_Flag"] == "FLAGGED").sum()),
            "Valid_RDS_Sample_Count": int(df["Valid_RDS_Sample_Count"].sum()),
            "Target_Condition_Transient_Count": int(df["Target_Condition_Transient_Count"].sum()),
            "Limit_Exceedance_Margin_Ohm": float(df["Max_RDS_Instantaneous_Ohm"].max() - engineering_limit_ohm),
            "Limit_Exceedance_Flag": "FLAGGED" if int(df["Limit_Exceedance_Count"].sum()) > 0 else "NOT FLAGGED",
            "Limit_Scan_Status": "EVALUATED",
            "Max_RDS_Evidence_Transient_ID": max_row["Max_RDS_Evidence_Transient_ID"],
            "Max_RDS_Evidence_Time_us": max_row["Max_RDS_Evidence_Time_us"],
            "Provisional_Min_Current_A": PROVISIONAL_MIN_CURRENT_A,
        })

    return pd.DataFrame(records)


def run_screening(dataset_bytes: bytes, lot_id: str, engineering_limits: dict[str, Any], file_name: str) -> dict[str, Any]:
    if not lot_id or not str(lot_id).strip():
        raise ValueError("lotId is required for live screening requests.")
    limits = validate_engineering_limits(engineering_limits)

    with tempfile.TemporaryDirectory(prefix="spad_") as tmp:
        root = Path(tmp)
        _safe_extract_zip(dataset_bytes, root)

        runlevel_file = _find_runlevel_csv(root)
        runlevel = pd.read_csv(runlevel_file)
        checkpoints_df = _load_checkpoint_csv(root)
        if checkpoints_df is None:
            checkpoints_df = _build_checkpoint_values(runlevel)

        wide = _build_wide(runlevel, checkpoints_df)
        _module_a_infer(wide)
        _module_b_infer(wide)
        component_limit = _module_c_infer(root, wide, limits)

        result = wide.merge(component_limit, on="Test_ID", how="left")
        result["Component_ID"] = result["Test_ID"].map(canonical_component_id)

        records = []
        for _, row in result.sort_values("Test_ID").iterrows():
            records.append({
                "componentId": row["Component_ID"],
                "lotId": str(lot_id),
                "measurement": {
                    "RDS0": float(row["RDS0"]),
                    "RDS33": float(row["RDS33"]),
                    "Delta_RDS_0_33": float(row["Delta_RDS_0_33"]),
                },
                "prediction": {
                    "Predicted_RDS100": float(row["Predicted_RDS100"]),
                    "Forecast_Residual": float(row["Forecast_Residual"]),
                    "Absolute_Forecast_Error": float(row["Absolute_Forecast_Error"]),
                    "Relative_Error_Percent": float(row["Relative_Error_Percent"]),
                },
                "lotAnomaly": {
                    "Module_A_IF_Score": float(row["Module_A_IF_Score_33"]),
                    "Module_A_Novelty_Percentile": float(row["Module_A_Novelty_Percentile_33"]),
                    "Module_A_Anomaly": str(row["Module_A_Flag_33"]),
                    "Module_A_IF_Scores": {
                        "0": float(row["Module_A_IF_Score_0"]),
                        "33": float(row["Module_A_IF_Score_33"]),
                        "66": float(row["Module_A_IF_Score_66"]),
                        "100": float(row["Module_A_IF_Score_100"]),
                    },
                    "Module_A_Novelty_Percentiles": {
                        "0": float(row["Module_A_Novelty_Percentile_0"]),
                        "33": float(row["Module_A_Novelty_Percentile_33"]),
                        "66": float(row["Module_A_Novelty_Percentile_66"]),
                        "100": float(row["Module_A_Novelty_Percentile_100"]),
                    },
                    "First_Anomaly_Stage": None if pd.isna(row["First_Anomaly_Stage"]) else int(row["First_Anomaly_Stage"]),
                },
                "engineering": {
                    "rdson": {
                        "limitValue": None if pd.isna(row.get("Engineering_Limit_Ohm", np.nan)) else float(row["Engineering_Limit_Ohm"]),
                        "unit": limits.get("rdson", {}).get("unit"),
                        "direction": limits.get("rdson", {}).get("direction", "UPPER"),
                        "maxRDSInstantaneousOhm": None if pd.isna(row.get("Max_RDS_Instantaneous_Ohm", np.nan)) else float(row["Max_RDS_Instantaneous_Ohm"]),
                        "limitExceedanceCount": int(row.get("Limit_Exceedance_Count", 0)),
                        "limitExceedanceFlag": str(row.get("Limit_Exceedance_Flag", "NOT_EVALUATED")),
                        "evidenceTransientId": None if pd.isna(row.get("Max_RDS_Evidence_Transient_ID", np.nan)) else int(row["Max_RDS_Evidence_Transient_ID"]),
                        "evidenceTimeUs": None if pd.isna(row.get("Max_RDS_Evidence_Time_us", np.nan)) else float(row["Max_RDS_Evidence_Time_us"]),
                    }
                },
            })

        return {
            "success": True,
            "lotId": str(lot_id),
            "fileName": file_name,
            "modelPackage": MODEL_PACKAGE_FILE.name,
            "records": records,
        }
