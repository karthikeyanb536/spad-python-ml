from __future__ import annotations

import io
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
        value = _as_finite_float(
            spec["limitValue"],
            f"engineeringLimits.{parameter}.limitValue",
        )
        direction = str(spec.get("direction", "UPPER")).upper()
        if direction not in {"UPPER", "LOWER"}:
            raise ValueError(f"Unsupported limit direction for '{parameter}': {direction}")
        resolved[parameter] = {
            "limitValue": value,
            "unit": spec.get("unit"),
            "direction": direction,
        }
    return resolved


def _open_bundle(dataset_bytes: bytes) -> tuple[io.BytesIO, zipfile.ZipFile, dict[str, str]]:
    """Open the uploaded ZIP without decompressing the large evidence CSV into RAM."""
    if not isinstance(dataset_bytes, (bytes, bytearray)) or not dataset_bytes:
        raise ValueError("Dataset bundle is empty.")

    buffer = io.BytesIO(dataset_bytes)
    try:
        zf = zipfile.ZipFile(buffer)
    except zipfile.BadZipFile as exc:
        buffer.close()
        raise ValueError("Uploaded file must be a valid ZIP bundle.") from exc

    names = {Path(n).name: n for n in zf.namelist() if not n.endswith("/")}
    if "component_data.csv" not in names:
        zf.close()
        buffer.close()
        raise ValueError("Bundle is missing component_data.csv.")
    if "transient_evidence.csv" not in names:
        zf.close()
        buffer.close()
        raise ValueError("Bundle is missing transient_evidence.csv.")

    return buffer, zf, names


def _read_component(zf: zipfile.ZipFile, component_name: str) -> pd.DataFrame:
    """Read the small component CSV fully; the large transient CSV is streamed later."""
    with zf.open(component_name) as f:
        component = pd.read_csv(f)
    return component

def _build_checkpoint_values(runlevel: pd.DataFrame) -> pd.DataFrame:
    checkpoints = [
        (1, 0.000000),
        (2, 0.333333),
        (3, 0.666667),
        (4, 1.000000),
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
        for checkpoint_id, progress in checkpoints:
            target_age = float(ages.min() + progress * span)
            nearest_idx = int(np.argmin(np.abs(ages - target_age)))
            rows.append({
                "Test_ID": int(group["Test_ID"].iloc[0]),
                "Checkpoint": checkpoint_id,
                "Checkpoint_Age_Equivalent_Hours": target_age,
                "RDSon_Interpolated_Ohm": float(np.interp(target_age, ages, rds)),
                "Nearest_Actual_Run_ID": int(group.loc[nearest_idx, "Run_ID"]),
            })
    if not rows:
        raise ValueError("Could not construct the 4-checkpoint trajectory.")
    return pd.DataFrame(rows)


def _build_wide(runlevel: pd.DataFrame, checkpoints_df: pd.DataFrame) -> pd.DataFrame:
    required_run = {
        "Test_ID", "Run_ID", "Cumulative_Aging_Hours",
        "RDSon_Median_Ohm", "ID_ON_Median_A",
    }
    missing = sorted(required_run - set(runlevel.columns))
    if missing:
        raise ValueError("Component CSV is missing required columns: " + ", ".join(missing))

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
    missing = [c for c in features if c not in wide.columns]
    if missing:
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


def _module_c_infer(
    zf: zipfile.ZipFile,
    evidence_name: str,
    component_ids: pd.Series,
    engineering_limits: dict[str, dict[str, Any]],
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """
    Evaluate the engineer-supplied RDS hard limit using exact sample-level CSV
    evidence, while streaming the large transient CSV in chunks.

    The input bundle can contain millions of transient samples. Only one
    chunk plus compact per-component accumulators is held in memory.
    """
    component_ids = component_ids.astype(int)

    if "rdson" not in engineering_limits:
        return pd.DataFrame({
            "Test_ID": component_ids,
            "Limit_Scan_Status": "NOT_EVALUATED",
            "Limit_Scan_Reason": "No rdson engineering limit was supplied.",
            "Engineering_Limit_Ohm": np.nan,
            "Max_RDS_Instantaneous_Ohm": np.nan,
            "Limit_Exceedance_Flag": "NOT_EVALUATED",
            "Limit_Exceedance_Count": 0,
            "Limit_Exceedance_Transient_Count": 0,
            "Provisional_Min_Current_A": PROVISIONAL_MIN_CURRENT_A,
        })

    limit = engineering_limits["rdson"]
    engineering_limit_ohm = float(limit["limitValue"])
    direction = str(limit.get("direction", "UPPER")).upper()
    if direction != "UPPER":
        raise ValueError("Compact CSV Module C currently supports direction=UPPER only.")

    required = {
        "Test_ID", "Run_ID", "Transient_ID", "Time_us", "RDS_Ohm",
        "Provisional_Min_Current_A", "Data_Format_Version",
    }
    available = set(pd.read_csv(zf.open(evidence_name), nrows=0).columns)
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            "transient_evidence.csv is missing columns: " + ", ".join(missing)
        )

    test_ids = [int(v) for v in component_ids.tolist()]
    test_id_set = set(test_ids)
    stats: dict[int, dict[str, Any]] = {
        test_id: {
            "valid_count": 0,
            "exceed_count": 0,
            "exceed_transients": set(),
            "target_transients": set(),
            "run_max": {},
            "max_rds": -np.inf,
            "max_transient_id": None,
            "max_time_us": np.nan,
            "first_exceed_time_us": np.nan,
            "first_exceed_transient_id": None,
            "seen_any": False,
            "format_versions": set(),
        }
        for test_id in test_ids
    }

    usecols = [
        "Test_ID", "Run_ID", "Transient_ID", "Time_us", "RDS_Ohm",
        "Data_Format_Version",
    ]

    with zf.open(evidence_name) as raw:
        reader = pd.read_csv(raw, usecols=usecols, chunksize=chunk_size)
        for chunk in reader:
            if chunk.empty:
                continue

            chunk["Test_ID"] = pd.to_numeric(chunk["Test_ID"], errors="raise").astype(int)
            chunk["Run_ID"] = pd.to_numeric(chunk["Run_ID"], errors="raise").astype(int)
            chunk["Transient_ID"] = pd.to_numeric(chunk["Transient_ID"], errors="raise").astype(int)
            chunk["Time_us"] = pd.to_numeric(chunk["Time_us"], errors="raise")
            chunk["RDS_Ohm"] = pd.to_numeric(chunk["RDS_Ohm"], errors="raise")

            versions = set(chunk["Data_Format_Version"].astype(str).str.strip())
            for test_id in test_id_set:
                sub = chunk[chunk["Test_ID"] == test_id]
                if sub.empty:
                    continue

                st = stats[test_id]
                st["seen_any"] = True
                st["format_versions"].update(versions)

                rds = sub["RDS_Ohm"].to_numpy(dtype=float)
                finite = np.isfinite(rds)
                if not finite.all():
                    raise ValueError(f"Non-finite RDS_Ohm evidence for Test_ID {test_id}.")
                times = sub["Time_us"].to_numpy(dtype=float)
                if not np.isfinite(times).all():
                    raise ValueError(f"Non-finite Time_us evidence for Test_ID {test_id}.")

                st["valid_count"] += int(len(sub))
                st["target_transients"].update(sub["Transient_ID"].astype(int).tolist())

                # Maximum RDS evidence: preserve first row in file order on ties.
                local_max_pos = int(np.argmax(rds))
                local_max = float(rds[local_max_pos])
                if local_max > st["max_rds"]:
                    row = sub.iloc[local_max_pos]
                    st["max_rds"] = local_max
                    st["max_transient_id"] = int(row["Transient_ID"])
                    st["max_time_us"] = float(row["Time_us"])

                exceed_mask = rds > engineering_limit_ohm
                if exceed_mask.any():
                    exceed_sub = sub.iloc[np.flatnonzero(exceed_mask)]
                    st["exceed_count"] += int(exceed_mask.sum())
                    st["exceed_transients"].update(
                        exceed_sub["Transient_ID"].astype(int).tolist()
                    )
                    # Current V2 semantics select the earliest Time_us, then
                    # retain the first row encountered on an exact tie.
                    local_first_pos = int(np.argmin(exceed_sub["Time_us"].to_numpy(dtype=float)))
                    local_first = exceed_sub.iloc[local_first_pos]
                    local_first_time = float(local_first["Time_us"])
                    if (
                        pd.isna(st["first_exceed_time_us"])
                        or local_first_time < float(st["first_exceed_time_us"])
                    ):
                        st["first_exceed_time_us"] = local_first_time
                        st["first_exceed_transient_id"] = int(local_first["Transient_ID"])

                # Build exact run-level max values without retaining raw samples.
                run_max_series = sub.groupby("Run_ID")["RDS_Ohm"].max()
                for run_id, run_max in run_max_series.items():
                    run_id = int(run_id)
                    run_max = float(run_max)
                    prior = st["run_max"].get(run_id, -np.inf)
                    if run_max > prior:
                        st["run_max"][run_id] = run_max

            if versions - {"SPAD_V4_WEB_CSV_V2"}:
                raise ValueError("Unsupported transient CSV Data_Format_Version.")

    records = []
    for test_id in test_ids:
        st = stats[test_id]
        if not st["seen_any"] or st["valid_count"] == 0:
            raise ValueError(f"No transient evidence for Test_ID {test_id}.")
        if st["format_versions"] != {"SPAD_V4_WEB_CSV_V2"}:
            raise ValueError(
                f"Unsupported transient CSV Data_Format_Version for Test_ID {test_id}."
            )

        limit_exceedance_runs = sum(
            1 for run_max in st["run_max"].values() if run_max > engineering_limit_ohm
        )

        records.append({
            "Test_ID": int(test_id),
            "Engineering_Limit_Ohm": engineering_limit_ohm,
            "Max_RDS_Instantaneous_Ohm": float(st["max_rds"]),
            "Limit_Exceedance_Count": int(st["exceed_count"]),
            "Limit_Exceedance_Transient_Count": int(len(st["exceed_transients"])),
            "Limit_Exceedance_Runs": int(limit_exceedance_runs),
            "Limit_Exceedance_Flag": (
                "FLAGGED" if st["exceed_count"] > 0 else "NOT FLAGGED"
            ),
            "Limit_Exceedance_Margin_Ohm": float(
                st["max_rds"] - engineering_limit_ohm
            ),
            "First_Limit_Exceedance_Time_us": (
                float(st["first_exceed_time_us"])
                if not pd.isna(st["first_exceed_time_us"]) else np.nan
            ),
            "First_Limit_Exceedance_Transient_ID": (
                int(st["first_exceed_transient_id"])
                if st["first_exceed_transient_id"] is not None else np.nan
            ),
            "Valid_RDS_Sample_Count": int(st["valid_count"]),
            "Target_Condition_Transient_Count": int(len(st["target_transients"])),
            "Limit_Scan_Status": "EVALUATED",
            "Limit_Scan_Reason": (
                "Exact sample-level CSV evidence streamed in chunks; no raw MAT files required."
            ),
            "Max_RDS_Evidence_Transient_ID": (
                int(st["max_transient_id"]) if st["max_transient_id"] is not None else np.nan
            ),
            "Max_RDS_Evidence_Time_us": float(st["max_time_us"]),
            "Provisional_Min_Current_A": PROVISIONAL_MIN_CURRENT_A,
        })

    return pd.DataFrame(records)

def run_screening(
    dataset_bytes: bytes,
    lot_id: str,
    engineering_limits: dict[str, Any],
    file_name: str,
) -> dict[str, Any]:
    if not lot_id or not str(lot_id).strip():
        raise ValueError("lotId is required for live screening requests.")

    limits = validate_engineering_limits(engineering_limits)
    buffer, zf, names = _open_bundle(dataset_bytes)
    try:
        component = _read_component(zf, names["component_data.csv"])

        required_component = {
            "Test_ID", "Run_ID", "Cumulative_Aging_Hours",
            "RDSon_Median_Ohm", "ID_ON_Median_A", "Data_Format_Version",
        }
        missing = sorted(required_component - set(component.columns))
        if missing:
            raise ValueError(
                "component_data.csv is missing required columns: " + ", ".join(missing)
            )

        if component.empty:
            raise ValueError("component_data.csv contains no rows.")
        if set(component["Data_Format_Version"].astype(str).str.strip()) != {
            "SPAD_V4_WEB_CSV_V2"
        }:
            raise ValueError("Unsupported component CSV Data_Format_Version.")

        component["Test_ID"] = pd.to_numeric(
            component["Test_ID"], errors="raise"
        ).astype(int)
        component["Run_ID"] = pd.to_numeric(
            component["Run_ID"], errors="raise"
        ).astype(int)
        duplicate = component.duplicated(["Test_ID", "Run_ID"], keep=False)
        if duplicate.any():
            raise ValueError("component_data.csv contains duplicate Test_ID/Run_ID rows.")

        checkpoints_df = _build_checkpoint_values(component)
        wide = _build_wide(component, checkpoints_df)
        _module_a_infer(wide)
        _module_b_infer(wide)
        component_limit = _module_c_infer(
            zf, names["transient_evidence.csv"], wide["Test_ID"], limits
        )
    finally:
        zf.close()
        buffer.close()

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
                    "limitExceedanceCount": int(row.get("Limit_Exceedance_Transient_Count", 0)),
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
        "dataFormatVersion": "SPAD_V4_WEB_CSV_V2",
        "records": records,
    }
