from __future__ import annotations

import hashlib
import io
import pickle
import re
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PACKAGE_FILE = BASE_DIR / "models" / "SPAD_V4_Final_Model_Package.pkl"
SERVICE_REVISION = "R1"
SERVICE_BUILD = "R1.1-HARDENED"
INPUT_FORMAT = "SPAD_V4_WEB_CSV_V2"
TARGET_CONDITION = (199, 200, 10, 5, 1000, 40)
TARGET_CONDITION_TEXT = ",".join(str(x) for x in TARGET_CONDITION)
CHECKPOINTS = ((1, 0.0), (2, 0.333333), (3, 0.666667), (4, 1.0))
MAX_ZIP_FILES = 2
MAX_TOTAL_UNCOMPRESSED_BYTES = 400 * 1024 * 1024
MAX_MEMBER_UNCOMPRESSED_BYTES = 350 * 1024 * 1024
MAX_COMPONENT_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_UPLOAD_BYTES = 250 * 1024 * 1024
ALLOWED_BUNDLE_NAMES = {"component_data.csv", "transient_evidence.csv"}
SUPPORTED_ENGINEERING_PARAMETERS = {"rdson"}
LOT_ID_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,128}$")

if not MODEL_PACKAGE_FILE.exists():
    raise RuntimeError(f"Model package not found: {MODEL_PACKAGE_FILE}")

MODEL_HASH_FILE = BASE_DIR / "MODEL_SHA256.txt"
MODEL_PACKAGE_BYTES = MODEL_PACKAGE_FILE.read_bytes()
MODEL_PACKAGE_SHA256 = hashlib.sha256(MODEL_PACKAGE_BYTES).hexdigest()


def _load_expected_model_sha256() -> str:
    if not MODEL_HASH_FILE.exists():
        raise RuntimeError(f"Model hash manifest not found: {MODEL_HASH_FILE}")
    parts = MODEL_HASH_FILE.read_text(encoding="utf-8").strip().split()
    expected_path = "models/SPAD_V4_Final_Model_Package.pkl"
    if len(parts) != 2 or parts[1].replace("\\", "/") != expected_path:
        raise RuntimeError("Malformed model hash manifest.")
    expected = parts[0].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError("Malformed model SHA-256 value in manifest.")
    return expected


EXPECTED_MODEL_PACKAGE_SHA256 = _load_expected_model_sha256()
if EXPECTED_MODEL_PACKAGE_SHA256 != MODEL_PACKAGE_SHA256:
    raise RuntimeError("Model package SHA-256 does not match MODEL_SHA256.txt.")

# Verify the package bytes before unpickling. The deployed pickle is trusted only
# after the immutable manifest hash matches.
MODEL_PACKAGE = pickle.loads(MODEL_PACKAGE_BYTES)
del MODEL_PACKAGE_BYTES

MODULE_A = MODEL_PACKAGE["module_a"]
MODULE_B = MODEL_PACKAGE["module_b"]
MODULE_C = MODEL_PACKAGE["module_c"]
PROVISIONAL_MIN_CURRENT_A = float(MODULE_C["provisional_min_current_A"])


def _validate_model_package_contract() -> None:
    if MODEL_PACKAGE.get("package_version") != "SPAD_V4":
        raise RuntimeError("Unsupported model package version.")
    if MODEL_PACKAGE.get("service_revision") != SERVICE_REVISION:
        raise RuntimeError("Model package/service revision mismatch.")

    metadata = MODEL_PACKAGE.get("metadata", {})
    if tuple(metadata.get("operating_condition", ())) != TARGET_CONDITION:
        raise RuntimeError("Frozen model operating condition mismatch.")

    reference_ids = [int(v) for v in MODULE_A.get("reference_test_ids", [])]
    held_out_ids = [int(v) for v in metadata.get("held_out_ids_for_current_audit", [])]
    declared_size = int(MODULE_A.get("reference_population_size", len(reference_ids)))
    if len(reference_ids) != len(set(reference_ids)):
        raise RuntimeError("Module A reference IDs contain duplicates.")
    if set(reference_ids) & set(held_out_ids):
        raise RuntimeError("Module A reference IDs overlap held-out IDs.")
    if len(reference_ids) != declared_size:
        raise RuntimeError("Module A reference population size mismatch.")

    expected_module_a_features = {
        0: ["RDS0"],
        33: ["RDS0", "RDS33", "Delta_RDS_0_33"],
        66: ["RDS0", "RDS33", "RDS66", "Delta_RDS_0_33", "Delta_RDS_33_66"],
        100: [
            "RDS0", "RDS33", "RDS66", "RDS100",
            "Delta_RDS_0_33", "Delta_RDS_33_66", "Delta_RDS_66_100",
        ],
    }
    packaged_feature_schema = MODULE_A.get("features_by_stage", {})
    if {int(k): list(v) for k, v in packaged_feature_schema.items()} != expected_module_a_features:
        raise RuntimeError("Module A feature schema mismatch.")

    for stage in (0, 33, 66, 100):
        model = MODULE_A["models"].get(stage, MODULE_A["models"].get(str(stage)))
        raw_scores = MODULE_A["reference_scores"].get(str(stage), MODULE_A["reference_scores"].get(stage))
        ref_scores = np.asarray(raw_scores, dtype=float)
        if model is None or len(ref_scores) != len(reference_ids) or not np.isfinite(ref_scores).all():
            raise RuntimeError(f"Module A package integrity check failed at stage {stage}%.")
        if int(getattr(model, "n_estimators", -1)) != 500:
            raise RuntimeError(f"Unexpected Module A tree count at stage {stage}%.")

    if list(MODULE_B.get("features", [])) != ["RDS0", "RDS33", "ID33_ID0_Ratio"]:
        raise RuntimeError("Module B feature schema mismatch.")
    if MODULE_B.get("target") != "RDS100":
        raise RuntimeError("Module B target mismatch.")
    rf = MODULE_B.get("model")
    if rf is None or int(getattr(rf, "n_features_in_", -1)) != 3 or len(getattr(rf, "estimators_", [])) != 100:
        raise RuntimeError("Module B model structure mismatch.")
    expected_module_b_params = {
        "n_estimators": 100,
        "max_depth": 2,
        "min_samples_leaf": 3,
        "min_samples_split": 2,
        "max_features": 1.0,
        "random_state": 42,
    }
    actual_module_b_params = MODULE_B.get("parameters", {})
    for key, expected in expected_module_b_params.items():
        actual = actual_module_b_params.get(key)
        if actual != expected:
            raise RuntimeError(f"Module B parameter mismatch for {key}.")
    if not np.isfinite(PROVISIONAL_MIN_CURRENT_A) or PROVISIONAL_MIN_CURRENT_A <= 0:
        raise RuntimeError("Module C provisional current configuration is invalid.")


_validate_model_package_contract()


def canonical_component_id(test_id: int | str) -> str:
    return f"TEST-{int(test_id):02d}"


def _coerce_integer_series(series: pd.Series, name: str) -> pd.Series:
    # Validate the textual representation first so very large IDs cannot be
    # rounded through an intermediate float64 representation.
    text = series.astype("string").str.strip()
    if text.isna().any() or not text.str.fullmatch(r"\+?\d+").all():
        raise ValueError(
            f"{name} must contain exact positive integer-valued IDs; "
            "fractional, scientific-notation, signed-negative, and non-integer IDs are invalid."
        )
    try:
        out = pd.to_numeric(text, errors="raise", downcast=None).astype("int64")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} contains IDs outside the supported int64 range.") from exc
    if (out <= 0).any():
        raise ValueError(f"{name} must contain positive IDs.")
    return pd.Series(out.to_numpy(dtype=np.int64), index=series.index)


def _as_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean.")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def validate_engineering_limits(engineering_limits: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if engineering_limits is None:
        return {}
    if not isinstance(engineering_limits, dict):
        raise ValueError("engineeringLimits must be an object.")
    unsupported = sorted(set(engineering_limits) - SUPPORTED_ENGINEERING_PARAMETERS)
    if unsupported:
        raise ValueError(
            "Unsupported engineering parameters for Python evaluation: "
            + ", ".join(unsupported)
            + ". Only 'rdson' is evaluated by this service."
        )

    resolved: dict[str, dict[str, Any]] = {}
    for parameter, spec in engineering_limits.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Engineering limit for '{parameter}' must be an object.")
        if "limitValue" not in spec:
            raise ValueError(f"Engineering limit for '{parameter}' is missing limitValue.")
        value = _as_finite_float(spec["limitValue"], f"engineeringLimits.{parameter}.limitValue")
        direction = str(spec.get("direction", "UPPER")).upper()
        if parameter == "rdson" and direction != "UPPER":
            raise ValueError("Compact CSV RDS(on) evaluation currently requires direction=UPPER.")
        if direction not in {"UPPER", "LOWER"}:
            raise ValueError(f"Unsupported limit direction for '{parameter}': {direction}")

        unit = spec.get("unit")
        if unit is not None and (not isinstance(unit, str) or len(unit) > 32):
            raise ValueError(f"Engineering limit unit for '{parameter}' must be a short string or null.")
        source = spec.get("source")
        if source is not None and (not isinstance(source, str) or len(source) > 64):
            raise ValueError(f"Engineering limit source for '{parameter}' must be a short string or null.")

        resolved[parameter] = {
            "limitValue": value,
            "unit": unit,
            "direction": direction,
            "source": source,
        }
    return resolved


def _validate_lot_id(lot_id: str) -> str:
    value = str(lot_id or "").strip()
    if not value or not LOT_ID_RE.fullmatch(value):
        raise ValueError("lotId must be 1-128 characters and contain no control characters.")
    return value


def _open_bundle(bundle: BinaryIO | bytes | bytearray) -> tuple[BinaryIO, zipfile.ZipFile]:
    if isinstance(bundle, (bytes, bytearray)):
        source: BinaryIO = io.BytesIO(bundle)
    else:
        source = bundle
        try:
            source.seek(0)
        except Exception as exc:
            raise ValueError("Uploaded bundle must be seekable.") from exc
    try:
        zf = zipfile.ZipFile(source)
    except zipfile.BadZipFile as exc:
        if isinstance(bundle, (bytes, bytearray)):
            source.close()
        raise ValueError("Uploaded file must be a valid ZIP bundle.") from exc

    infos = zf.infolist()
    if len(infos) != MAX_ZIP_FILES:
        zf.close()
        if isinstance(bundle, (bytes, bytearray)):
            source.close()
        raise ValueError("Bundle must contain exactly component_data.csv and transient_evidence.csv.")

    names = [info.filename for info in infos]
    if any(name not in ALLOWED_BUNDLE_NAMES for name in names) or set(names) != ALLOWED_BUNDLE_NAMES:
        zf.close()
        if isinstance(bundle, (bytes, bytearray)):
            source.close()
        raise ValueError("Bundle must contain exactly top-level component_data.csv and transient_evidence.csv.")

    total_uncompressed = 0
    for info in infos:
        if info.is_dir() or info.file_size < 0:
            zf.close()
            if isinstance(bundle, (bytes, bytearray)):
                source.close()
            raise ValueError("Bundle contains an invalid ZIP member.")
        if info.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
            zf.close()
            if isinstance(bundle, (bytes, bytearray)):
                source.close()
            raise ValueError(f"ZIP member '{info.filename}' exceeds the uncompressed size limit.")
        if info.filename == "component_data.csv" and info.file_size > MAX_COMPONENT_UNCOMPRESSED_BYTES:
            zf.close()
            if isinstance(bundle, (bytes, bytearray)):
                source.close()
            raise ValueError("component_data.csv exceeds the uncompressed component-data limit.")
        total_uncompressed += int(info.file_size)
    if total_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES:
        zf.close()
        if isinstance(bundle, (bytes, bytearray)):
            source.close()
        raise ValueError("ZIP bundle exceeds the total uncompressed size limit.")

    return source, zf


def _read_component(zf: zipfile.ZipFile) -> pd.DataFrame:
    with zf.open("component_data.csv") as f:
        return pd.read_csv(f)


def _validate_component(component: pd.DataFrame) -> tuple[pd.DataFrame, set[int], set[tuple[int, int]]]:
    required = {
        "Test_ID", "Run_ID", "Cumulative_Aging_Hours",
        "RDSon_Median_Ohm", "ID_ON_Median_A",
        "Data_Format_Version", "Target_Condition", "Provisional_Min_Current_A",
    }
    missing = sorted(required - set(component.columns))
    if missing:
        raise ValueError("component_data.csv is missing required columns: " + ", ".join(missing))
    if component.empty:
        raise ValueError("component_data.csv contains no rows.")

    versions = set(component["Data_Format_Version"].astype(str).str.strip())
    if versions != {INPUT_FORMAT}:
        raise ValueError("Unsupported component CSV Data_Format_Version.")
    conditions = set(component["Target_Condition"].astype(str).str.strip())
    if conditions != {TARGET_CONDITION_TEXT}:
        raise ValueError("component_data.csv Target_Condition does not match the frozen model condition.")

    min_current = pd.to_numeric(component["Provisional_Min_Current_A"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(min_current).all() or not np.allclose(min_current, PROVISIONAL_MIN_CURRENT_A, rtol=0, atol=1e-12):
        raise ValueError("component_data.csv Provisional_Min_Current_A does not match the frozen model configuration.")

    component = component.copy()
    component["Test_ID"] = _coerce_integer_series(component["Test_ID"], "Test_ID")
    component["Run_ID"] = _coerce_integer_series(component["Run_ID"], "Run_ID")

    duplicate = component.duplicated(["Test_ID", "Run_ID"], keep=False)
    if duplicate.any():
        raise ValueError("component_data.csv contains duplicate Test_ID/Run_ID rows.")

    numeric_cols = ["Cumulative_Aging_Hours", "RDSon_Median_Ohm", "ID_ON_Median_A"]
    numeric = component[numeric_cols].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("component_data.csv contains non-finite numeric values.")
    if (numeric["Cumulative_Aging_Hours"] < 0).any():
        raise ValueError("Cumulative_Aging_Hours must be non-negative.")
    if (numeric["RDSon_Median_Ohm"] <= 0).any():
        raise ValueError("RDSon_Median_Ohm must be positive.")
    if (numeric["ID_ON_Median_A"] <= 0).any():
        raise ValueError("ID_ON_Median_A must be positive for the frozen ratio feature.")

    for test_id, group in component.groupby("Test_ID"):
        ages = np.sort(group["Cumulative_Aging_Hours"].to_numpy(dtype=float))
        if len(group) < 2:
            raise ValueError(f"Test_ID {test_id} requires at least 2 component runs.")
        if np.any(np.diff(ages) <= 0):
            raise ValueError(f"Test_ID {test_id} has duplicate/non-increasing Cumulative_Aging_Hours.")

    # Return the sanitized frame so downstream grouping cannot accidentally
    # operate on the original string/float representations.
    component["Cumulative_Aging_Hours"] = numeric["Cumulative_Aging_Hours"].astype(float)
    component["RDSon_Median_Ohm"] = numeric["RDSon_Median_Ohm"].astype(float)
    component["ID_ON_Median_A"] = numeric["ID_ON_Median_A"].astype(float)

    test_ids = set(component["Test_ID"].astype(int).tolist())
    valid_pairs = set(zip(component["Test_ID"].astype(int), component["Run_ID"].astype(int)))
    return component, test_ids, valid_pairs


def _build_checkpoint_values(component: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for test_id, group in component.groupby("Test_ID"):
        group = group.sort_values("Cumulative_Aging_Hours").reset_index(drop=True)
        ages = group["Cumulative_Aging_Hours"].to_numpy(dtype=float)
        rds = group["RDSon_Median_Ohm"].to_numpy(dtype=float)
        if len(group) < 2:
            raise ValueError(f"Test_ID {test_id} requires at least 2 component runs.")
        span = float(ages[-1] - ages[0])
        if not np.isfinite(span) or span <= 0:
            raise ValueError(f"Test_ID {test_id} has no positive aging span.")
        for checkpoint_id, progress in CHECKPOINTS:
            target_age = float(ages[0] + progress * span)
            nearest_idx = int(np.argmin(np.abs(ages - target_age)))
            rows.append({
                "Test_ID": int(test_id),
                "Checkpoint": int(checkpoint_id),
                "Normalized_Progress": float(progress),
                "Checkpoint_Age_Equivalent_Hours": target_age,
                "RDSon_Interpolated_Ohm": float(np.interp(target_age, ages, rds)),
                "Nearest_Actual_Run_ID": int(group.loc[nearest_idx, "Run_ID"]),
            })
    return pd.DataFrame(rows)


def _build_wide(component: pd.DataFrame, checkpoints_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for test_id, cp_group in checkpoints_df.groupby("Test_ID"):
        runs = component[component["Test_ID"] == test_id].sort_values("Cumulative_Aging_Hours")
        age = runs["Cumulative_Aging_Hours"].to_numpy(dtype=float)
        ids = runs["ID_ON_Median_A"].to_numpy(dtype=float)
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
    wide["ID33_ID0_Ratio"] = wide["ID33"] / wide["ID0"]
    required = [
        "Test_ID", "RDS0", "RDS33", "RDS66", "RDS100", "ID0", "ID33",
        "Delta_RDS_0_33", "Delta_RDS_33_66", "Delta_RDS_66_100", "ID33_ID0_Ratio",
    ]
    if not np.isfinite(wide[required[1:]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite trajectory feature generated for Module A/B.")
    return wide


def _module_a_infer(wide: pd.DataFrame) -> None:
    for stage in (0, 33, 66, 100):
        features = MODULE_A["features_by_stage"][stage]
        model = MODULE_A["models"][stage]
        X = wide[features].to_numpy(dtype=float)
        scores = model.decision_function(X)
        flags = np.where(model.predict(X) == -1, "FLAGGED", "NOT FLAGGED")
        reference_scores = np.asarray(MODULE_A["reference_scores"][str(stage)], dtype=float)
        novelty = np.array([100.0 * np.mean(reference_scores >= score) for score in scores])
        wide[f"Module_A_IF_Score_{stage}"] = scores
        wide[f"Module_A_Novelty_Percentile_{stage}"] = novelty
        wide[f"Module_A_Flag_{stage}"] = flags

    flags = wide[[f"Module_A_Flag_{s}" for s in (0, 33, 66, 100)]].eq("FLAGGED")
    wide["Progressive_ML_Flag"] = np.where(flags.any(axis=1), "FLAGGED", "NOT FLAGGED")
    wide["First_Anomaly_Stage"] = pd.Series(pd.NA, index=wide.index, dtype="Int64")
    for stage in (0, 33, 66, 100):
        mask = wide["First_Anomaly_Stage"].isna() & wide[f"Module_A_Flag_{stage}"].eq("FLAGGED")
        wide.loc[mask, "First_Anomaly_Stage"] = stage


def _module_b_infer(wide: pd.DataFrame) -> None:
    features = MODULE_B["features"]
    X = wide[features].to_numpy(dtype=float)
    wide["Predicted_RDS100"] = MODULE_B["model"].predict(X)
    wide["Forecast_Residual"] = wide["RDS100"] - wide["Predicted_RDS100"]
    wide["Absolute_Forecast_Error"] = wide["Forecast_Residual"].abs()
    wide["Relative_Error_Percent"] = np.where(
        np.abs(wide["RDS100"]) > 0,
        wide["Absolute_Forecast_Error"] / np.abs(wide["RDS100"]) * 100.0,
        np.nan,
    )


def _module_c_infer(
    zf: zipfile.ZipFile,
    component_ids: set[int],
    valid_pairs: set[tuple[int, int]],
    engineering_limits: dict[str, dict[str, Any]],
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    has_rdson_limit = "rdson" in engineering_limits
    engineering_limit = None
    if has_rdson_limit:
        engineering_limit = float(engineering_limits["rdson"]["limitValue"])
        if not np.isfinite(engineering_limit) or engineering_limit <= 0:
            raise ValueError("RDS(on) engineering limit must be a finite positive value.")

    with zf.open("transient_evidence.csv") as header_raw:
        header = pd.read_csv(header_raw, nrows=0)
    required = {
        "Test_ID", "Run_ID", "Transient_ID", "Time_us", "RDS_Ohm",
        "Target_Condition", "Provisional_Min_Current_A", "Data_Format_Version",
    }
    missing = sorted(required - set(header.columns))
    if missing:
        raise ValueError("transient_evidence.csv is missing columns: " + ", ".join(missing))

    seen_pairs: set[tuple[int, int]] = set()
    stats: dict[int, dict[str, Any]] = {
        tid: {
            "valid_count": 0,
            "exceed_count": 0,
            "exceed_transients": set(),
            "target_transients": set(),
            "run_max": {},
            "max_rds": -np.inf,
            "max_run_id": None,
            "max_transient_id": None,
            "max_time_us": np.nan,
            "earliest_local_exceed_time": np.nan,
            "earliest_local_run_id": None,
            "earliest_local_transient_id": None,
            "seen_any": False,
        } for tid in component_ids
    }

    usecols = [
        "Test_ID", "Run_ID", "Transient_ID", "Time_us", "RDS_Ohm",
        "Target_Condition", "Provisional_Min_Current_A", "Data_Format_Version",
    ]

    with zf.open("transient_evidence.csv") as raw:
        reader = pd.read_csv(raw, usecols=usecols, chunksize=chunk_size)
        for chunk in reader:
            if chunk.empty:
                continue

            chunk["Test_ID"] = _coerce_integer_series(chunk["Test_ID"], "transient Test_ID")
            chunk["Run_ID"] = _coerce_integer_series(chunk["Run_ID"], "transient Run_ID")
            chunk["Transient_ID"] = _coerce_integer_series(chunk["Transient_ID"], "Transient_ID")
            chunk["Time_us"] = pd.to_numeric(chunk["Time_us"], errors="raise")
            chunk["RDS_Ohm"] = pd.to_numeric(chunk["RDS_Ohm"], errors="raise")
            chunk["Provisional_Min_Current_A"] = pd.to_numeric(
                chunk["Provisional_Min_Current_A"], errors="raise"
            )

            if not np.isfinite(
                chunk[["Time_us", "RDS_Ohm", "Provisional_Min_Current_A"]]
                .to_numpy(dtype=float)
            ).all():
                raise ValueError("transient_evidence.csv contains non-finite numeric values.")
            if (chunk["Time_us"] < 0).any():
                raise ValueError("Time_us must be non-negative.")
            if (chunk["RDS_Ohm"] <= 0).any():
                raise ValueError("RDS_Ohm must be positive.")

            if set(chunk["Data_Format_Version"].astype(str).str.strip()) != {INPUT_FORMAT}:
                raise ValueError("Unsupported transient CSV Data_Format_Version.")
            if set(chunk["Target_Condition"].astype(str).str.strip()) != {TARGET_CONDITION_TEXT}:
                raise ValueError("transient_evidence.csv Target_Condition mismatch.")
            if not np.allclose(
                chunk["Provisional_Min_Current_A"].to_numpy(float),
                PROVISIONAL_MIN_CURRENT_A,
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError("transient_evidence.csv Provisional_Min_Current_A mismatch.")

            unknown_tests = set(chunk["Test_ID"].astype(int)) - component_ids
            if unknown_tests:
                raise ValueError(
                    "transient_evidence.csv contains Test_ID values absent from component_data.csv: "
                    + str(sorted(unknown_tests))
                )

            pairs = set(zip(chunk["Test_ID"].astype(int), chunk["Run_ID"].astype(int)))
            seen_pairs.update(pairs)
            bad_pairs = sorted(pair for pair in pairs if pair not in valid_pairs)
            if bad_pairs:
                raise ValueError(
                    "transient_evidence.csv contains Test_ID/Run_ID pairs absent from component_data.csv: "
                    + str(bad_pairs[:10])
                )

            for test_id in sorted(component_ids):
                sub = chunk[chunk["Test_ID"] == test_id]
                if sub.empty:
                    continue

                st = stats[test_id]
                st["seen_any"] = True
                st["valid_count"] += int(len(sub))

                transient_pairs = set(
                    zip(sub["Run_ID"].astype(int), sub["Transient_ID"].astype(int))
                )
                st["target_transients"].update(transient_pairs)

                # Module C compliance metrics are only computed when an RDS limit exists.
                if has_rdson_limit:
                    rds = sub["RDS_Ohm"].to_numpy(dtype=float)
                    local_max_pos = int(np.argmax(rds))
                    local_max = float(rds[local_max_pos])

                    if local_max > st["max_rds"]:
                        row = sub.iloc[local_max_pos]
                        st["max_rds"] = local_max
                        st["max_run_id"] = int(row["Run_ID"])
                        st["max_transient_id"] = int(row["Transient_ID"])
                        st["max_time_us"] = float(row["Time_us"])

                    exceed_mask = rds > engineering_limit
                    if exceed_mask.any():
                        exceed_sub = sub.iloc[np.flatnonzero(exceed_mask)]
                        st["exceed_count"] += int(exceed_mask.sum())
                        st["exceed_transients"].update(
                            zip(
                                exceed_sub["Run_ID"].astype(int),
                                exceed_sub["Transient_ID"].astype(int),
                            )
                        )
                        # Time_us restarts for every transient. This is the earliest
                        # local time among violating evidence, not a lifetime chronology.
                        local_first_pos = int(
                            np.argmin(exceed_sub["Time_us"].to_numpy(dtype=float))
                        )
                        first = exceed_sub.iloc[local_first_pos]
                        t = float(first["Time_us"])
                        if (
                            pd.isna(st["earliest_local_exceed_time"])
                            or t < float(st["earliest_local_exceed_time"])
                        ):
                            st["earliest_local_exceed_time"] = t
                            st["earliest_local_run_id"] = int(first["Run_ID"])
                            st["earliest_local_transient_id"] = int(first["Transient_ID"])

                    for run_id, run_max in sub.groupby("Run_ID")["RDS_Ohm"].max().items():
                        run_id = int(run_id)
                        st["run_max"][run_id] = max(
                            float(run_max), st["run_max"].get(run_id, -np.inf)
                        )

    missing_pairs = sorted(valid_pairs - seen_pairs)
    if missing_pairs:
        raise ValueError(
            "transient_evidence.csv is missing evidence for component Test_ID/Run_ID pairs: "
            + str(missing_pairs[:20])
        )

    records = []
    for test_id in sorted(component_ids):
        st = stats[test_id]
        if not st["seen_any"] or st["valid_count"] == 0:
            raise ValueError(f"No transient evidence for Test_ID {test_id}.")

        if has_rdson_limit:
            exceed_runs = sum(
                1 for value in st["run_max"].values() if value > engineering_limit
            )
            records.append({
                "Test_ID": int(test_id),
                "Engineering_Limit_Ohm": engineering_limit,
                "Max_RDS_Instantaneous_Ohm": float(st["max_rds"]),
                "Limit_Exceedance_Count": int(st["exceed_count"]),
                "Limit_Exceedance_Transient_Count": int(len(st["exceed_transients"])),
                "Limit_Exceedance_Runs": int(exceed_runs),
                "Limit_Exceedance_Flag": (
                    "FLAGGED" if st["exceed_count"] > 0 else "NOT FLAGGED"
                ),
                "Limit_Exceedance_Margin_Ohm": float(
                    st["max_rds"] - engineering_limit
                ),
                "Earliest_Local_Exceedance_Time_us": (
                    float(st["earliest_local_exceed_time"])
                    if not pd.isna(st["earliest_local_exceed_time"])
                    else np.nan
                ),
                "Earliest_Local_Exceedance_Run_ID": st["earliest_local_run_id"],
                "Earliest_Local_Exceedance_Transient_ID": st["earliest_local_transient_id"],
                "Valid_RDS_Sample_Count": int(st["valid_count"]),
                "Target_Condition_Transient_Count": int(len(st["target_transients"])),
                "Limit_Scan_Status": "EVALUATED",
                "Limit_Scan_Reason": "Exact sample-level CSV evidence streamed in chunks.",
                "Max_RDS_Evidence_Run_ID": st["max_run_id"],
                "Max_RDS_Evidence_Transient_ID": st["max_transient_id"],
                "Max_RDS_Evidence_Time_us": float(st["max_time_us"]),
                "Provisional_Min_Current_A": PROVISIONAL_MIN_CURRENT_A,
            })
        else:
            records.append({
                "Test_ID": int(test_id),
                "Engineering_Limit_Ohm": np.nan,
                "Max_RDS_Instantaneous_Ohm": np.nan,
                "Limit_Exceedance_Count": 0,
                "Limit_Exceedance_Transient_Count": 0,
                "Limit_Exceedance_Runs": 0,
                "Limit_Exceedance_Flag": "NOT_EVALUATED",
                "Limit_Exceedance_Margin_Ohm": np.nan,
                "Earliest_Local_Exceedance_Time_us": np.nan,
                "Earliest_Local_Exceedance_Run_ID": None,
                "Earliest_Local_Exceedance_Transient_ID": None,
                "Valid_RDS_Sample_Count": int(st["valid_count"]),
                "Target_Condition_Transient_Count": int(len(st["target_transients"])),
                "Limit_Scan_Status": "NOT_EVALUATED",
                "Limit_Scan_Reason": (
                    "No rdson engineering limit supplied; transient evidence contract was validated."
                ),
                "Max_RDS_Evidence_Run_ID": None,
                "Max_RDS_Evidence_Transient_ID": None,
                "Max_RDS_Evidence_Time_us": np.nan,
                "Provisional_Min_Current_A": PROVISIONAL_MIN_CURRENT_A,
            })

    return pd.DataFrame(records)
def run_screening(
    dataset: BinaryIO | bytes | bytearray,
    lot_id: str,
    engineering_limits: dict[str, Any] | None,
    file_name: str,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> dict[str, Any]:
    lot_id = _validate_lot_id(lot_id)
    limits = validate_engineering_limits(engineering_limits)

    source, zf = _open_bundle(dataset)
    try:
        # For file objects, check compressed upload size without copying to RAM.
        try:
            source.seek(0, 2)
            compressed_size = int(source.tell())
            source.seek(0)
        except Exception:
            compressed_size = None
        if compressed_size is not None and compressed_size > max_upload_bytes:
            raise ValueError(f"Dataset exceeds the {max_upload_bytes // (1024 * 1024)} MB upload limit.")

        component = _read_component(zf)
        component, component_ids, valid_pairs = _validate_component(component)
        checkpoints_df = _build_checkpoint_values(component)
        wide = _build_wide(component, checkpoints_df)
        _module_a_infer(wide)
        _module_b_infer(wide)
        component_limit = _module_c_infer(zf, component_ids, valid_pairs, limits)
    finally:
        zf.close()
        if isinstance(dataset, (bytes, bytearray)):
            source.close()

    result = wide.merge(component_limit, on="Test_ID", how="left")
    result["Component_ID"] = result["Test_ID"].map(canonical_component_id)

    records = []
    for _, row in result.sort_values("Test_ID").iterrows():
        engineering = {
            "limitValue": None if pd.isna(row.get("Engineering_Limit_Ohm", np.nan)) else float(row["Engineering_Limit_Ohm"]),
            "unit": limits.get("rdson", {}).get("unit"),
            "direction": limits.get("rdson", {}).get("direction", "UPPER"),
            "source": limits.get("rdson", {}).get("source"),
            "maxRDSInstantaneousOhm": None if pd.isna(row.get("Max_RDS_Instantaneous_Ohm", np.nan)) else float(row["Max_RDS_Instantaneous_Ohm"]),
            # Backward-compatible field name now has correct SAMPLE-count semantics.
            "limitExceedanceCount": int(row.get("Limit_Exceedance_Count", 0)),
            "limitExceedanceSampleCount": int(row.get("Limit_Exceedance_Count", 0)),
            "limitExceedanceTransientCount": int(row.get("Limit_Exceedance_Transient_Count", 0)),
            "limitExceedanceTransientGroupCount": int(row.get("Limit_Exceedance_Transient_Count", 0)),
            "limitExceedanceRunCount": int(row.get("Limit_Exceedance_Runs", 0)),
            "limitExceedanceFlag": str(row.get("Limit_Exceedance_Flag", "NOT_EVALUATED")),
            "maxEvidenceRunId": None if pd.isna(row.get("Max_RDS_Evidence_Run_ID", np.nan)) else int(row["Max_RDS_Evidence_Run_ID"]),
            "evidenceTransientId": None if pd.isna(row.get("Max_RDS_Evidence_Transient_ID", np.nan)) else int(row["Max_RDS_Evidence_Transient_ID"]),
            "evidenceTimeUs": None if pd.isna(row.get("Max_RDS_Evidence_Time_us", np.nan)) else float(row["Max_RDS_Evidence_Time_us"]),
            "validRdsSampleCount": int(row.get("Valid_RDS_Sample_Count", 0)),
            "targetConditionTransientCount": int(row.get("Target_Condition_Transient_Count", 0)),
        }
        records.append({
            "componentId": row["Component_ID"],
            "lotId": lot_id,
            "evaluation": {
                "type": "COMPLETED_TRAJECTORY_ENDPOINT_ESTIMATION",
                "normalizedCheckpointProgressPercent": [0.0, 33.3333, 66.6667, 100.0],
                "zeroPercentDefinition": "earliest observed trajectory point",
                "hundredPercentDefinition": "latest observed trajectory point in the submitted bundle",
                "temporalCausalityClaim": False,
            },
            "measurement": {
                "RDS0": float(row["RDS0"]),
                "RDS33": float(row["RDS33"]),
                "Delta_RDS_0_33": float(row["Delta_RDS_0_33"]),
            },
            "prediction": {
                "Predicted_RDS100": float(row["Predicted_RDS100"]),
                "predictedNormalizedEndpoint": float(row["Predicted_RDS100"]),
                "Forecast_Residual": float(row["Forecast_Residual"]),
                "Absolute_Forecast_Error": float(row["Absolute_Forecast_Error"]),
                "Relative_Error_Percent": float(row["Relative_Error_Percent"]),
                "retrospectiveEndpointResidual": float(row["Forecast_Residual"]),
                "retrospectiveEndpointAbsoluteError": float(row["Absolute_Forecast_Error"]),
                "retrospectiveEndpointRelativeErrorPercent": float(row["Relative_Error_Percent"]),
                "target": "RDS at the 100% normalized trajectory endpoint",
                "semantics": "Normalized endpoint estimate; error fields are retrospective when the full trajectory endpoint is present.",
            },
            "lotAnomaly": {
                "scope": "global_reference_population",
                "Module_A_IF_Score": float(row["Module_A_IF_Score_33"]),
                "Module_A_Novelty_Percentile": float(row["Module_A_Novelty_Percentile_33"]),
                "Module_A_Anomaly": str(row["Module_A_Flag_33"]),
                "Module_A_Authority": "EVIDENCE_ONLY",
                "Module_A_Interpretation": "Outlier evidence relative to the frozen reference population; not a probability, failure prediction, or engineering disposition.",
                "Module_A_ProbabilityClaim": False,
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
            "engineering": {"rdson": engineering},
        })

    return {
        "success": True,
        "serviceRevision": SERVICE_REVISION,
        "serviceBuild": SERVICE_BUILD,
        "lotId": lot_id,
        "fileName": file_name,
        "modelPackage": MODEL_PACKAGE_FILE.name,
        "modelPackageSha256": MODEL_PACKAGE_SHA256,
        "dataFormatVersion": INPUT_FORMAT,
        "modelMetadata": {
            "packageVersion": MODEL_PACKAGE.get("package_version"),
            "serviceRevision": SERVICE_REVISION,
            "serviceBuild": SERVICE_BUILD,
            "moduleA": {
                "decisionAuthority": "evidence_only",
                "referencePopulationSize": int(len(MODULE_A["reference_test_ids"])),
                "referenceTestIds": [int(v) for v in MODULE_A["reference_test_ids"]],
                "referencePopulationDomain": "Frozen NASA-derived reference population; cross-lot domain shift is not calibrated.",
                "noveltyDefinition": MODULE_A.get("novelty_definition"),
                "flagSemantics": MODULE_A.get("flag_semantics"),
            },
            "moduleB": {
                "features": list(MODULE_B["features"]),
                "developmentDeviceCount": int(len(MODULE_A["reference_test_ids"])),
                "heldOutDeviceIds": [int(v) for v in MODEL_PACKAGE.get("metadata", {}).get("held_out_ids_for_current_audit", [])],
                "target": "RDS100",
                "targetSemantics": MODULE_B.get("target_semantics"),
                "documentedNestedLOOMaeOhm": MODEL_PACKAGE.get("metadata", {}).get("module_b_documented_nested_loo_mae_ohm"),
                "documentedNestedLOORmseOhm": MODEL_PACKAGE.get("metadata", {}).get("module_b_documented_nested_loo_rmse_ohm"),
                "frozenConfigLOOMaeOhmOnWebBundle": MODEL_PACKAGE.get("metadata", {}).get("module_b_current_final_config_loo_mae_ohm_on_web_bundle"),
                "frozenConfigLOORmseOhmOnWebBundle": MODEL_PACKAGE.get("metadata", {}).get("module_b_current_final_config_loo_rmse_ohm_on_web_bundle"),
                "predictionSemantics": MODULE_B.get("prediction_semantics"),
                "errorSemantics": MODULE_B.get("error_semantics"),
            },
            "checkpoints": {
                "progressPercent": [0.0, 33.3333, 66.6667, 100.0],
                "physicalTimeLabels": False,
                "normalization": "per-device observed trajectory span",
                "zeroPercentDefinition": "earliest observed trajectory point",
                "hundredPercentDefinition": "latest observed trajectory point in the submitted bundle",
            },
            "moduleC": {
                "transientGroupKey": ["Test_ID", "Run_ID", "Transient_ID"],
                "limitExceedanceCountSemantics": "sample_count",
                "limitExceedanceTransientCountSemantics": "unique_run_transient_groups",
            },
        },
        "records": records,
    }
