from __future__ import annotations

import csv
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io
from scipy.ndimage import median_filter

# ============================================================
# USER CONFIGURATION
# ============================================================

DATA_ROOT = Path(r"D:\MOSFET_Thermal_Overstress_Aging_v0")
RUNLEVEL_FILE = Path(
    r"C:\Users\Admin\Downloads\MOSFET_199_200C_V2_Electrical_Features_RunLevel.csv"
)
MODEL_PACKAGE_FILE = Path(
    r"C:\Users\Admin\Downloads\SPAD_V4_EXTERNAL_SERVICE_FINAL\models\SPAD_V4_Final_Model_Package.pkl"
)
OUTPUT_DIR = Path(r"C:\Users\Admin\Downloads\SPAD_V4_WEB_INPUT_V2")
COMPONENT_CSV = OUTPUT_DIR / "component_data.csv"
TRANSIENT_CSV = OUTPUT_DIR / "transient_evidence.csv"
BUNDLE_ZIP = Path(r"C:\Users\Admin\Downloads\SPAD_V4_WEB_INPUT_V2.zip")

# ============================================================
# LOAD V4 METADATA
# ============================================================
with MODEL_PACKAGE_FILE.open("rb") as f:
    package = pickle.load(f)

if package.get("package_version") != "SPAD_V4":
    raise RuntimeError("Expected SPAD_V4 model package.")

TARGET_CONDITION = tuple(package["metadata"]["operating_condition"])
PROVISIONAL_MIN_CURRENT_A = float(
    package["module_c"]["provisional_min_current_A"]
)

# ============================================================
# SAME CONDITION / ON-WINDOW LOGIC AS V4 PIPELINE
# ============================================================
def get_pwm_condition(p: Any):
    fields = (
        "lowTemp", "highTemp", "gateVoltage",
        "supplyVoltage", "switchingFrequency", "dutyCycle",
    )
    values = []
    for field in fields:
        value = getattr(p, field, np.nan)
        try:
            value = float(value)
        except Exception:
            value = np.nan
        if not np.isfinite(value):
            return None
        values.append(int(round(value)))
    return tuple(values)


def build_pwm_timeline(measurement: Any):
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


def detect_on_window(gate: np.ndarray):
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

# ============================================================
# BUILD ONE TRANSIENT-EVIDENCE ROW PER TARGET CONDITION TRANSIENT
# Store one row per valid instantaneous RDS sample. This preserves
# the exact Module C sample-level semantics while keeping the data
# much cleaner than embedding a giant JSON array inside a CSV cell.
# ============================================================
def extract_transient_rows(mat_file: Path, test_id: int, run_id: int):
    data = scipy.io.loadmat(
        mat_file, struct_as_record=False, squeeze_me=True
    )
    if "measurement" not in data:
        raise ValueError(f"{mat_file.name}: missing measurement object.")

    measurement = data["measurement"]
    pwm_times, pwm_conditions = build_pwm_timeline(measurement)
    transients = np.atleast_1d(getattr(measurement, "transient", []))
    rows = []

    for transient_id, tr in enumerate(transients):
        try:
            transient_time = float(getattr(tr, "timeEpoch"))
        except Exception:
            continue
        if not np.isfinite(transient_time) or len(pwm_times) == 0:
            continue

        pwm_idx = np.searchsorted(
            pwm_times, transient_time, side="right"
        ) - 1
        if pwm_idx < 0 or pwm_conditions[pwm_idx] != TARGET_CONDITION:
            continue

        try:
            td = tr.timeDomain
            gate = np.asarray(td.gateSignalVoltage, dtype=float).squeeze()
            vds = np.asarray(td.drainSourceVoltage, dtype=float).squeeze()
            ids = np.asarray(td.drainCurrent, dtype=float).squeeze()
            dt = float(np.asarray(td.dt).squeeze())
        except Exception:
            continue

        n = min(len(gate), len(vds), len(ids))
        if n < 20 or not np.isfinite(dt) or dt <= 0:
            continue
        gate, vds, ids = gate[:n], vds[:n], ids[:n]

        detected = detect_on_window(gate)
        if detected is None:
            continue
        start, end = detected
        if end <= start:
            continue

        on_vds = vds[start:end]
        on_ids = ids[start:end]
        idx = np.arange(start, end)
        time_us = idx * dt * 1e6

        valid = (
            np.isfinite(on_vds)
            & np.isfinite(on_ids)
            & (np.abs(on_ids) >= PROVISIONAL_MIN_CURRENT_A)
        )
        if not np.any(valid):
            continue

        rds = on_vds[valid] / on_ids[valid]
        times = time_us[valid]
        valid_rds = np.isfinite(rds) & (rds > 0)
        rds = rds[valid_rds]
        times = times[valid_rds]
        if len(rds) == 0:
            continue

        imax = int(np.argmax(rds))
        for rds_value, time_value in zip(rds, times):
            rows.append({
                "Test_ID": int(test_id),
                "Run_ID": int(run_id),
                "Transient_ID": int(transient_id),
                "Target_Condition": ",".join(str(x) for x in TARGET_CONDITION),
                "Time_us": float(time_value),
                "RDS_Ohm": float(rds_value),
                "Provisional_Min_Current_A": float(PROVISIONAL_MIN_CURRENT_A),
                "Data_Format_Version": "SPAD_V4_WEB_CSV_V2",
            })

    return rows

# ============================================================
# READ RUN-LEVEL CSV
# ============================================================
if not RUNLEVEL_FILE.exists():
    raise FileNotFoundError(RUNLEVEL_FILE)

runlevel = pd.read_csv(RUNLEVEL_FILE)
required = {
    "Test_ID", "Run_ID", "Cumulative_Aging_Hours",
    "RDSon_Median_Ohm", "ID_ON_Median_A",
}
missing = sorted(required - set(runlevel.columns))
if missing:
    raise ValueError(
        "Run-level CSV is missing required columns: " + ", ".join(missing)
    )

runlevel["Test_ID"] = pd.to_numeric(runlevel["Test_ID"], errors="raise").astype(int)
runlevel["Run_ID"] = pd.to_numeric(runlevel["Run_ID"], errors="raise").astype(int)

# ============================================================
# CREATE COMPACT TRANSIENT TABLE
# ============================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

transient_rows = []
for row in runlevel[["Test_ID", "Run_ID"]].itertuples(index=False):
    test_id, run_id = int(row.Test_ID), int(row.Run_ID)
    matches = list(DATA_ROOT.rglob(f"Test_{test_id}_run_{run_id}.mat"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find Test_{test_id}_run_{run_id}.mat under {DATA_ROOT}"
        )
    transient_rows.extend(
        extract_transient_rows(matches[0], test_id, run_id)
    )

transient_df = pd.DataFrame(transient_rows)
if transient_df.empty:
    raise ValueError("No target-condition transient evidence was extracted.")

# ============================================================
# CREATE COMPONENT TABLE
# Keep all run-level columns used by Module A/B plus metadata.
# ============================================================
component_df = runlevel.copy()
component_df.insert(
    len(component_df.columns),
    "Data_Format_Version",
    "SPAD_V4_WEB_CSV_V2",
)
component_df.insert(
    len(component_df.columns),
    "Target_Condition",
    ",".join(str(x) for x in TARGET_CONDITION),
)
component_df.insert(
    len(component_df.columns),
    "Provisional_Min_Current_A",
    PROVISIONAL_MIN_CURRENT_A,
)

# ============================================================
# WRITE CSVs
# Use UTF-8 and normal quoting. No giant JSON columns.
# ============================================================
component_df.to_csv(
    COMPONENT_CSV,
    index=False,
    encoding="utf-8",
    quoting=csv.QUOTE_MINIMAL,
)
transient_df.to_csv(
    TRANSIENT_CSV,
    index=False,
    encoding="utf-8",
    quoting=csv.QUOTE_MINIMAL,
)

# ============================================================
# CREATE ZIP BUNDLE
# ============================================================
with zipfile.ZipFile(
    BUNDLE_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
) as zf:
    zf.write(COMPONENT_CSV, arcname="component_data.csv")
    zf.write(TRANSIENT_CSV, arcname="transient_evidence.csv")

print("=" * 90)
print("SPAD V4 COMPACT WEB INPUT BUNDLE CREATED")
print("=" * 90)
print(f"Physical Test IDs          : {component_df['Test_ID'].nunique()}")
print(f"Component/run rows         : {len(component_df)}")
print(f"Transient evidence rows    : {len(transient_df)}")
print(f"Valid RDS samples covered  : {len(transient_df)}")
print(f"Target condition           : {TARGET_CONDITION}")
print(f"Minimum valid current (A)  : {PROVISIONAL_MIN_CURRENT_A}")
print(f"Component CSV              : {COMPONENT_CSV}")
print(f"Transient evidence CSV     : {TRANSIENT_CSV}")
print(f"ZIP bundle                 : {BUNDLE_ZIP}")
print("\nNOTE: No giant JSON column is stored. Every valid instantaneous")
print("RDS sample is a normal CSV row, so Module C can reproduce the")
print("exact sample-level exceedance count and first-exceedance logic.")
