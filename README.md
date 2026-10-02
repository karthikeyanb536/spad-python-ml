# SPAD V4 External ML Service — Deployment Revision R1.1 Hardened

This revision keeps the frozen SPAD V4 ML model objects unchanged and corrects deployment semantics, input validation, resource handling, Module C evidence identity, and provenance reporting.

## Input bundle

The server requires a ZIP containing exactly these two top-level files:

- `component_data.csv`
- `transient_evidence.csv`

`component_data.csv` must include `Test_ID`, `Run_ID`, `Cumulative_Aging_Hours`, `RDSon_Median_Ohm`, `ID_ON_Median_A`, `Data_Format_Version`, `Target_Condition`, and `Provisional_Min_Current_A`.

`transient_evidence.csv` must include `Test_ID`, `Run_ID`, `Transient_ID`, `Time_us`, `RDS_Ohm`, `Target_Condition`, `Provisional_Min_Current_A`, and `Data_Format_Version`.

## Semantic correction

The model uses 0%, 33.33%, 66.67%, and 100% normalized trajectory checkpoints. These are percentage-progress stages of the submitted device trajectory. The service does not claim fixed physical-time checkpoints and does not claim causal early-time prediction from an unknown future endpoint.

Module B `Predicted_RDS100` is therefore an endpoint estimate in normalized trajectory space. When the uploaded bundle contains the endpoint, residual/error values are retrospective validation evidence.

Module A is returned as anomaly/novelty evidence only. It is not an engineering disposition and is not a calibrated probability.

## Module C correction

Transient identity is defined by `(Test_ID, Run_ID, Transient_ID)`. `limitExceedanceCount` now means sample count. Unique violating transient groups and violating runs are returned separately.

Maximum-RDS evidence includes both `maxEvidenceRunId` and `evidenceTransientId`.

## Security/resource hardening

The service streams the uploaded ZIP from the FastAPI file object instead of copying the entire upload into a second in-memory buffer. It checks compressed upload size, ZIP member count, exact filenames, per-member uncompressed size, and total uncompressed size; the HTTP middleware also enforces the request-body limit while receiving chunked uploads.

Set `SPAD_API_KEY` in deployment. API-key protection is fail-closed by default; set `SPAD_REQUIRE_API_KEY=false` only for controlled local testing. The service checks the API key in middleware before multipart body parsing to reduce unauthenticated upload/CPU abuse.

Python evaluates only the `rdson` engineering parameter. Other engineering parameters must be handled by the upstream engineering-governance layer and must not be silently sent to Python.

## Reproducibility

`requirements.txt` pins scikit-learn to 1.8.0, matching the frozen model package. The API reports the model package SHA-256 hash on `/health` and in every successful response.

## Frozen-model provenance

The underlying Isolation Forest and Random Forest tree structures were re-verified against the pre-R1 package before deployment packaging. The pre-R1 package SHA-256 is recorded inside the R1 model package audit metadata. The R1 package hash is exposed by `/health` and successful screening responses. The immutable `MODEL_SHA256.txt` manifest is verified before the pickle is unpickled.


## Build identity

- `serviceRevision`: `R1`
- `serviceBuild`: `R1.1-HARDENED`
- ZIP total uncompressed ceiling: 400 MB
- ZIP member uncompressed ceiling: 350 MB
- component CSV uncompressed ceiling: 50 MB

The audited regression bundle expands to about 313 MB, so it remains below the hardened 400 MB total ceiling.
