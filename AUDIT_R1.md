# SPAD V4 — Full Remaining Audit + Deployment Revision R1

## Audit scope

This audit covers the current CSV V2 deployment source, the frozen SPAD V4 model package, the supplied `SPAD_V4_WEB_INPUT_V2.zip` regression bundle, and the documented Node/React integration lineage examined during the prior system audit.

The current R1 revision is based on the CSV V2 source at the last audited deployment commit. The older MAT-based `inference.py` artifact was not used as the R1 implementation.

---

## 1. Executive findings

| Area | Finding | Severity | R1 disposition |
|---|---|---:|---|
| Checkpoint construction | 0%, 33.33%, 66.67%, 100% are normalized progress over each device's submitted observed trajectory; 33.33% therefore depends on the trajectory endpoint. | 🔴 Critical modeling limitation | Semantics corrected; no causal claim made. A truly causal version requires a different feature/label design. |
| Physical-time interpretation | The 100% point is the latest submitted observation, not a guaranteed fixed physical-time endpoint. | 🔴 Critical semantic limitation | All deployed checkpoint terminology is percentage-based. |
| Module A validation | The frozen Isolation Forest self-flags 8/13 reference devices at 0%, 6/13 at 33.33%, 5/13 at 66.67%, and 4/13 at 100%. This is not an independent false-positive estimate; it demonstrates that `contamination=auto` with this small reference population is not a calibrated specificity measurement. | 🟠 Major | Module A is explicitly evidence-only. |
| Module A novelty | Novelty is an empirical extremeness rank against frozen in-sample reference scores, with discrete 1/13 increments. It is not a probability. | 🟠 Major | Semantics documented in metadata/output. |
| Module B statistics | Development is only 13 physical normal devices. The documented nested-LOO benchmark is not an external validation estimate. | 🟠 Major | Metrics are labeled as protocol evidence, not deployment certification. |
| Module B evaluation | Frozen-configuration LOO recomputation on the current V2 web bundle: MAE 0.03877335 Ω; RMSE 0.05469440 Ω. | 🟠 Major limitation | Stored as reproducibility metadata; not presented as independent generalization performance. |
| Module B held-out behavior | Test 10 and Test 13 show large endpoint errors, confirming the model is useful as a normal-endpoint reference signal but not an accurate extrapolation for those abnormal trajectories. | 🟠 Major | Preserve as evaluation evidence, not a risk probability. |
| Module C transient identity | `Transient_ID` is run-local. The previous implementation counted by `Transient_ID` alone, causing collisions across runs. | 🔴 Confirmed bug | R1 uses `(Test_ID, Run_ID, Transient_ID)`. |
| Module C exceedance count | Previous API serialization mapped transient-group count into `limitExceedanceCount`. | 🔴 Confirmed bug | R1 `limitExceedanceCount` is sample count; transient and run counts are separate. |
| Module C chronology | `Time_us` restarts for each transient, so the prior "first exceedance" was not a component-wide chronological first event. | 🟠 Major semantic issue | R1 renames this evidence to earliest local exceedance and includes Run_ID. |
| Transient referential integrity | Previous runtime did not require every transient `(Test_ID, Run_ID)` pair to exist in component data. | 🟠 Major validation gap | R1 rejects unknown/orphan pairs and missing component-run evidence. |
| Identifier parsing | Previous `.astype(int)` could silently truncate fractional IDs. | 🟠 Validation gap | R1 rejects non-integer IDs. |
| Engineering authority | Node's catalog limit recovery was based on an existing screening record for the same lot, which can become stale or ambiguous. | 🔴 System governance risk | R1 contract requires current catalog/specification provenance to be resolved upstream. |
| Overall status authority | Previous Node mapping allowed Module A's ML flag to feed `overallStatus`. | 🔴 System governance risk | R1 contract requires engineering status to come from engineering evidence/specification logic, not Module A. |
| Future risk | Previous Node preserved/scaled `futureRiskScore` / `futureRiskPercent` despite no calibrated risk model. | 🔴 Semantic risk | R1 Python emits neither field; Node must stop synthesizing them. |
| Physical-time UI name | Previous Node mapped `Predicted_RDS100` to a physical-time UI field. | 🔴 Semantic risk | R1 exposes `predictedNormalizedEndpoint`; Node migration is required. |
| Model/runtime versions | Historical deployment loaded a package trained under scikit-learn 1.8.0 with runtime 1.6.1 and emitted `InconsistentVersionWarning`. | 🔴 Deployment reproducibility risk | R1 pins scikit-learn 1.8.0. |
| Upload resource handling | Previous app read the whole upload and created another in-memory ZIP buffer; ZIP expansion limits were not bounded. | 🟠 Security/resource risk | R1 streams the FastAPI file object and adds compressed/uncompressed limits. |
| Health endpoint | Previous `/health` could return 200 even if models were unavailable. | 🟠 Operational risk | R1 returns 503 when required models are not loaded. |
| Provenance | Prior production traces showed metadata drift from the current committed contract. | 🔴 Deployment governance risk | R1 reports service revision, input format, package name and SHA-256 hash. |

---

## 2. What R1 changes

### Model semantics

- Checkpoints are exposed as **0%, 33.33%, 66.67%, and 100% normalized trajectory progress**.
- 0% means the earliest observed point in the submitted device trajectory.
- 100% means the latest observed point in the submitted bundle.
- No fixed physical-time mapping is part of the deployment contract.
- Module B is an endpoint estimate in normalized trajectory space.
- Module B error fields are explicitly retrospective when the submitted bundle contains the endpoint.
- Module A is an evidence-only reference-population outlier signal and is not an engineering disposition or calibrated probability.

### Input validation

R1 enforces:

- exactly two top-level ZIP members;
- exact filenames `component_data.csv` and `transient_evidence.csv`;
- compressed upload size limit;
- per-member and total uncompressed ZIP limits;
- required component and transient columns;
- frozen target condition;
- frozen provisional current threshold;
- exact input format version;
- finite, positive physical measurements;
- integer-valued positive Test_ID, Run_ID and Transient_ID;
- at least two component runs per device;
- strictly increasing component trajectory ages;
- transient-to-component `(Test_ID, Run_ID)` referential integrity;
- complete transient evidence coverage for every component run.

### Module C

R1 defines transient identity using:

`(Test_ID, Run_ID, Transient_ID)`

and reports:

- `limitExceedanceCount` = violating sample count;
- `limitExceedanceTransientCount` = unique violating run/transient groups;
- `limitExceedanceRunCount` = runs containing at least one violating sample.

Maximum-RDS evidence now includes both Run_ID and Transient_ID.

### API / provenance

R1 adds:

- `serviceRevision = R1`;
- `dataFormatVersion = SPAD_V4_WEB_CSV_V2`;
- `modelPackage`;
- `modelPackageSha256`;
- model metadata describing Module A authority, Module B semantics, checkpoint semantics, and Module C counting semantics.

An optional `SPAD_API_KEY` can require `X-SPAD-API-Key` on the screening endpoint.

---

## 3. Frozen-model parity verification

The R1 package was augmented with deployment metadata, but the fitted tree structures were re-verified against the pre-R1 package.

- Isolation Forests: 500 trees at each of 0%, 33.33%, 66.67%, 100% stages.
- Random Forest: 100 trees.
- Tree structure arrays were checked exactly for all fitted trees.
- Model parameters and offsets were checked.
- No retraining was performed for R1.

Pre-R1 package SHA-256:

`17558414104bea30009bae8d44e2a42cbb7cfc7ca8d5de2d0d904ad8be18c7df`

R1 package SHA-256:

`cdb0990104b484c213f3d4c5b1a671c673e2391bbe80d1442055983f26ecbdad`

The hash changed because deployment/audit metadata were added; the fitted tree structures were re-verified as identical.

---

## 4. Full V2 regression result

The R1 service was executed against the full supplied `SPAD_V4_WEB_INPUT_V2.zip`.

Observed result:

- 15 component records returned;
- 3,656,923 valid RDS samples;
- 6,900 unique physical transient groups using `(Test_ID, Run_ID, Transient_ID)`;
- at an RDS(on) limit of 0.7 Ω:
  - 327,975 violating samples;
  - 1,956 violating transient groups;
  - 27 violating runs.

These values are internally consistent with the R1 counting semantics.

The old implementation's run-local ID collision produced a lower transient-group count because the same Transient_ID values recur across different runs.

---

## 5. Module A audit

The frozen reference population contains 13 devices; Test 10 and Test 13 are not in that reference set.

Re-running the frozen Module A models on the current V2 web bundle produced the following self-flag counts among the 13 reference devices:

| Normalized stage | Reference devices flagged | Reference population |
|---|---:|---:|
| 0% | 8 | 13 |
| 33.33% | 6 | 13 |
| 66.67% | 5 | 13 |
| 100% | 4 | 13 |

Any-stage self-flagging occurred for 8 of the 13 reference devices.

Interpretation:

- These are **not false-positive rates** because the same fitted data are being scored in-sample.
- They show that the current Isolation Forest configuration is not a calibrated probability or validated specificity model.
- The runtime flag remains useful as a frozen reference-population outlier signal, but engineering disposition must not be derived from it.

The novelty value is also discrete because the reference population has only 13 devices; the increments are approximately 7.69 percentage points.

---

## 6. Module B audit

### Training design

The model-development evidence documents nested leave-one-out selection on 13 normal physical MOSFETs, with Test 10 and Test 13 held aside until final evaluation.

The final frozen configuration is:

- 100 trees;
- maximum depth 2;
- minimum leaf size 3;
- minimum split size 2;
- all features eligible at each split;
- random state 42.

### Reproducibility result

Using the current V2 web bundle and the exact frozen configuration, the independent device-level LOO recomputation gave:

- MAE = 0.0387733508 Ω;
- RMSE = 0.0546944006 Ω;
- median absolute error = 0.01523024 Ω;
- median relative error = 2.40055%.

These metrics are **not** an external validation estimate because they are still based on the same small 13-device normal development population.

The documented nested-LOO benchmark in the training record was approximately MAE 0.039628 Ω and RMSE 0.055447 Ω. The distinction between the nested protocol benchmark and the final frozen-configuration recomputation is retained in R1 metadata.

### Holdout behavior

The two held-out devices show very large endpoint residuals under the frozen model. This confirms that the frozen normal-reference model does not reproduce those abnormal endpoints and should not be described as a generally accurate endpoint predictor for arbitrary degradation states.

### Core semantic limitation

The feature construction defines the 33.33% and 66.67% checkpoints using the submitted trajectory's eventual maximum observed progress. Therefore the feature itself is not causally available when the endpoint is unknown.

R1 explicitly sets `temporalCausalityClaim = false` to prevent the API from representing this retrospective endpoint estimation as causal live forecasting.

---

## 7. Module C audit

### Previous confirmed defects

1. `Transient_ID` was treated as globally unique even though it restarts across runs.
2. The canonical `limitExceedanceCount` field was populated from transient count rather than sample count.
3. Evidence identity omitted Run_ID.
4. Minimum local `Time_us` across reset transients was previously presented as a component-wide first exceedance.
5. Orphan transient run pairs were not strictly rejected.

### R1 status

All five items are addressed in the Python R1 implementation.

Chronology remains intentionally limited to **local transient time** because the input schema provides reset-like per-transient timing rather than one global component clock. R1 therefore does not claim a global first event.

---

## 8. Node / MongoDB integration audit

The Python service itself is not the only deployment boundary. The prior Node integration has six changes that must be applied before treating the full stack as semantically revised.

### A. Status authority

The prior Node layer normalizes Module A's flag and can place it into `aiAssessment.overallStatus`. Module A must remain evidence-only; it must not determine engineering disposition.

The authoritative engineering status should come from the engineering specification / engineering evidence path.

### B. Prediction naming

The prior Node layer maps `Predicted_RDS100` to a physical-time frontend field. That name is misleading under the normalized-percent semantics.

Recommended canonical field:

`aiAssessment.prediction.parameters.rdson.predictedNormalizedEndpoint`

Python R1 emits this as a compatibility alias while retaining `Predicted_RDS100` for the existing interface.

### C. Remove uncalibrated risk score synthesis

Do not recreate `futureRiskScore` or `futureRiskPercent` from the normalized endpoint estimate. R1 deliberately does not emit those fields.

### D. Correct engineering count mapping

Persist the three distinct fields from Python R1 rather than mapping transient groups into the sample-count field.

### E. Preserve evidence identity

When storing the maximum transient evidence, persist both Run_ID and Transient_ID.

### F. Engineering-limit provenance

Do not use an arbitrary previous screening record as the only authority for the current catalog specification. Resolve the current specification from the actual catalog/component/specification context and store its version/source.

The prior Node source was documented as querying `ScreeningRecord` by lot ID for a prior `DATABASE_CATALOG` field, which creates a potential stale-context risk.

---

## 9. Deployment/runtime audit

### Fixed in R1

- scikit-learn pinned to 1.8.0, matching the frozen package;
- optional API key gate;
- HTTP 503 on incomplete model readiness;
- direct file-object ZIP processing instead of duplicating the full upload in a second byte buffer;
- bounded ZIP member count and uncompressed size;
- exact ZIP member names;
- model package SHA-256 in health and result metadata.

### Remaining operational requirement

The exact Docker runtime was not executed in this analysis container because Docker/Python 3.12 were unavailable here. R1 was syntax-compiled and executed successfully in the available Python environment with the pinned scikit-learn major/minor version.

Render must therefore perform one final deployment smoke test after the new image is built.

---

## 10. Final deployment gate

### Python R1 gate

**PASS** for the audited V2 bundle and local service checks.

### Full stack gate

**CONDITIONALLY READY** pending the Node changes in Section 8 and one post-deploy live smoke test.

The revised Python package should be deployed first; do not treat the full stack as semantically revised until Node no longer converts Module A into engineering status, no longer fabricates a future-risk percentage, and no longer labels the normalized endpoint as a physical-time endpoint.

---

## 11. Verification checklist

After Render deployment:

1. `GET /health` must return HTTP 200 and `serviceRevision: R1`.
2. `modelPackageSha256` must equal the R1 package hash in `MODEL_SHA256.txt`.
3. A V2 bundle smoke test must return exactly 15 records.
4. At a 0.7 Ω test limit, aggregate counts should match the R1 regression totals above for the same V2 bundle.
5. The response must contain no `futureRiskScore` or `futureRiskPercent` fields.
6. The response must expose percentage-based checkpoint semantics only.
7. Node must persist the R1 provenance fields.
8. MongoDB/UI must use the normalized endpoint terminology and preserve Module A as evidence-only.

---

## Audit conclusion

The revision does not silently retrain or change the fitted SPAD V4 models. It corrects the deployment contract, input validation, transient evidence identity, counting semantics, provenance, and the most important semantic overstatements.

The remaining research-level issue is the underlying normalized-trajectory feature construction: a future endpoint is used to define the normalized progress axis. Eliminating that issue requires redesigning the learning problem around fixed, causally available observation times or another endpoint-independent time basis; it should not be hidden by deployment-layer wording.
