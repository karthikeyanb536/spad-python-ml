# SPAD V4 — R1.1 Hardened Audit and Deployment Gate

## Scope

Reviewed the complete R1 Python deployment bundle:

- `inference.py`
- `app.py`
- `requirements.txt`
- `Dockerfile`
- `.dockerignore`
- `MODEL_SHA256.txt`
- model package
- regression / adversarial tests
- smoke test
- README, deployment runbook, and Node integration contract

Also re-checked the previously audited Node/React integration lineage where source evidence was available.

## Executive status

### Production-code critical issues found in R1 and fixed in R1.1

| Issue | R1.1 status |
|---|---|
| Model pickle could be loaded before its manifest hash was checked | FIXED — hash is verified before `pickle.loads()` |
| API authentication was optional by default | FIXED — fail-closed by default; explicit opt-out required for controlled local testing |
| Authentication happened in the route after multipart parsing | FIXED — middleware guard runs before the route |
| Oversized HTTP requests could bypass early size rejection | HARDENED — Content-Length preflight plus route/file-size enforcement |
| Non-RDS engineering parameters were silently accepted/ignored by Python | FIXED — Python rejects unsupported parameters |
| No-limit requests skipped transient evidence validation | FIXED — transient evidence is validated even when no RDS limit is supplied |
| Model package could be structurally wrong while still importing | HARDENED — package contract checks for revision, operating condition, reference/holdout separation, Module A trees/scores, and Module B structure |
| Invalid boolean values could be coerced to numeric limit values | FIXED — booleans rejected as numeric engineering limits |
| Component age span overflow was not explicitly handled | FIXED — non-finite span rejected |
| Component CSV could be disproportionately large | HARDENED — explicit component-member uncompressed cap |
| Parser/Unicode/ZIP payload errors could surface as generic 500 responses | FIXED — classified as 400 invalid telemetry |
| Ambiguous Module A semantics | FIXED — explicit evidence-only / non-probability metadata |
| Ambiguous retrospective Module B error naming | IMPROVED — explicit retrospective endpoint error aliases added |
| Engineering count semantics were still easy to confuse downstream | IMPROVED — explicit sample-count and transient-group-count aliases added |

## Python R1.1 model semantics

The frozen model objects were not retrained.

The deployed service now states explicitly:

- checkpoints are 0%, 33.33%, 66.67%, and 100% normalized progress;
- 0% is the earliest observed trajectory point in the submitted bundle;
- 100% is the latest observed trajectory point in the submitted bundle;
- there is no fixed physical-time checkpoint claim;
- Module B estimates the RDS value at the submitted trajectory's normalized endpoint;
- Module B residual/error fields are retrospective validation evidence when the endpoint is included;
- Module A is reference-population outlier evidence only;
- Module A is not a probability or engineering decision.

A genuine causal fixed-time early-life forecast still requires a redesigned training/label protocol. R1.1 prevents the deployed API from claiming otherwise.

## Model integrity

The package hash in `MODEL_SHA256.txt` matches the deployed package:

`cdb0990104b484c213f3d4c5b1a671c673e2391bbe80d1442055983f26ecbdad`

The R1.1 loader verifies this hash before unpickling the package.

Additional startup checks verify:

- package version `SPAD_V4`;
- service revision `R1`;
- operating condition `(199, 200, 10, 5, 1000, 40)`;
- Module A reference population is unique;
- held-out IDs do not overlap Module A reference IDs;
- reference population size matches metadata;
- four expected Module A stages exist;
- each Isolation Forest has 500 trees and 13 reference scores;
- Module B features remain `RDS0`, `RDS33`, `ID33_ID0_Ratio`;
- target remains `RDS100`;
- Module B has 3 input features and 100 fitted trees;
- Module C provisional current threshold is finite and positive.

## Module C correctness

Transient identity is:

`(Test_ID, Run_ID, Transient_ID)`

The service now reports:

- `limitExceedanceCount` and `limitExceedanceSampleCount` = violating samples;
- `limitExceedanceTransientCount` and `limitExceedanceTransientGroupCount` = unique violating run/transient groups;
- `limitExceedanceRunCount` = violating runs.

Maximum-RDS evidence includes both Run_ID and Transient_ID.

Transient evidence referential integrity is enforced bidirectionally:

- no unknown Test_ID;
- no unknown `(Test_ID, Run_ID)` pair;
- no missing transient evidence for any component run.

Even when no RDS limit is supplied, transient evidence is still schema/condition/referentially validated.

## Module A findings

The frozen reference population contains 13 devices.

Self-flag counts when the fitted models are scored on their own reference devices:

- 0%: 8/13
- 33.33%: 6/13
- 66.67%: 5/13
- 100%: 4/13

These are not independent false-positive estimates.

R1.1 exposes this signal as:

`Module_A_Authority = EVIDENCE_ONLY`

and additionally records that the signal is not a probability, failure prediction, or engineering disposition.

The reference population is historical/frozen; cross-lot domain shift is not calibrated.

## Module B findings

Development remains based on 13 normal physical devices.

The documented nested-LOO benchmark and the current frozen-configuration LOO recomputation remain evidence from the same small development population, not independent external validation.

The current model therefore remains suitable for:

- normalized endpoint estimation;
- retrospective endpoint-error analysis on completed trajectories.

It is not represented by R1.1 as:

- a calibrated failure probability;
- a guaranteed fixed-time forecast;
- an independent validation-certified predictor.

## Security and resource audit

Final hardening pass after the initial R1.1 audit added exact integer-ID lexical validation, stricter frozen-model feature/parameter contract checks, a streamed request-body size guard for chunked/no-Content-Length requests, and tighter ZIP expansion ceilings (400 MB total / 350 MB per member while preserving the 312.6 MB audited transient bundle).

R1.1 protections:

- fail-closed API-key authentication;
- constant-time API-key comparison;
- authentication middleware before multipart route handling;
- Content-Length preflight;
- streamed request-body size enforcement for chunked/no-Content-Length requests;
- configurable upload limit with hard 250 MB ceiling;
- exact two-member ZIP contract;
- exact top-level filenames;
- per-member uncompressed limits;
- total uncompressed limit (400 MB);
- per-member uncompressed limit (350 MB);
- component CSV uncompressed cap;
- no ZIP extraction to arbitrary paths;
- hash verification before pickle loading;
- bounded engineering-limit JSON size;
- strict identifier parsing without float64 conversion/rounding;
- exact Module A feature-schema and Module B parameter contract checks.

Remaining general operational controls that belong at the platform/proxy layer rather than in this small service include rate limiting, WAF policy, request concurrency limits, and external observability.

## Regression tests executed

### Application tests

`pytest -q test_api_r1.py`

Result:

`3 passed`

Covered:

- healthy service response;
- fail-closed startup when no API key is configured;
- unauthorized screening request rejected by the pre-route guard.

### Inference tests

`python test_r1.py`

Result:

`ALL R1 TESTS PASSED`

Covered:

- clean bytes mode;
- file-object mode;
- fractional IDs;
- orphan Run_ID;
- unknown Test_ID;
- missing evidence pair;
- bad target condition;
- nested ZIP path;
- duplicate ZIP member;
- invalid RDS direction;
- unsupported engineering parameter;
- no-RDS-limit mode;
- malformed transient evidence even without an engineering limit;
- full V2 regression.

### Full regression totals

For the supplied `SPAD_V4_WEB_INPUT_V2.zip`:

- 15 device records;
- 3,656,923 valid RDS samples;
- 6,900 transient groups;
- 327,975 violating samples at 0.7 Ω;
- 1,956 violating transient groups at 0.7 Ω;
- 27 violating runs at 0.7 Ω.

### Python HTTP regression

FastAPI TestClient returned:

- HTTP 200;
- 15 records;
- model hash matching `/health`;
- normalized endpoint field present;
- no `futureRiskScore` / `futureRiskPercent`;
- Module A probability claim false;
- regression counts unchanged.

### Hash-gate security regression

A deliberately replaced malicious pickle with the original manifest was rejected on SHA mismatch before unpickling, and the test marker proving pickle execution was not created.

## Frozen-model parity

Comparison of original R1 and hardened R1.1 core outputs on the same regression bundle showed only floating-point noise at approximately `5.1e-14` maximum in the prediction/error values; the fitted model objects and substantive outputs are unchanged.

## Deployment files

The R1.1 folder contains a single deployment-ready Python service source plus documentation and tests.

The model package itself remains the previously audited frozen SPAD V4 package.

## Full-stack release gate

Python R1.1 is regression-tested.

The full application is **not yet fully release-cleared** until the Node layer has actually implemented `NODE_REVISED_CONTRACT.md`.

Mandatory Node changes:

1. Do not derive `overallStatus` / engineering status from Module A.
2. Do not generate or persist `futureRiskScore` or `futureRiskPercent`.
3. Map `predictedNormalizedEndpoint`, not the old physical-time field.
4. Preserve R1.1 engineering sample/transient/run counts without substituting one for another.
5. Preserve maximum evidence Run_ID + Transient_ID.
6. Resolve the engineering catalog/specification from the actual component/specification context, not an arbitrary prior screening record keyed only by lot.
7. Persist service revision, data format, model package name/hash, and engineering-limit provenance/version.

Until those Node changes are present in the actual deployed Node code, the Python service can be considered hardened independently, but the complete React → Node → Python → MongoDB chain should remain in integration-test status.

## Final classification

### Python R1.1
**Regression status: PASS**

### Scientific/model scope
**Explicit limitation, not hidden**

### Critical runtime/security defects
**Addressed in R1.1**

### Remaining external dependency
**Actual Node implementation of the revised contract**

### Deployment recommendation

Deploy Python R1.1 only together with the corresponding Node contract changes and the supplied smoke test. Keep the existing production deployment available for rollback until the full-stack regression passes.

## Final hardening verification

The final post-hardening code was re-run locally:

- `python -m py_compile inference.py app.py test_r1.py test_api_r1.py` — PASS.
- `python test_r1.py` — PASS; full regression remains 15 devices / 3,656,923 samples / 6,900 transient groups / 327,975 violating samples / 1,956 violating transient groups / 27 violating runs.
- `pytest -q test_api_r1.py` — 3 passed.
- FastAPI TestClient full V2 regression — HTTP 200, 15 records, model hash parity, corrected semantic fields and corrected engineering counts.

A fresh network-based dependency resolution and a live Render verification were not performed in this environment; those remain deployment-time checks.
