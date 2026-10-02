# SPAD V4 R1.1 Hardened — Deployment Runbook

## 1. Render environment

Set:

- `SPAD_MAX_UPLOAD_BYTES=262144000` (or a lower value appropriate for your service)
- `SPAD_API_KEY=<strong-random-secret>`
- `SPAD_REQUIRE_API_KEY=true` (default; keep this enabled in deployment)

Keep the existing Render start command from the Dockerfile.

## 2. Post-deploy health check

Call:

`GET /health`

Expected:

- HTTP 200
- `status = "ok"`
- `serviceRevision = "R1"`
- `serviceBuild = "R1.1-HARDENED"`
- `apiKeyRequired = true`
- `randomForest = "loaded"`
- `isolationForest = "loaded"`
- `inputFormat = "SPAD_V4_WEB_CSV_V2"`
- `modelPackageSha256 = cdb0990104b484c213f3d4c5b1a671c673e2391bbe80d1442055983f26ecbdad`

## 3. Screening smoke test

Use the same `SPAD_V4_WEB_INPUT_V2.zip` regression bundle used in the audit.

POST multipart fields:

- `file` = ZIP bundle
- `lotId` = test lot identifier
- `engineeringLimits` = `{"rdson":{"limitValue":0.7,"direction":"UPPER"}}`
- header `X-SPAD-API-Key` = deployment key when enabled

Expected regression totals for the same V2 bundle:

- 15 component records
- 3,656,923 valid RDS samples
- 6,900 transient groups
- 327,975 violating samples at 0.7 Ω
- 1,956 violating transient groups at 0.7 Ω
- 27 violating runs at 0.7 Ω

## 4. Response-semantic checks

Confirm:

- `evaluation.normalizedCheckpointProgressPercent` is `[0, 33.3333, 66.6667, 100]`.
- `evaluation.temporalCausalityClaim` is `false`.
- `prediction.predictedNormalizedEndpoint` is present.
- `prediction.retrospectiveEndpointResidual` / absolute / relative error fields are present on completed trajectories.
- `futureRiskScore` and `futureRiskPercent` are absent.
- `lotAnomaly.Module_A_Authority = "EVIDENCE_ONLY"`.
- `engineering.rdson.limitExceedanceCount` is sample count.
- `engineering.rdson.limitExceedanceTransientCount` is unique run/transient-group count.
- `engineering.rdson.limitExceedanceRunCount` is violating run count.
- maximum evidence contains both Run_ID and Transient_ID.

## 5. Node changes required before full-stack release

Apply `NODE_REVISED_CONTRACT.md`.

In particular, remove the old physical-time UI field mapping, remove risk-score synthesis, stop using Module A as overall engineering status, preserve R1 engineering counts, and obtain the current catalog limit from the authoritative component/specification context.

## 6. Rollback

Keep the previous Render deployment/image available until the R1 smoke test and Node integration test complete successfully.

## 7. Dependency verification note

`requirements.txt` pins the service dependencies, including scikit-learn 1.8.0 to match the frozen model. A fresh external package-resolution/install was not performed in this offline audit environment; Render must complete a clean build from the committed requirements before production traffic is enabled.


## 8. Node-to-Python authentication

Configure the Node/Render environment with `SPAD_PYTHON_API_KEY` equal to the Python service `SPAD_API_KEY`. `aiService.js` must send that value as the `X-SPAD-API-Key` request header. Never expose the secret in React or browser configuration.
