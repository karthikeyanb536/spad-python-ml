# Node/Express Contract Changes Required for SPAD V4 R1

Python R1 is backward-compatible on the core measurement/prediction/anomaly field names, but the backend must correct authority and semantics before deployment.

## 1. Module A must not control engineering/overall status

Keep `aiAssessment.lotAnomaly.parameters.rdson.aiFlag` as ML evidence, but do not use it to derive `aiAssessment.overallStatus` or root `engineeringStatus`.

Authoritative engineering status should be determined from the engineering specification / Module C evidence path only.

## 2. Use normalized-percent terminology in the UI

The legacy physical-time label on normalized RDS100 is semantically misleading. Prefer:

`aiAssessment.prediction.parameters.rdson.predictedNormalizedEndpoint`

while retaining the existing Python `Predicted_RDS100` field as the compatibility source.

Display language should refer to a `100% normalized endpoint estimate`, not a guaranteed fixed-time endpoint.

## 3. Remove/disable futureRiskScore until a calibrated risk model exists

The R1 Python service does not generate `futureRiskScore` or `futureRiskPercent`. Do not recreate these fields with an arbitrary scaling rule.

Use the actual normalized-endpoint estimate and clearly labeled Module A evidence instead.

## 4. Engineering count mapping

Python R1:

- `engineering.rdson.limitExceedanceCount` = violating sample count
- `engineering.rdson.limitExceedanceTransientCount` = unique `(Run_ID,Transient_ID)` violating groups
- `engineering.rdson.limitExceedanceRunCount` = violating runs

Do not substitute transient count into the sample-count field.

## 5. Preserve evidence identity

Store both `maxEvidenceRunId` and `evidenceTransientId` whenever transient evidence is persisted. `Transient_ID` is run-local and must not be treated as globally unique.

## 6. Limit source governance

Do not recover the current engineering catalog limit only from an arbitrary previous screening document for the same `lotId`. Resolve the current catalog/specification record using the actual component/specification context, then attach its provenance/version to the screening record.

## 7. Python request contract for engineering limits

The R1 Python service evaluates only:

```json
{"rdson":{"limitValue":<positive finite number>,"direction":"UPPER",...}}
```

Do not forward `temp`, `vgs`, `vds`, `freq`, `dutyCycle`, or other parameters to Python R1. Keep those parameters in the upstream engineering-governance path where their authoritative specification logic is implemented. Python R1 rejects unsupported engineering parameter names instead of silently ignoring them.

## 8. Provenance

Persist and expose:

- `serviceRevision = R1`
- `dataFormatVersion = SPAD_V4_WEB_CSV_V2`
- `modelPackage = SPAD_V4_Final_Model_Package.pkl`
- `modelPackageSha256`
- the authoritative engineering-limit source/version used by Node

This allows every screening result to be tied to the exact deployed model package and engineering specification provenance.


## 9. Preferred R1.1 field names

For the revised UI/data model, prefer these non-ambiguous fields:

- `prediction.predictedNormalizedEndpoint`
- `prediction.retrospectiveEndpointResidual`
- `prediction.retrospectiveEndpointAbsoluteError`
- `prediction.retrospectiveEndpointRelativeErrorPercent`
- `engineering.rdson.limitExceedanceSampleCount`
- `engineering.rdson.limitExceedanceTransientGroupCount`
- `engineering.rdson.limitExceedanceRunCount`

The older compatibility aliases (`Predicted_RDS100`, `Forecast_Residual`, `Absolute_Forecast_Error`, `Relative_Error_Percent`, `limitExceedanceCount`, `limitExceedanceTransientCount`) may remain in the Python response for compatibility, but Node should map the preferred fields into the canonical schema.

Module A should be stored with its `EVIDENCE_ONLY` authority and interpretation; it must not be translated into a probability, failure probability, or engineering status.


## 10. Python authentication header

When Node calls the Python service, `aiService.js` must send:

`X-SPAD-API-Key: ${SPAD_PYTHON_API_KEY}`

The secret must exist only in the Node/Render server environment; do not expose it to React/browser code.

Python returns both:

- `serviceRevision = R1` (frozen model/protocol revision)
- `serviceBuild = R1.1-HARDENED` (hardened service implementation)

Persist both for deployment traceability.
