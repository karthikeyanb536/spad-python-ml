# SPAD V4 R1.1 Hardened — Release Checklist

## Python R1.1 hardened build

- [x] Model hash checked before pickle unpickling.
- [x] Separate serviceBuild marker distinguishes hardened code from frozen model revision.
- [x] Model package contract validated at startup.
- [x] API key fail-closed by default.
- [x] Authentication guard runs before route multipart parsing.
- [x] Upload size hard ceiling.
- [x] ZIP member and total uncompressed limits (350 MB per member / 400 MB total).
- [x] Exact two-member/top-level ZIP contract.
- [x] Strict ID validation without float64 rounding.
- [x] Exact Module A feature-schema validation.
- [x] Exact Module B frozen-parameter validation.
- [x] Streamed request-body size guard for chunked/no-Content-Length uploads.
- [x] Strict condition/version/current-threshold validation.
- [x] Transient `(Test_ID, Run_ID)` referential integrity.
- [x] Complete evidence coverage for every component run.
- [x] Correct sample/transient/run exceedance semantics.
- [x] Explicit Module A evidence-only semantics.
- [x] Explicit normalized-endpoint semantics.
- [x] No future-risk score generation.
- [x] scikit-learn pinned to the model training version.
- [x] Regression and API tests pass.

## Before changing Render

- [ ] Set `SPAD_API_KEY` to a strong secret.
- [ ] Keep `SPAD_REQUIRE_API_KEY=true`.
- [ ] Deploy the final R1.1 hardened Python source.
- [ ] Run `/health` and confirm `serviceBuild=R1.1-HARDENED`.
- [ ] Run the supplied smoke test against `SPAD_V4_WEB_INPUT_V2.zip`.
- [ ] Confirm model hash exactly matches the expected hash in the runbook.

## Before full-stack release

- [ ] Node sends only `rdson` to Python R1.
- [ ] Node no longer creates `futureRiskScore` / `futureRiskPercent`.
- [ ] Node uses `prediction.predictedNormalizedEndpoint`.
- [ ] Node uses preferred retrospective endpoint error fields.
- [ ] Node preserves sample/transient-group/run counts with their correct semantics.
- [ ] Node preserves max-evidence Run_ID + Transient_ID.
- [ ] Module A is evidence-only and cannot set engineering/overall status.
- [ ] Current engineering specification is resolved from authoritative component/specification context, not a previous screening document keyed only by lot.
- [ ] Engineering specification provenance/version is persisted.
- [ ] React labels the result as a normalized endpoint estimate, not a guaranteed physical-time forecast.
- [ ] Full React → Node → Python → MongoDB regression test passes.
