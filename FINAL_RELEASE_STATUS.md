# SPAD V4 R1.1 Hardened — Final Release Status

## Python service
PASS — final source review and local regression tests completed.

- serviceRevision: R1
- serviceBuild: R1.1-HARDENED
- model SHA-256: cdb0990104b484c213f3d4c5b1a671c673e2391bbe80d1442055983f26ecbdad
- pytest API tests: 3 passed
- inference regression: all tests passed
- full V2 regression: 15 devices, 3,656,923 valid samples, 6,900 transient groups
- 0.7 ohm regression: 327,975 violating samples, 1,956 violating transient groups, 27 violating runs
- FastAPI TestClient: HTTP 200, 15 records, correct semantic fields

## Critical/runtime hardening
Implemented:
- model hash verification before pickle unpickling
- fail-closed API-key authentication
- pre-route authentication
- streamed request-body size enforcement
- hard upload/ZIP expansion limits
- exact ZIP structure
- strict identifier validation without float64 rounding
- exact frozen model feature/parameter contract
- strict engineering-parameter handling
- complete transient evidence validation
- corrected transient identity and exceedance counting
- explicit non-causal normalized-endpoint semantics
- Module A evidence-only semantics
- model/package provenance

## Scientific scope that remains explicit
- 0/33.33/66.67/100 are normalized trajectory percentages, not fixed physical times
- endpoint normalization uses the submitted trajectory span, so it is not a causal early-life predictor
- Module B development population is 13 physical devices
- Module A reference population is 13 devices and its scores are outlier evidence, not calibrated probabilities
- cross-lot/domain shift is not calibrated
- independent external validation remains limited

## Full-stack release gate
Do not treat the complete React -> Node -> Python -> MongoDB application as release-cleared until Node actually implements NODE_REVISED_CONTRACT.md.

Mandatory Node items:
- send only rdson to Python
- send X-SPAD-API-Key from a server-side secret
- stop synthesizing futureRiskScore/futureRiskPercent
- map predictedNormalizedEndpoint
- preserve sample/transient-group/run counts with correct meanings
- preserve max evidence Run_ID + Transient_ID
- keep Module A out of engineering/overall status
- resolve engineering specification from authoritative component/specification context
- persist serviceBuild, model hash, and engineering-limit provenance/version

## Verification limitations
A fresh external package install/build and live Render smoke test were not performed in this environment. These are deployment-time gates.
