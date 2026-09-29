# SPAD V4 External Python ML Service

Standalone FastAPI wrapper around the validated SPAD V4 model package.

## Endpoints

- `GET /health`
- `POST /run-screening`

## Request

`POST /run-screening` uses multipart form data:

- `file`: ZIP dataset bundle containing the run-level CSV and raw MATLAB `.mat` files.
- `lotId`: runtime lot ID.
- `engineeringLimits`: JSON string, e.g.

```json
{
  "rdson": {
    "limitValue": 8.0,
    "unit": "Ω",
    "direction": "UPPER"
  }
}
```

The live service processes the uploaded ZIP. It does not fall back to a local NASA dataset.

## Expected dataset bundle

At minimum:

- a run-level CSV such as `MOSFET_199_200C_V2_Electrical_Features_RunLevel.csv`
- raw files named `Test_<Test_ID>_run_<Run_ID>.mat`

If `MOSFET_Normalized_4Checkpoints.csv` is included it is used; otherwise the service reconstructs the existing four-checkpoint interpolation from the run-level dataset.

## Deployment

Deploy this directory as a Render Docker Web Service. The model package is under `models/SPAD_V4_Final_Model_Package.pkl`.

The service exposes `/health` and `/run-screening` and keeps the ML models in Python.
