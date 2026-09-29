from pathlib import Path

from inference import MODEL_PACKAGE, canonical_component_id, validate_engineering_limits


def test_health_artifacts():
    assert MODEL_PACKAGE["package_version"] == "SPAD_V4"
    assert MODEL_PACKAGE["module_b"]["prediction_field"] == "Predicted_RDS100"
    assert set(MODEL_PACKAGE["module_a"]["models"].keys()) == {0, 33, 66, 100}


def test_component_id():
    assert canonical_component_id(6) == "TEST-06"
    assert canonical_component_id(34) == "TEST-34"


def test_limits():
    out = validate_engineering_limits({"rdson": {"limitValue": 8.0, "unit": "Ω", "direction": "UPPER"}})
    assert out["rdson"]["limitValue"] == 8.0


def test_model_file_exists():
    assert Path("models/SPAD_V4_Final_Model_Package.pkl").exists()
