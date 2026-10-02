
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ["SPAD_REQUIRE_API_KEY"] = "false"
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
import app as app_module


def test_health():
    client = TestClient(app_module.app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["serviceRevision"] == "R1"
    assert body["modelPackageSha256"] == app_module.MODEL_PACKAGE_SHA256


def test_unauthenticated_startup_fails_closed():
    env = os.environ.copy()
    env.pop("SPAD_API_KEY", None)
    env.pop("SPAD_REQUIRE_API_KEY", None)
    p = subprocess.run(
        [sys.executable, "-c", "import app"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0
    assert "SPAD_API_KEY must be configured" in p.stderr


def test_api_key_guard_runs_before_route_body():
    # Reload the app in a subprocess-like isolated interpreter isn't convenient
    # here; middleware behavior is verified by configuring auth in-process.
    env = os.environ.copy()
    env["SPAD_API_KEY"] = "unit-test-secret"
    env["SPAD_REQUIRE_API_KEY"] = "true"
    p = subprocess.run(
        [sys.executable, "-c", "from fastapi.testclient import TestClient; import app; c=TestClient(app.app); r=c.post('/run-screening', data={'lotId':'T','engineeringLimits':'{}'}); print(r.status_code)"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip().endswith("401")
