from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import inference

LIMITS = {"rdson": {"limitValue": 0.7, "direction": "UPPER", "unit": "ohm", "source": "TEST"}}

COMPONENT_HEADER = (
    "Test_ID,Run_ID,Cumulative_Aging_Hours,RDSon_Median_Ohm,ID_ON_Median_A,"
    "Data_Format_Version,Target_Condition,Provisional_Min_Current_A\n"
)
TRANSIENT_HEADER = (
    "Test_ID,Run_ID,Transient_ID,Time_us,RDS_Ohm,Target_Condition,"
    "Provisional_Min_Current_A,Data_Format_Version\n"
)


def make_component(rows: list[str]) -> str:
    return COMPONENT_HEADER + "\n".join(rows) + "\n"


def make_transient(rows: list[str]) -> str:
    return TRANSIENT_HEADER + "\n".join(rows) + "\n"


def make_zip(component_name="component_data.csv", transient_name="transient_evidence.csv", duplicate=False, component_rows=None, transient_rows=None) -> bytes:
    c = make_component(component_rows or [
        "6,1,0.0,0.50,7.0,SPAD_V4_WEB_CSV_V2,199,200,10,5,1000,40,0.15",
        "6,2,1.0,0.55,6.9,SPAD_V4_WEB_CSV_V2,199,200,10,5,1000,40,0.15",
    ])
    # Normalize the odd CSV row shape above explicitly: use target condition as one CSV field.
    c = "Test_ID,Run_ID,Cumulative_Aging_Hours,RDSon_Median_Ohm,ID_ON_Median_A,Data_Format_Version,Target_Condition,Provisional_Min_Current_A\n" + \
        "\n".join(component_rows or [
            "6,1,0.0,0.50,7.0,SPAD_V4_WEB_CSV_V2,\"199,200,10,5,1000,40\",0.15",
            "6,2,1.0,0.55,6.9,SPAD_V4_WEB_CSV_V2,\"199,200,10,5,1000,40\",0.15",
        ]) + "\n"
    t = "Test_ID,Run_ID,Transient_ID,Time_us,RDS_Ohm,Target_Condition,Provisional_Min_Current_A,Data_Format_Version\n" + \
        "\n".join(transient_rows or [
            "6,1,1,50.0,0.60,\"199,200,10,5,1000,40\",0.15,SPAD_V4_WEB_CSV_V2",
            "6,1,1,60.0,0.75,\"199,200,10,5,1000,40\",0.15,SPAD_V4_WEB_CSV_V2",
            "6,2,1,50.0,0.56,\"199,200,10,5,1000,40\",0.15,SPAD_V4_WEB_CSV_V2",
            "6,2,1,60.0,0.57,\"199,200,10,5,1000,40\",0.15,SPAD_V4_WEB_CSV_V2",
        ]) + "\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr(component_name, c)
        z.writestr(transient_name, t)
        if duplicate:
            z.writestr(component_name, c)
    return buf.getvalue()


def expect_error(name, payload, needle):
    try:
        inference.run_screening(payload, "TEST", LIMITS, "x.zip")
    except ValueError as e:
        msg = str(e)
        assert needle.lower() in msg.lower(), (name, msg, needle)
        print(f"PASS {name}: {msg}")
        return
    raise AssertionError(f"{name}: expected ValueError")

# Clean mini bundle, bytes mode.
out = inference.run_screening(make_zip(), "TEST", LIMITS, "x.zip")
assert out['serviceRevision'] == 'R1'
assert out['dataFormatVersion'] == inference.INPUT_FORMAT
assert len(out['records']) == 1
r = out['records'][0]
e = r['engineering']['rdson']
assert e['limitExceedanceCount'] == 1
assert e['limitExceedanceTransientCount'] == 1
assert e['limitExceedanceRunCount'] == 1
assert e['maxEvidenceRunId'] == 1
assert r['evaluation']['temporalCausalityClaim'] is False
assert 'futureRiskScore' not in r['prediction']
print('PASS clean mini bundle')

# File-object mode.
with io.BytesIO(make_zip()) as f:
    out2 = inference.run_screening(f, "TEST", LIMITS, "x.zip")
assert out2['records'][0]['engineering']['rdson']['limitExceedanceCount'] == 1
print('PASS file-object mode')

expect_error('fractional component Test_ID', make_zip(component_rows=[
    "6.5,1,0.0,0.50,7.0,SPAD_V4_WEB_CSV_V2,\"199,200,10,5,1000,40\",0.15",
    "6.5,2,1.0,0.55,6.9,SPAD_V4_WEB_CSV_V2,\"199,200,10,5,1000,40\",0.15",
]), needle='integer-valued IDs')

expect_error('orphan Run_ID', make_zip(transient_rows=[
    "6,1,1,50.0,0.60,\"199,200,10,5,1000,40\",0.15,SPAD_V4_WEB_CSV_V2",
    "6,99,2,50.0,0.60,\"199,200,10,5,1000,40\",0.15,SPAD_V4_WEB_CSV_V2",
]), needle='absent from component_data.csv')

expect_error('unknown Test_ID', make_zip(transient_rows=[
    "99,1,1,50.0,0.60,\"199,200,10,5,1000,40\",0.15,SPAD_V4_WEB_CSV_V2",
]), needle='Test_ID values absent')

expect_error('missing evidence pair', make_zip(transient_rows=[
    "6,1,1,50.0,0.60,\"199,200,10,5,1000,40\",0.15,SPAD_V4_WEB_CSV_V2",
]), needle='missing evidence for component')

expect_error('bad Target_Condition', make_zip(component_rows=[
    "6,1,0.0,0.50,7.0,SPAD_V4_WEB_CSV_V2,\"198,200,10,5,1000,40\",0.15",
    "6,2,1.0,0.55,6.9,SPAD_V4_WEB_CSV_V2,\"198,200,10,5,1000,40\",0.15",
]), needle='Target_Condition')

expect_error('nested path', make_zip(component_name='nested/component_data.csv'), needle='exactly top-level')
expect_error('duplicate member name', make_zip(duplicate=True), needle='exactly component_data.csv')
expect_error('lower RDS direction', make_zip(), needle='direction=UPPER') if False else None
try:
    inference.run_screening(make_zip(), 'TEST', {'rdson': {'limitValue':0.7,'direction':'LOWER'}}, 'x.zip')
except ValueError as e:
    assert 'direction=UPPER' in str(e)
    print('PASS lower RDS direction')
else:
    raise AssertionError('lower RDS direction should fail')

# Unsupported engineering parameters are rejected instead of silently ignored.
try:
    inference.run_screening(
        make_zip(),
        'TEST',
        {
            'rdson': {'limitValue': 0.7, 'direction': 'UPPER'},
            'temp': {'limitValue': 200, 'direction': 'UPPER'},
        },
        'x.zip',
    )
except ValueError as e:
    assert 'Unsupported engineering parameters' in str(e)
    print('PASS unsupported engineering parameter')
else:
    raise AssertionError('unsupported engineering parameter should fail')

# No RDS limit: transient evidence is still validated, but compliance is not evaluated.
out3 = inference.run_screening(make_zip(), 'TEST', {}, 'x.zip')
assert out3['records'][0]['engineering']['rdson']['limitExceedanceFlag'] == 'NOT_EVALUATED'
assert out3['records'][0]['engineering']['rdson']['validRdsSampleCount'] == 4
assert out3['records'][0]['engineering']['rdson']['targetConditionTransientCount'] == 2
print('PASS no rdson limit')

# No-limit requests must still reject malformed transient evidence.
bad_no_limit = make_zip(transient_rows=[
    '6,1,1,50.0,0.60,"198,200,10,5,1000,40",0.15,SPAD_V4_WEB_CSV_V2',
    '6,2,2,50.0,0.60,"198,200,10,5,1000,40",0.15,SPAD_V4_WEB_CSV_V2',
])
try:
    inference.run_screening(bad_no_limit, 'TEST', {}, 'x.zip')
except ValueError as e:
    assert 'Target_Condition' in str(e)
    print('PASS no-limit transient validation')
else:
    raise AssertionError('no-limit request should validate transient evidence')

# Full real V2 bundle regression.
with open('/mnt/data/SPAD_V4_WEB_INPUT_V2.zip','rb') as f:
    real = inference.run_screening(f, 'FULL_AUDIT', LIMITS, 'SPAD_V4_WEB_INPUT_V2.zip')
assert len(real['records']) == 15
E = lambda r: r['engineering']['rdson']
assert sum(E(r)['validRdsSampleCount'] for r in real['records']) == 3656923
assert sum(E(r)['targetConditionTransientCount'] for r in real['records']) == 6900
assert sum(E(r)['limitExceedanceCount'] for r in real['records']) == 327975
assert sum(E(r)['limitExceedanceTransientCount'] for r in real['records']) == 1956
assert sum(E(r)['limitExceedanceRunCount'] for r in real['records']) == 27
print('PASS full V2 regression: 15 devices, 3,656,923 samples, 6,900 transient groups')

print('ALL R1 TESTS PASSED')
