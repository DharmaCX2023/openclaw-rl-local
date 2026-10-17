import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/generate_opsd_evidence_anchors.py"
SPEC = importlib.util.spec_from_file_location("generate_opsd_evidence_anchors", SCRIPT)
generator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(generator)


def test_extract_xml_recovers_only_complete_records_from_length_limit():
    raw = """<evidences>
<evidence><condition>a & b < 3</condition><artifact></artifact><verifiedFact>fact</verifiedFact><explanation>why</explanation></evidence>
<evidence><condition>unfinished"""
    parsed = generator.parse_evidence_anchor_xml(
        generator._extract_xml(raw), task_id="task_test"
    )
    assert len(parsed["anchors"]) == 1
    assert parsed["anchors"][0]["condition"] == "a & b < 3"
    assert parsed["anchors"][0]["verified_fact"] == "fact"


def test_sanitize_clears_unlisted_paths_and_deduplicates_conditions():
    record = {
        "anchors": [
            {"condition": "Write result", "artifact": "output/result.json"},
            {"condition": " Write   Result ", "artifact": "hidden/answer.json"},
            {"condition": "Document it", "artifact": "hidden/answer.json"},
        ]
    }
    cleaned = generator._sanitize_artifacts(
        record, user_query="Write output/result.json and document it."
    )
    assert len(cleaned["anchors"]) == 2
    assert cleaned["anchors"][0]["artifact"] == "output/result.json"
    assert cleaned["anchors"][1]["artifact"] == ""


def test_publicly_requested_test_terms_are_not_false_positive():
    record = {
        "anchors": [
            {
                "condition": "Write pytest coverage",
                "artifact": "output/test_plan.py",
                "verified_fact": "pytest exercises the public contract",
                "explanation": "The task explicitly requests pytest tests",
            }
        ]
    }
    errors = generator._audit_record(
        record,
        user_query="Write pytest tests to output/test_plan.py.",
        verifier="",
    )
    assert errors == []
