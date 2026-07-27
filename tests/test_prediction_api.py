"""Unit tests for the immutable-model HTTP request boundary."""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from aramina.prediction_api import _validated_request


def test_prediction_api_accepts_the_minimal_request_contract():
    request = _validated_request(
        json.dumps(
            {
                "analysis_author": "Demo analyst",
                "prediction_comment": "review case",
                "patient_id": "Nova_214",
                "target_side": "LEFT",
            }
        )
    )

    assert request == {
        "analysis_author": "Demo analyst",
        "prediction_comment": "review case",
        "patient_id": "Nova_214",
        "target_side": "left",
    }


@pytest.mark.parametrize(
    "payload, detail",
    [
        ({"analysis_author": "A", "patient_id": "P", "target_side": "up"}, "target_side"),
        ({"analysis_author": "A", "patient_id": "P", "target_side": "left", "model": "x"}, "Unsupported"),
        ({"analysis_author": "", "patient_id": "P", "target_side": "left"}, "analysis_author"),
    ],
)
def test_prediction_api_rejects_invalid_request_contract(payload, detail):
    with pytest.raises(HTTPException, match=detail):
        _validated_request(json.dumps(payload))
