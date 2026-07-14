from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aramis.model_utils import compute_binary_thresholds, profile_matrix


def test_threshold_falls_back_to_youden_when_requested_sensitivity_is_unavailable():
    result = compute_binary_thresholds(
        np.array([0, 1]),
        np.array([0.9, 0.1]),
        target_sensitivity=1.1,
    )

    assert result["target_reached"] is False
    assert result["threshold_target"] == result["threshold_youden"]


@pytest.mark.parametrize(
    ("frame", "error"),
    [
        (pd.DataFrame({"other": [[1.0, 2.0]]}), "Missing required columns"),
        (pd.DataFrame({"profile": [[1.0], [1.0, 2.0]]}), "equal length"),
        (pd.DataFrame({"profile": [[1.0, np.nan]]}), "non-finite"),
    ],
)
def test_profile_matrix_rejects_invalid_profiles(frame: pd.DataFrame, error: str):
    with pytest.raises((KeyError, ValueError), match=error):
        profile_matrix(frame, "profile")
