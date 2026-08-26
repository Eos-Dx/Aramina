from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aramina.experiments import polar_basis_compression as polar


@pytest.mark.parametrize(
    "name",
    [
        "config_polar_basis_compression_pilot_v0_1.yaml",
        "config_polar_basis_compression_v0_1.yaml",
    ],
)
def test_committed_polar_configs_validate(name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    config = polar.load_config(root / "config" / "experiments" / name)
    assert config["polar_cakes"]["radial_q_range"] == [2.0, 23.0]
    assert config["polar_cakes"]["azimuthal_range"] == [-180.0, 180.0]


@pytest.mark.parametrize("n_chi", polar.ALLOWED_N_CHI)
def test_all_supported_angular_sector_counts_validate(n_chi: int) -> None:
    root = Path(__file__).resolve().parents[1]
    config = polar.load_config(
        root / "config" / "experiments" / "config_polar_basis_compression_v0_1.yaml"
    )
    config["polar_cakes"]["n_chi"] = n_chi
    polar._validate_config(config)


def test_angular_harmonic_amplitudes_are_rotation_invariant() -> None:
    chi = np.linspace(-175.0, 175.0, 36)
    theta = np.deg2rad(chi)
    q_scale = np.linspace(1.0, 2.0, 16)
    count = np.ones((36, 16))

    def cake(rotation: float) -> np.ndarray:
        angle = theta[:, None] - rotation
        return (
            4.0 * q_scale
            + 0.8 * np.cos(2.0 * angle) * q_scale
            + 0.4 * np.sin(2.0 * angle) * q_scale
            + 0.3 * np.cos(4.0 * angle) * q_scale
        )

    first = polar.angular_harmonic_channels(cake(0.0), count, chi, max_mode=4)
    rotated = polar.angular_harmonic_channels(cake(0.37), count, chi, max_mode=4)
    first_candidate = polar._candidate_tensor(first[None, ...])[0]
    rotated_candidate = polar._candidate_tensor(rotated[None, ...])[0]
    np.testing.assert_allclose(
        first_candidate, rotated_candidate, rtol=1e-10, atol=1e-10
    )


def test_weighted_a0_matches_independent_angular_mean() -> None:
    chi = np.linspace(-175.0, 175.0, 36)
    theta = np.deg2rad(chi)[:, None]
    radial = np.linspace(1.0, 2.0, 12)[None, :]
    intensity = 3.0 * radial + 0.4 * radial * np.cos(2.0 * theta)
    count = np.broadcast_to(np.linspace(2.0, 5.0, 12), intensity.shape)
    channels = polar.angular_harmonic_channels(
        intensity,
        count,
        chi,
        max_mode=4,
    )
    expected = np.average(intensity, axis=0, weights=count)
    np.testing.assert_allclose(channels[0], expected, rtol=1e-12, atol=1e-12)


def test_masked_angular_sectors_are_excluded_instead_of_zero_filled() -> None:
    chi = np.linspace(-175.0, 175.0, 36)
    theta = np.deg2rad(chi)[:, None]
    radial = np.linspace(1.0, 1.5, 10)[None, :]
    intensity = 2.0 * radial + 0.6 * radial * np.cos(2.0 * theta)
    count = np.ones_like(intensity)
    count[[1, 5, 9, 13, 17], :] = 0.0
    intensity[count == 0.0] = np.nan
    channels = polar.angular_harmonic_channels(
        intensity,
        count,
        chi,
        max_mode=4,
    )
    expected_a0 = np.nansum(intensity * count, axis=0) / np.sum(count, axis=0)
    np.testing.assert_allclose(channels[0], expected_a0, atol=1e-12)
    np.testing.assert_allclose(channels[3], 0.6 * radial[0], rtol=1e-3)


def test_chi_permutation_breaks_harmonic_alignment() -> None:
    chi = np.linspace(-175.0, 175.0, 36)
    theta = np.deg2rad(chi)[:, None]
    radial = np.linspace(1.0, 2.0, 12)[None, :]
    intensity = 3.0 * radial + 0.7 * radial * np.cos(2.0 * theta)
    count = np.ones_like(intensity)
    aligned = polar.angular_harmonic_channels(
        intensity,
        count,
        chi,
        max_mode=4,
    )
    permuted = polar.angular_harmonic_channels(
        intensity[np.random.default_rng(11).permutation(len(chi))],
        count,
        chi,
        max_mode=4,
    )
    assert not np.allclose(
        polar._candidate_tensor(aligned[None, ...]),
        polar._candidate_tensor(permuted[None, ...]),
    )


def test_rank_deficient_angular_support_fails() -> None:
    intensity = np.ones((36, 8))
    count = np.zeros_like(intensity)
    count[:3] = 1.0
    with pytest.raises(polar.PolarBasisExperimentError, match="insufficient angular"):
        polar.angular_harmonic_channels(
            intensity,
            count,
            np.linspace(-175.0, 175.0, 36),
            max_mode=4,
        )


def test_twelve_sector_harmonic_fit_has_full_rank() -> None:
    chi = np.linspace(-165.0, 165.0, 12)
    theta = np.deg2rad(chi)[:, None]
    intensity = 2.0 + 0.3 * np.cos(2.0 * theta) + 0.1 * np.sin(4.0 * theta)
    intensity = np.broadcast_to(intensity, (12, 7))
    channels, qc = polar.angular_harmonic_channels(
        intensity,
        np.ones_like(intensity),
        chi,
        max_mode=4,
        return_qc=True,
    )
    assert channels.shape == (9, 7)
    assert qc["harmonic_min_valid_sectors"] == 12
    assert qc["harmonic_min_rank"] == 4
    assert qc["harmonic_max_angular_gap_degrees"] == pytest.approx(30.0)
    np.testing.assert_allclose(channels[0], 2.0)


def test_twelve_sector_rank_deficiency_fails_closed() -> None:
    intensity = np.ones((12, 5))
    with pytest.raises(polar.PolarBasisExperimentError, match="rank-deficient"):
        polar.angular_harmonic_channels(
            intensity,
            np.ones_like(intensity),
            np.zeros(12),
            max_mode=4,
        )


def test_partial_angular_coverage_is_recorded_without_interpolation() -> None:
    chi = np.linspace(-165.0, 165.0, 12)
    intensity = np.ones((12, 5))
    count = np.ones_like(intensity)
    count[3:6] = 0.0
    channels, qc = polar.angular_harmonic_channels(
        intensity,
        count,
        chi,
        max_mode=4,
        return_qc=True,
    )
    assert np.isfinite(channels).all()
    assert qc["harmonic_max_angular_gap_degrees"] == pytest.approx(120.0)


@pytest.mark.parametrize("mode_set", polar.MODE_SETS)
@pytest.mark.parametrize(
    "coefficients_per_channel", polar.COEFFICIENTS_PER_CHANNEL
)
def test_fold_encoder_has_exact_budget_and_finite_reconstruction(
    mode_set: str,
    coefficients_per_channel: int,
) -> None:
    rng = np.random.default_rng(17)
    harmonics = rng.normal(size=(80, 9, 64))
    specification = polar.RepresentationSpec(
        mode_set=mode_set,
        coefficients_per_channel=coefficients_per_channel,
    )
    encoder = polar.PolarBasisEncoder(
        spec=specification,
        q=np.linspace(2.0, 23.0, 64),
        seed=19,
    ).fit(harmonics[:60])
    coefficients = encoder.transform(harmonics[60:])
    reconstructed = encoder.inverse_transform(coefficients)
    assert coefficients.shape == (20, specification.budget)
    assert reconstructed.shape == (20, len(specification.modes), 64)
    assert np.isfinite(reconstructed).all()


def test_nested_mode_sets_preserve_identical_a0_coefficients() -> None:
    rng = np.random.default_rng(29)
    harmonics = rng.normal(size=(24, 9, 64))
    coefficients = {}
    for mode_set in polar.MODE_SETS:
        encoder = polar.PolarBasisEncoder(
            spec=polar.RepresentationSpec(
                mode_set=mode_set,
                coefficients_per_channel=12,
            ),
            q=np.linspace(2.0, 23.0, 64),
            seed=31,
        ).fit(harmonics[:16])
        coefficients[mode_set] = encoder.transform(harmonics[16:])[:, :12]
    np.testing.assert_array_equal(coefficients["A0"], coefficients["A0+A2"])
    np.testing.assert_array_equal(coefficients["A0"], coefficients["A0+A2+A4"])


def test_mlflow_mode_key_contains_no_plus_sign() -> None:
    mode_key = polar._mlflow_mode_key("A0+A2+A4")
    assert mode_key == "A0_plus_A2_plus_A4"


def test_polar_cache_reuses_unique_measurements(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_integration(row, **kwargs):
        calls.append(kwargs)
        assert row["interpolation_q_range"] == (2.0, 23.0)
        assert row["azimuthal_range"] == (-180.0, 180.0)
        return SimpleNamespace(
            intensity=np.ones((36, 256)),
            count=np.ones((36, 256)),
            sigma=np.full((36, 256), 0.1),
            q=np.linspace(2.0 + 21.0 / 512.0, 23.0 - 21.0 / 512.0, 256),
            azimuth=np.linspace(-175.0, 175.0, 36),
        )

    monkeypatch.setattr(polar, "perform_polar_cake_integration", fake_integration)
    rows = pd.DataFrame(
        {
            "measurement_key": ["one", "two"],
            "patientId": ["p1", "p2"],
            polar.TARGET_CASE_ID: ["p1::LEFT", "p2::RIGHT"],
            "_label": [0, 1],
            polar.RAW_FRAME_COLUMN: [np.ones((2, 2)), np.ones((2, 2))],
            polar.MASK_COLUMN: [np.zeros((2, 2), bool), np.zeros((2, 2), bool)],
            "ponifile": ["a.poni", "a.poni"],
            "sample_thickness_mm": [40.0, 41.0],
            "calibrant_thickness_mm": [40.0, 40.0],
            "interpolation_q_range": [(3.0, 12.0), (4.0, 15.0)],
            "azimuthal_range": [(-90.0, 90.0), (-120.0, 120.0)],
        }
    )
    kwargs = {
        "cache_folder": tmp_path,
        "dataset_sha256": "a" * 64,
        "n_q": 256,
        "n_chi": 36,
        "force_rebuild": False,
    }
    first, _ = polar.build_or_reuse_polar_cakes(rows, **kwargs)
    second, _ = polar.build_or_reuse_polar_cakes(rows, **kwargs)
    assert len(first) == len(second) == 2
    assert len(calls) == 2
    assert first["measurement_key"].is_unique
    assert first["axis_contract_sha256"].nunique() == 1
    axis_contract = first["axis_contract_sha256"].iloc[0]
    namespace = polar._polar_cache_namespace(
        tmp_path,
        n_q=256,
        n_chi=36,
        axis_contract=axis_contract,
    )
    (namespace / "polar_cake_cache_manifest.csv").unlink()
    recovered, _ = polar.build_or_reuse_polar_cakes(rows, **kwargs)
    assert len(recovered) == 2
    assert len(calls) == 2


def test_polar_cache_paths_are_separated_by_n_chi(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_integration(row, **kwargs):
        calls.append(kwargs["npt_azimuthal"])
        n_chi = int(kwargs["npt_azimuthal"])
        return SimpleNamespace(
            intensity=np.ones((n_chi, 256)),
            count=np.ones((n_chi, 256)),
            sigma=np.full((n_chi, 256), 0.1),
            q=np.linspace(2.0 + 21.0 / 512.0, 23.0 - 21.0 / 512.0, 256),
            azimuth=np.linspace(
                -180.0 + 180.0 / n_chi,
                180.0 - 180.0 / n_chi,
                n_chi,
            ),
        )

    monkeypatch.setattr(polar, "perform_polar_cake_integration", fake_integration)
    rows = pd.DataFrame(
        {
            "measurement_key": ["same"],
            "patientId": ["p1"],
            polar.TARGET_CASE_ID: ["p1::LEFT"],
            "_label": [0],
            polar.RAW_FRAME_COLUMN: [np.ones((2, 2))],
            polar.MASK_COLUMN: [np.zeros((2, 2), bool)],
            "ponifile": ["a.poni"],
            "sample_thickness_mm": [40.0],
            "calibrant_thickness_mm": [40.0],
        }
    )
    common = {
        "cache_folder": tmp_path,
        "dataset_sha256": "b" * 64,
        "n_q": 256,
        "force_rebuild": False,
    }
    manifest_12, _ = polar.build_or_reuse_polar_cakes(rows, n_chi=12, **common)
    manifest_72, _ = polar.build_or_reuse_polar_cakes(rows, n_chi=72, **common)
    assert calls == [12, 72]
    assert manifest_12["artifact"].iloc[0] != manifest_72["artifact"].iloc[0]
    assert (tmp_path / manifest_12["artifact"].iloc[0]).is_file()
    assert (tmp_path / manifest_72["artifact"].iloc[0]).is_file()


def test_axes_contract_rejects_different_grid() -> None:
    assert polar._axes_match_contract(
        np.linspace(2.0 + 21.0 / 512.0, 23.0 - 21.0 / 512.0, 256),
        np.linspace(-175.0, 175.0, 36),
        n_q=256,
        n_chi=36,
        radial_q_range=(2.0, 23.0),
        azimuthal_range=(-180.0, 180.0),
    )
    assert not polar._axes_match_contract(
        np.linspace(3.0, 12.0, 256),
        np.linspace(-175.0, 175.0, 36),
        n_q=256,
        n_chi=36,
        radial_q_range=(2.0, 23.0),
        azimuthal_range=(-180.0, 180.0),
    )


def test_embedded_poni_text_is_hashed_as_content() -> None:
    content = "poni_version: 2.1\n" + "Detector_config: x\n" * 500
    assert (
        polar._poni_fingerprint(content)
        == polar.hashlib.sha256(content.encode("utf-8")).hexdigest()
    )


def test_all_representations_reuse_one_patient_safe_manifest() -> None:
    dataframe = _synthetic_product_dataframe()
    model = _model_definition()
    context = polar._build_context(dataframe, model)
    target_rows = polar._target_measurement_rows(dataframe, model)
    rng = np.random.default_rng(23)
    target_rows["harmonic_matrix"] = [
        rng.normal(size=(9, 64)).astype(np.float32) for _ in range(len(target_rows))
    ]
    target_rows["qc_m1_energy"] = [
        polar._mode_energy(value, 1) for value in target_rows["harmonic_matrix"]
    ]
    target_rows["qc_m3_energy"] = [
        polar._mode_energy(value, 3) for value in target_rows["harmonic_matrix"]
    ]
    splits, manifest = polar._shared_patient_folds(context, folds=2, repeats=1, seed=42)
    result = polar.evaluate_representations(
        dataframe=dataframe,
        context=context,
        polar_rows=target_rows,
        axes=polar.PolarAxes(
            q=np.linspace(2.0, 23.0, 64),
            chi=np.linspace(-175.0, 175.0, 36),
        ),
        split_pairs=splits,
        fold_manifest=manifest,
        model_definition=model,
        model_info={"thresholds": {"threshold_target": 0.24665932038818544}},
        config={
            "evaluation": {
                "folds": 2,
                "seed": 42,
                "target_sensitivity": 0.95,
                "threshold_policy": "training_fold_target_sensitivity",
            },
            "runtime": {"reconstruction_examples_per_variant": 1},
        },
    )
    assert len(result["summary"]) == 9
    assert len(result["raw100_summary"]) == 1
    assert result["raw100_summary"].iloc[0]["representation"] == "raw100"
    assert len(result["polar_to_raw100"]) == 9
    assert {
        "raw100_roc_auc_mean",
        "delta_roc_auc_mean",
        "raw100_sensitivity_mean",
        "delta_sensitivity_mean",
    }.issubset(result["polar_to_raw100"].columns)
    assert set(result["summary"]["mode_set"]) == set(polar.MODE_SETS)
    assert set(result["summary"]["coefficients_per_channel"]) == {
        8,
        12,
        16,
    }
    assert set(result["summary"]["n_chi"]) == {36}
    assert len(result["bases"]) == 18
    held_out = manifest[manifest["partition"] == "test"]
    assert held_out.groupby("patientId")["fold_id"].nunique().eq(1).all()
    case_sets = [
        set(group[polar.TARGET_CASE_ID])
        for _, group in result["predictions"].groupby(
            list(polar.POLAR_VARIANT_COLUMNS), sort=False
        )
    ]
    assert all(values == case_sets[0] for values in case_sets[1:])
    assert set(result["raw100_predictions"][polar.TARGET_CASE_ID]) == case_sets[0]
    for split_id, (train_index, _) in enumerate(splits):
        expected_patients = set(context.iloc[train_index]["patientId"].astype(str))
        expected_fingerprint = polar._text_fingerprint(expected_patients)
        basis_keys = [
            f"fourier_bspline_A0_A2_A4_k{coefficients}/split_{split_id:03d}"
            for coefficients in polar.COEFFICIENTS_PER_CHANNEL
        ]
        assert all(
            result["basis_metadata"][key]["training_patient_fingerprint"]
            == expected_fingerprint
            for key in basis_keys
        )


def test_inner_oof_lr1_has_no_patient_leakage() -> None:
    dataframe = _synthetic_product_dataframe()
    model = _model_definition()
    context = polar._build_context(dataframe, model)
    train = polar._target_measurement_rows(dataframe, model)
    train["polar_coefficients"] = list(
        polar.profile_matrix(train, model["profile_column"])
    )
    patients = set(context["patientId"].astype(str))
    scores, manifest = polar._inner_oof_lr1_scores(
        dataframe=dataframe,
        context=context,
        train=train,
        train_patients=patients,
        model_definition=model,
        seed=43,
    )
    for _, fold in manifest.groupby("inner_fold_id"):
        fold_train = set(fold.loc[fold["partition"] == "train", "patientId"])
        fold_test = set(fold.loc[fold["partition"] == "validation", "patientId"])
        assert fold_train.isdisjoint(fold_test)
    held_out = manifest[manifest["partition"] == "validation"]
    assert held_out["patientId"].value_counts().eq(1).all()
    assert set(scores[polar.TARGET_CASE_ID]) == set(context[polar.TARGET_CASE_ID])


def test_shared_folds_keep_mixed_label_bilateral_cases_together() -> None:
    records = []
    for patient_index in range(20):
        patient_id = f"P{patient_index:02d}"
        labels = (0, 1) if patient_index == 0 else (patient_index % 2,)
        for case_index, label in enumerate(labels):
            records.append(
                {
                    "patientId": patient_id,
                    polar.TARGET_CASE_ID: f"{patient_id}::{case_index}",
                    "label": label,
                }
            )
    context = pd.DataFrame(records)
    _, manifest = polar._shared_patient_folds(
        context,
        folds=2,
        repeats=2,
        seed=47,
    )
    mixed = manifest[manifest["patientId"] == "P00"]
    assert mixed.groupby("split_id")["partition"].nunique().eq(1).all()
    held_out = mixed[mixed["partition"] == "test"]
    assert held_out.groupby("repeat_id")["fold_id"].nunique().eq(1).all()
    assert held_out.groupby("repeat_id")[polar.TARGET_CASE_ID].nunique().eq(2).all()


def _model_definition() -> dict[str, object]:
    return {
        "profile_column": "radial_profile_data",
        "label_column": "product_status_group",
        "group_column": "patientId",
        "specimen_column": "specimenId",
        "side_column": "side",
        "q_column": "q_range",
        "age_column": "age",
        "biopsy_column": "biopsy",
        "lr1_row_policy": "biopsy_only",
        "lr1_logreg_c": 0.1,
        "lr2_logreg_c": 0.3,
    }


def _synthetic_product_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(101)
    q = np.linspace(2.0, 23.0, 100)
    rows = []
    for patient_index in range(120):
        label = "CANCER" if patient_index % 2 else "BENIGN"
        patient = f"P{patient_index:03d}"
        for side in ("Left", "Right"):
            for position in range(3):
                target = side == "Left"
                signal = 0.25 * (label == "CANCER") * np.exp(-(((q - 13.0) / 2.0) ** 2))
                rows.append(
                    {
                        "patientId": patient,
                        "specimenId": f"{patient}-{side}",
                        "side": side,
                        "position": f"P{position + 1}",
                        "product_status_group": label if target else "NORMAL",
                        "biopsy": target,
                        "age": 40 + patient_index,
                        "q_range": q.copy(),
                        "radial_profile_data": 1.0
                        + signal
                        + rng.normal(0.0, 0.01, len(q)),
                        "sample_thickness_mm": 35.0 + patient_index % 8,
                        "calibration_session_uid": f"S{patient_index % 4}",
                        "started_at": f"2026-01-{(patient_index % 28) + 1:02d}T10:00:00Z",
                    }
                )
    return pd.DataFrame(rows)
