from __future__ import annotations

from pathlib import Path

from xrd_preprocessing import (
    H5SessionFilter,
    calibrant_thickness_h5_filters,
    list_h5_sessions,
    load_preprocessing_config,
)
from xrd_preprocessing.transformers import (
    H5ToDataFrameTransformer,
    ProductColumnBuilder,
    ProductStatusGroupFilter,
)


DATA_DIR = Path(__file__).parent / "data"
REAL_H5_SUBSET = DATA_DIR / "aramis_real_h5_subset_20260128_5_patients.h5"
ARAMIS_CONFIG = (
    Path(__file__).parents[1]
    / "config"
    / "preprocessing"
    / "aramis_biopsy_patients_model_input_v0_1.yaml"
)


def test_real_h5_subset_uses_gfrm_reader_and_xrd_transformers():
    config = load_preprocessing_config(ARAMIS_CONFIG)
    config["filters"]["accepted_dates"] = ["2026-01-28"]
    calibrant_min_mm, calibrant_max_mm = config["filters"][
        "calibrant_thickness_range_mm"
    ]
    h5_filters = [
        H5SessionFilter(
            column="started_at",
            op="date in",
            values=config["filters"]["accepted_dates"],
        ),
        *calibrant_thickness_h5_filters(
            min_mm=calibrant_min_mm,
            max_mm=calibrant_max_mm,
        ),
    ]
    reader = H5ToDataFrameTransformer(
        data_preference=config["raw_data"]["source"],
        drop_missing_sample_thickness=config["filters"][
            "require_sample_thickness_mm"
        ],
        h5_filters=h5_filters,
        session_category="SAMPLE",
        set_category="SAMPLE",
    )
    raw_df = reader.fit_transform(REAL_H5_SUBSET)
    product_builder = ProductColumnBuilder(
        cancer_values=config["labels"]["product_column_builder"]["cancer_values"],
        benign_values=config["labels"]["product_column_builder"]["benign_values"],
        normal_values=config["labels"]["product_column_builder"]["normal_values"],
    )
    product_df = product_builder.fit_transform(raw_df)
    label_filter = ProductStatusGroupFilter(
        config["product_filter"]["product_status_group_keep"],
    )
    binary_df = label_filter.fit_transform(product_df)

    session_df = list_h5_sessions(REAL_H5_SUBSET)
    assert config["xrd_preprocessing"]["release_tag"] == "v0.1.7-beta"
    assert len(session_df) == 11
    assert session_df["category"].value_counts().to_dict() == {
        "SAMPLE": 10,
        "CALIBRATION": 1,
    }
    assert len(raw_df) == 30
    assert len(binary_df) == 30
    assert set(product_df["patientId"]) == {
        "Nova_376",
        "Nova_378",
        "Nova_379",
        "Nova_383",
        "Nova_384",
    }
    assert product_df["product_status_group"].value_counts().to_dict() == {
        "BENIGN": 24,
        "CANCER": 6,
    }
    assert set(binary_df["product_status_group"]) == {"BENIGN", "CANCER"}
    assert set(product_df["calibrant_thickness_mm"]) == {40.0}
    assert product_df["sample_thickness_mm"].notna().all()
    assert set(product_df["measurement_data_source"]) == {
        "embedded_raw_file_gfrm_to_photons"
    }
    assert product_df["measurement_data"].iloc[0].shape == (512, 768)
    assert reader.stats_["measurement_rows"] == 30
    assert label_filter.stats_["rows_fail"] == 0
