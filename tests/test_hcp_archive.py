from pathlib import Path

import pytest

from src.data.hcp_archive import (
    complete_subjects,
    matlab_subject_runs,
    numpy_subject_runs,
    validated_matlab_subjects,
)


DATA = Path(__file__).parents[1] / "original-data"


@pytest.mark.skipif(not (DATA / "HCP_rs_power.zip").exists(), reason="local HCP archive unavailable")
def test_matlab_archive_has_expected_complete_subjects():
    runs = matlab_subject_runs(DATA / "HCP_rs_power.zip")
    assert len(runs) == 880
    assert len(complete_subjects(runs)) == 830



@pytest.mark.skipif(not (DATA / "ts_power_s1200.zip").exists(), reason="local HCP archive unavailable")
def test_numpy_archive_has_expected_subject_coverage():
    runs = numpy_subject_runs(DATA / "ts_power_s1200.zip")
    assert len(runs) == 213
    assert sum(len(files) == 4 for files in runs.values()) == 185
