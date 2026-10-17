"""Validate provenance and internal consistency of the local HCP archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.hcp_archive import (
    complete_subjects,
    load_matlab_correlation,
    load_matlab_timeseries,
    load_numpy_timeseries,
    matlab_correlation_runs,
    matlab_subject_runs,
    numpy_subject_runs,
    run_count_distribution,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("original-data"))
    parser.add_argument("--subject", default="100206")
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()

    mat_zip = args.data_dir / "HCP_rs_power.zip"
    npy_zip = args.data_dir / "ts_power_s1200.zip"
    ts_runs = matlab_subject_runs(mat_zip)
    corr_runs = matlab_correlation_runs(mat_zip)
    npy_runs = numpy_subject_runs(npy_zip)

    ts = load_matlab_timeseries(mat_zip, args.subject, args.run)
    stored = load_matlab_correlation(mat_zip, args.subject, args.run)
    recomputed = np.corrcoef(ts)
    off_diagonal = ~np.eye(264, dtype=bool)
    max_error = float(np.max(np.abs(recomputed[off_diagonal] - stored[off_diagonal])))

    sample_member = npy_runs[sorted(npy_runs)[0]][0]
    npy_sample = load_numpy_timeseries(npy_zip, sample_member)
    report = {
        "reference_archive": str(mat_zip),
        "secondary_archive": str(npy_zip),
        "matlab_timeseries_subjects": len(ts_runs),
        "matlab_complete_subjects": len(complete_subjects(ts_runs)),
        "matlab_timeseries_run_distribution": run_count_distribution(ts_runs),
        "matlab_correlation_subjects": len(corr_runs),
        "correlation_complete_subjects": len(complete_subjects(corr_runs)),
        "numpy_subjects": len(npy_runs),
        "numpy_complete_subjects": sum(len(files) == 4 for files in npy_runs.values()),
        "sample_subject": args.subject,
        "sample_ts_shape": list(ts.shape),
        "sample_correlation_shape": list(stored.shape),
        "sample_off_diagonal_correlation_max_error": max_error,
        "numpy_sample_member": sample_member,
        "numpy_sample_shape": list(npy_sample.shape),
    }
    print(json.dumps(report, indent=2))
    if max_error > 1e-5:
        raise SystemExit("Stored MATLAB correlation does not match its time series")


if __name__ == "__main__":
    main()
