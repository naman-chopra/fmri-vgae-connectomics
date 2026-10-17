"""Cross-reference subject groups with the archived HCP metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=Path, default=Path("data/processed/subject-groups-original-style"))
    parser.add_argument("--metadata", type=Path, default=Path("original-data/HCP_metadata.csv"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    subjects = json.loads((args.groups / "subject_ids.json").read_text())
    labels = pd.Series(__import__("numpy").load(args.groups / "subject_labels.npy"), index=subjects, name="group")
    metadata = pd.read_csv(args.metadata, dtype={"Subject": str}).set_index("Subject")
    metadata.index = metadata.index.str.strip()
    joined = metadata.join(labels, how="inner")
    missing_metadata = sorted(set(subjects) - set(joined.index))

    report = {
        "group_artifacts": str(args.groups),
        "subjects_in_groups": len(subjects),
        "subjects_matched_to_metadata": len(joined),
        "subjects_missing_metadata": missing_metadata,
        "group_sizes": {str(k): int(v) for k, v in joined["group"].value_counts().sort_index().items()},
        "gender_by_group": {},
        "age_by_group": {},
    }
    if "Gender" in joined:
        table = pd.crosstab(joined["group"], joined["Gender"])
        report["gender_by_group"] = {str(k): {str(c): int(v) for c, v in row.items()} for k, row in table.to_dict(orient="index").items()}
        if table.shape[0] > 1 and table.shape[1] > 1:
            report["gender_chi2_pvalue"] = float(chi2_contingency(table)[1])
    if "Age" in joined:
        table = pd.crosstab(joined["group"], joined["Age"])
        report["age_bracket_by_group"] = {
            str(k): {str(c): int(v) for c, v in row.items()}
            for k, row in table.to_dict(orient="index").items()
        }
        if table.shape[0] > 1 and table.shape[1] > 1:
            report["age_bracket_chi2_pvalue"] = float(chi2_contingency(table)[1])
    target = args.output or args.groups / "demographic_report.json"
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
