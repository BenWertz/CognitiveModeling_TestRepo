import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "All_Data_Visual.csv"
CONDITIONS = ["silence", "lofi", "music"]


def rad(x):
    return math.radians(float(x) % 360)


def make_stan_data(max_subjects=None, frame_radius=None):
    df = pd.read_csv(CSV_PATH, sep=" ", quotechar='"', skipinitialspace=True)
    df["anomID"] = df["anomID"].astype(str)
    df["block"] = df["block"].astype(str)

    subjects = sorted(df["anomID"].unique())
    if max_subjects:
        subjects = subjects[:max_subjects]
        df = df[df["anomID"].isin(subjects)]

    subject_id = {name: i + 1 for i, name in enumerate(subjects)}
    condition_id = {name: i + 1 for i, name in enumerate(CONDITIONS)}

    data = {
        "N": 0,
        "S": len(subjects),
        "K": len(CONDITIONS),
        "J": 19,
        "subj": [],
        "cond": [],
        "response": [],
        "target_mu": [],
        "swap_mu": [],
    }

    if frame_radius is not None:
        data["swap_in_frame"] = []
        data["swap_frame_count"] = []

    for (subj, cond, _list), group in df.groupby(["anomID", "block", "list"]):
        group = group.sort_values("outputOrder").reset_index(drop=True)
        if len(group) != 20:
            raise ValueError(f"Expected 20 trials for one list, got {len(group)}")

        targets = [rad(x) for x in group["targetColor"]]

        for i, row in group.iterrows():
            swaps = [j for j in range(20) if j != i]
            data["subj"].append(subject_id[subj])
            data["cond"].append(condition_id[cond])
            data["response"].append(rad(row["response"]))
            data["target_mu"].append(targets[i])
            data["swap_mu"].append([targets[j] for j in swaps])

            if frame_radius is not None:
                in_frame = [1 if abs(i - j) <= frame_radius else 0 for j in swaps]
                data["swap_in_frame"].append(in_frame)
                data["swap_frame_count"].append(sum(in_frame))

    data["N"] = len(data["response"])
    lookup = {
        "conditions": {str(i + 1): name for i, name in enumerate(CONDITIONS)},
        "subjects": {str(i + 1): name for i, name in enumerate(subjects)},
    }
    return data, lookup


def save_json(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, indent=2), encoding="utf-8")


def summarize_draws(draws, lookup, stems, out_csv):
    rows = []
    for stem in stems:
        for number, condition in lookup["conditions"].items():
            col = f"{stem}[{number}]"
            if col not in draws:
                col = f"{stem}.{number}"
            if col not in draws:
                continue

            x = draws[col]
            rows.append(
                {
                    "parameter": stem,
                    "condition": condition,
                    "mean": x.mean(),
                    "sd": x.std(),
                    "q05": x.quantile(0.05),
                    "q50": x.quantile(0.50),
                    "q95": x.quantile(0.95),
                }
            )

    summary = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    return summary
