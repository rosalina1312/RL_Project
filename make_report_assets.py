import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report_assets"

DR_TARGET_RESULTS = [
    ("PPO ADR", 0.260, -3.2035181522369385),
    ("PPO UDR [0.5, 5.0]", 0.185, -3.5629554986953735),
    ("PPO UDR [0.8, 3.0]", 0.160, -3.7178163528442383),
    ("SAC ADR 200k", 0.300, -3.065),
    ("SAC ADR 500k", 1.000, -0.407),
    ("SAC UDR 200k", 0.660, -1.7757229804992676),
    ("SAC UDR 500k", 1.000, -0.339),
]

DR_MASS_RESULTS = [
    ("PPO ADR", [0.14, 0.30, 0.30, 0.18, 0.18]),
    ("PPO UDR [0.5, 5.0]", [0.08, 0.14, 0.26, 0.20, 0.18]),
    ("PPO UDR [0.8, 3.0]", [0.16, 0.30, 0.16, 0.28, 0.16]),
    ("SAC ADR 200k", [0.30, 0.30, 0.30, 0.30, 0.30]),
    ("SAC ADR 500k", [0.98, 1.00, 1.00, 1.00, 1.00]),
    ("SAC UDR 200k", [0.64, 0.64, 0.64, 0.70, 0.66]),
    ("SAC UDR 500k", [1.00, 1.00, 1.00, 1.00, 1.00]),
]


def pct(value):
    if pd.isna(value):
        return ""
    return f"{100 * float(value):.1f}%"


def fmt(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def label_for(name):
    if "ppo_none_source_to_source" in name:
        return "PPO source to source"
    if "ppo_none_source_to_target" in name:
        return "PPO source to target"
    if "ppo_none_target_to_target" in name:
        return "PPO target to target"
    if "sac_none_source_to_source" in name:
        return "SAC source to source"
    if "sac_none_source_to_target" in name:
        return "SAC source to target"
    if "sac_none_target_to_target" in name:
        return "SAC target to target"
    if "ppo_udr" in name:
        return "PPO UDR"
    if "ppo_adr" in name:
        return "PPO ADR"
    if "sac_udr" in name:
        return "SAC UDR"
    if "sac_adr" in name:
        return "SAC ADR"
    if "udr_v2" in name:
        return "PPO UDR [0.8, 3.0]"
    if "udr_v3" in name:
        return "PPO UDR [0.5, 5.0]"
    if "adr_v2" in name:
        return "PPO ADR"
    return name


def save_table(df, title, path, font_size=9):
    rows = max(len(df), 1)
    fig_height = 1.2 + rows * 0.35
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=10)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.25)

    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#243b53")
        else:
            cell.set_facecolor("#f6f8fa" if row % 2 == 0 else "white")

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def copy_graphs():
    copies = {
        ROOT / "part1" / "part1_comparison_learning_curve.png": OUT / "part1_learning_curve.png",
        ROOT / "part2" / "results" / "domain_randomization_target_barplot.png": OUT / "part2_dr_target_success.png",
        ROOT / "part2" / "results" / "mean_return_barplot.png": OUT / "part2_dr_mean_return.png",
        ROOT / "part2" / "results" / "mass_robustness_curve.png": OUT / "part2_mass_robustness_curve.png",
        ROOT / "part2" / "results" / "lower_upper_success_barplot.png": OUT / "part2_lower_upper_success.png",
        ROOT / "renders" / "part2_sac_target_380k_seed51_start_finish.png": OUT / "part2_sac_target_380k_seed51_start_finish.png",
        ROOT / "renders" / "part2_sac_udr_source_200k_target_seed53_start_finish.png": OUT / "part2_sac_udr_source_200k_target_seed53_start_finish.png",
    }

    for src, dst in copies.items():
        if src.exists():
            shutil.copy2(src, dst)


def part1_table():
    csv_path = ROOT / "part1" / "part1_summary.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    table = pd.DataFrame(
        {
            "Experiment": df["experiment"],
            "Avg reward": df["avg_reward"].map(fmt),
            "Final avg 10": df["final_avg_10"].map(fmt),
            "Final avg 50": df["final_avg_50"].map(fmt),
            "Final x": df["final_avg_x_delta_50"].map(fmt),
            "Best x": df["best_x_delta"].map(fmt),
            "Time (s)": df["elapsed_time_sec"].map(lambda x: f"{float(x):.2f}"),
        }
    )
    save_table(table, "Part 1: Hopper training summary", OUT / "part1_summary_table.png")


def part2_tables():
    csv_path = ROOT / "part2" / "results" / "eval_results.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    df["success_rate"] = pd.to_numeric(df["success_rate"], errors="coerce")
    df["mean_return"] = pd.to_numeric(df["mean_return"], errors="coerce")
    df["eval_mass"] = pd.to_numeric(df["eval_mass"], errors="coerce")
    df["label"] = df["experiment_name"].map(label_for)

    lower_upper = df[
        df["experiment_name"].str.contains(
            "ppo_none_source_to_source|ppo_none_source_to_target|ppo_none_target_to_target|"
            "sac_none_source_to_source|sac_none_source_to_target|sac_none_target_to_target",
            regex=True,
        )
    ]
    lower_upper = (
        lower_upper.groupby("label", as_index=False)
        .agg(success_rate=("success_rate", "mean"), mean_return=("mean_return", "mean"))
    )
    lower_upper = pd.DataFrame(
        {
            "Experiment": lower_upper["label"],
            "Success": lower_upper["success_rate"].map(pct),
            "Mean return": lower_upper["mean_return"].map(fmt),
        }
    )
    save_table(lower_upper, "Part 2: lower and upper bounds", OUT / "part2_lower_upper_table.png")

    dr_table = pd.DataFrame(
        {
            "Method": [row[0] for row in DR_TARGET_RESULTS],
            "Target success": [pct(row[1]) for row in DR_TARGET_RESULTS],
            "Mean return": [fmt(row[2]) for row in DR_TARGET_RESULTS],
        }
    )
    save_table(dr_table, "Part 2: domain randomization on target domain", OUT / "part2_domain_randomization_table.png")

    mass_table = pd.DataFrame(
        {
            "Method": [row[0] for row in DR_MASS_RESULTS],
            "1 kg": [pct(row[1][0]) for row in DR_MASS_RESULTS],
            "2 kg": [pct(row[1][1]) for row in DR_MASS_RESULTS],
            "3 kg": [pct(row[1][2]) for row in DR_MASS_RESULTS],
            "4 kg": [pct(row[1][3]) for row in DR_MASS_RESULTS],
            "5 kg": [pct(row[1][4]) for row in DR_MASS_RESULTS],
        }
    )
    save_table(mass_table, "Part 2: mass robustness", OUT / "part2_mass_robustness_table.png")


def main():
    OUT.mkdir(exist_ok=True)
    part1_table()
    part2_tables()
    copy_graphs()

    print("Report assets written to:", OUT)


if __name__ == "__main__":
    main()
