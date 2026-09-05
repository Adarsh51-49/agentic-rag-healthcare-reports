"""Combine the three frozen Stage 6.2 blinded-judge evidence archives."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "granite": (
        ROOT / "private_inputs/Adarsh_Konderu_Stage_6_2_Granite_Judge_Evidence.zip",
        "stage_6_2_granite_judge_evidence/",
    ),
    "mistral": (
        ROOT / "private_inputs/Adarsh_Konderu_Stage_6_2_Mistral_Judge_Evidence.zip",
        "stage_6_2_mistral_judge_evidence/",
    ),
    "phi": (
        ROOT / "Adarsh_Konderu_Stage_6_2_Phi_Judge_Evidence_CORRECTED.zip",
        "stage_6_2_phi_judge_evidence_corrected/",
    ),
}
OUTPUT = ROOT / "Adarsh_Konderu_Stage_6_2_Combined_Three_Judge_Evidence.zip"
PREFIX = "stage_6_2_combined_three_judge_evidence/"
CRITERIA = [
    "factual_support",
    "relevance",
    "coherence",
    "completeness",
    "responsible_framing",
    "criterion_mean",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csv_bytes(rows: list[dict], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def pearson(x: list[float], y: list[float]) -> float:
    mx, my = mean(x), mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    )
    return numerator / denominator if denominator else float("nan")


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = ((position + 1) + end) / 2
        for cursor in range(position, end):
            ranks[order[cursor]] = average_rank
        position = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(average_ranks(x), average_ranks(y))


def icc_two_way_random_absolute(matrix: list[list[float]]) -> dict[str, float]:
    """Return ICC(2,1) and ICC(2,k) for targets x judges."""
    n = len(matrix)
    k = len(matrix[0])
    grand = mean([value for row in matrix for value in row])
    row_means = [mean(row) for row in matrix]
    column_means = [mean([row[column] for row in matrix]) for column in range(k)]
    ss_rows = k * sum((value - grand) ** 2 for value in row_means)
    ss_columns = n * sum((value - grand) ** 2 for value in column_means)
    ss_total = sum((value - grand) ** 2 for row in matrix for value in row)
    ss_error = ss_total - ss_rows - ss_columns
    ms_rows = ss_rows / (n - 1)
    ms_columns = ss_columns / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    single_denominator = (
        ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    )
    average_denominator = ms_rows + (ms_columns - ms_error) / n
    return {
        "icc_2_1_single_absolute": (ms_rows - ms_error) / single_denominator,
        "icc_2_k_average_absolute": (ms_rows - ms_error) / average_denominator,
        "targets": n,
        "judges": k,
    }


def friedman_three_conditions(question_means: dict[str, dict[str, float]]) -> dict[str, float]:
    rank_sums = {condition: 0.0 for condition in "ABC"}
    for values in question_means.values():
        condition_values = [values[condition] for condition in "ABC"]
        ranks = average_ranks(condition_values)
        for condition, rank in zip("ABC", ranks):
            rank_sums[condition] += rank
    n = len(question_means)
    k = 3
    statistic = 12 / (n * k * (k + 1)) * sum(value**2 for value in rank_sums.values()) - 3 * n * (k + 1)
    # Chi-square survival function is exp(-x/2) when df = 2.
    return {
        "n_questions": n,
        "conditions": k,
        "chi_square": statistic,
        "degrees_of_freedom": 2,
        "asymptotic_p": math.exp(-statistic / 2),
        "rank_sums": rank_sums,
    }


def main() -> None:
    all_rows: list[dict] = []
    source_metadata: dict[str, dict] = {}
    expected_input_hash = None
    expected_rubric_hash = None

    for judge, (path, prefix) in SOURCES.items():
        with zipfile.ZipFile(path) as archive:
            rows = list(
                csv.DictReader(io.StringIO(archive.read(prefix + "judge_scores.csv").decode("utf-8")))
            )
            trace = json.loads(archive.read(prefix + "judge_trace.json"))
        if len(rows) != 45:
            raise AssertionError(f"{judge}: expected 45 score rows, got {len(rows)}")
        if {row["condition"] for row in rows} != {"A", "B", "C"}:
            raise AssertionError(f"{judge}: missing condition")
        if expected_input_hash is None:
            expected_input_hash = trace["input_archive_sha256"]
            expected_rubric_hash = trace["rubric_sha256"]
        if trace["input_archive_sha256"] != expected_input_hash:
            raise AssertionError(f"{judge}: frozen input mismatch")
        if trace["rubric_sha256"] != expected_rubric_hash:
            raise AssertionError(f"{judge}: rubric mismatch")
        for row in rows:
            for criterion in CRITERIA:
                row[criterion] = float(row[criterion])
            all_rows.append(row)
        source_metadata[judge] = {
            "archive": path.name,
            "archive_sha256": sha256(path.read_bytes()),
            "model_repo": trace["model_repo"],
            "model_revision": trace["model_revision"],
            "score_rows": len(rows),
        }

    if len(all_rows) != 135:
        raise AssertionError(f"Expected 135 ratings, got {len(all_rows)}")

    judge_condition_rows = []
    for judge in SOURCES:
        for condition in "ABC":
            subset = [row for row in all_rows if row["judge_id"] == judge and row["condition"] == condition]
            result = {"judge_id": judge, "condition": condition, "n": len(subset)}
            result.update({criterion: round(mean([row[criterion] for row in subset]), 4) for criterion in CRITERIA})
            judge_condition_rows.append(result)

    combined_condition_rows = []
    for condition in "ABC":
        subset = [row for row in all_rows if row["condition"] == condition]
        result = {"condition": condition, "n_ratings": len(subset)}
        for criterion in CRITERIA:
            values = [row[criterion] for row in subset]
            result[criterion + "_mean"] = round(mean(values), 4)
            result[criterion + "_sd"] = round(statistics.stdev(values), 4)
        combined_condition_rows.append(result)

    keys = sorted({(row["question_id"], row["condition"]) for row in all_rows})
    by_key_judge = {
        (row["question_id"], row["condition"], row["judge_id"]): row["criterion_mean"]
        for row in all_rows
    }
    matrix = [[by_key_judge[(question, condition, judge)] for judge in SOURCES] for question, condition in keys]
    agreement = icc_two_way_random_absolute(matrix)
    agreement["pairwise_correlations"] = {}
    for first, second in (("granite", "mistral"), ("granite", "phi"), ("mistral", "phi")):
        x = [by_key_judge[(question, condition, first)] for question, condition in keys]
        y = [by_key_judge[(question, condition, second)] for question, condition in keys]
        agreement["pairwise_correlations"][first + "_vs_" + second] = {
            "pearson": pearson(x, y),
            "spearman": spearman(x, y),
        }

    question_condition_means: dict[str, dict[str, float]] = {}
    for question in sorted({row["question_id"] for row in all_rows}):
        question_condition_means[question] = {}
        for condition in "ABC":
            values = [
                row["criterion_mean"]
                for row in all_rows
                if row["question_id"] == question and row["condition"] == condition
            ]
            question_condition_means[question][condition] = mean(values)
    condition_test = friedman_three_conditions(question_condition_means)

    all_fields = [
        "judge_id", "question_id", "anonymous_label", "condition",
        "factual_support", "relevance", "coherence", "completeness",
        "responsible_framing", "criterion_mean", "brief_reason", "parse_status",
    ]
    judge_summary_fields = ["judge_id", "condition", "n", *CRITERIA]
    combined_fields = ["condition", "n_ratings"] + [
        suffix for criterion in CRITERIA for suffix in (criterion + "_mean", criterion + "_sd")
    ]

    winner = max(combined_condition_rows, key=lambda row: row["criterion_mean_mean"])["condition"]
    report_lines = [
        "# Stage 6.2 Combined Three-Judge Analysis",
        "",
        "## Dataset integrity",
        "",
        "All three judges used the same frozen input and rubric. The analysis contains 135 ratings: 45 reports rated by each of three independent, blinded judges.",
        "",
        "## Combined criterion means (1–5)",
        "",
        "| Condition | Factual support | Relevance | Coherence | Completeness | Responsible framing | Overall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined_condition_rows:
        report_lines.append(
            f"| {row['condition']} | {row['factual_support_mean']:.3f} | {row['relevance_mean']:.3f} | "
            f"{row['coherence_mean']:.3f} | {row['completeness_mean']:.3f} | "
            f"{row['responsible_framing_mean']:.3f} | {row['criterion_mean_mean']:.3f} |"
        )
    report_lines += [
        "",
        "## Interpretation",
        "",
        f"Condition {winner} has the highest pooled descriptive mean. However, the differences are small and the judges do not agree strongly on the ordering. Therefore, the result should be described as descriptive rather than as proof that one condition is universally superior.",
        "",
        f"Absolute-agreement ICC(2,1): {agreement['icc_2_1_single_absolute']:.3f}.",
        f"Average-measures ICC(2,3): {agreement['icc_2_k_average_absolute']:.3f}.",
        f"Question-level Friedman test: chi-square({condition_test['degrees_of_freedom']}) = {condition_test['chi_square']:.3f}, p = {condition_test['asymptotic_p']:.4f}.",
        "",
        "Granite displayed a strong ceiling effect; Mistral preferred B; Phi preferred A. This disagreement is itself an important evaluation finding and supports triangulating LLM judgments with the deterministic Stage 6.1 metrics rather than relying on one judge.",
        "",
        "## Phi parser recovery",
        "",
        "Ten Phi responses contained a complete valid JSON object followed by extra commentary. Their scores were recovered verbatim with `JSONDecoder.raw_decode`; no model rerun, score editing, or imputation occurred. The original raw responses and source archive checksum are retained in the corrected Phi evidence package.",
        "",
    ]

    trace = {
        "stage": "Stage 6.2 combined independent blinded LLM judge analysis",
        "frozen_input_sha256": expected_input_hash,
        "rubric_sha256": expected_rubric_hash,
        "judges": list(SOURCES),
        "source_archives": source_metadata,
        "questions": 15,
        "reports_per_judge": 45,
        "total_ratings": 135,
        "criteria": CRITERIA[:-1],
        "combined_descriptive_winner": winner,
    }

    files = {
        "all_judge_scores.csv": csv_bytes(all_rows, all_fields),
        "judge_condition_summary.csv": csv_bytes(judge_condition_rows, judge_summary_fields),
        "combined_condition_summary.csv": csv_bytes(combined_condition_rows, combined_fields),
        "agreement_summary.json": (json.dumps(agreement, indent=2) + "\n").encode(),
        "condition_test_summary.json": (json.dumps(condition_test, indent=2) + "\n").encode(),
        "combined_analysis_trace.json": (json.dumps(trace, indent=2) + "\n").encode(),
        "STAGE_6_2_COMBINED_ANALYSIS.md": ("\n".join(report_lines) + "\n").encode(),
    }
    checksums = {name: sha256(data) for name, data in files.items()}
    files["checksums_sha256.json"] = (json.dumps(checksums, indent=2, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(PREFIX + name, data)

    print(OUTPUT)
    print(json.dumps(combined_condition_rows, indent=2))
    print(json.dumps(agreement, indent=2))
    print(json.dumps(condition_test, indent=2))
    print("ZIP_SHA256", sha256(OUTPUT.read_bytes()))


if __name__ == "__main__":
    main()
