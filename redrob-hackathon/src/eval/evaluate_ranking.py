import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

RELEVANCE_THRESHOLD = 2  # >= this counts as "relevant" for precision


def load_ranked_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_labels(path: str) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["candidate_id"]: row for row in csv.DictReader(f)}


def precision_at_k(ranked_ids: list[str], labels: dict[str, dict], k: int) -> float:
    top_k = ranked_ids[:k]
    relevant = sum(
        1 for cid in top_k
        if int(labels.get(cid, {}).get("relevance_label", 0)) >= RELEVANCE_THRESHOLD
    )
    return relevant / min(k, len(top_k)) if top_k else 0.0


def mean_relevance_at_k(ranked_ids: list[str], labels: dict[str, dict], k: int) -> float:
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    scores = [int(labels.get(cid, {}).get("relevance_label", 0)) for cid in top_k]
    return sum(scores) / len(scores)


def dcg(relevances: list[int]) -> float:
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(ranked_ids: list[str], labels: dict[str, dict], k: int) -> float:
    """NDCG@k where IDCG is computed from the FULL labeled universe (not just the
    system's own top-k), so failing to retrieve a highly-relevant labeled candidate
    is correctly penalized rather than ignored."""
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    system_relevance = [int(labels.get(cid, {}).get("relevance_label", 0)) for cid in top_k]
    all_relevances = sorted((int(r["relevance_label"]) for r in labels.values()), reverse=True)
    ideal_relevance = all_relevances[:k]
    idcg = dcg(ideal_relevance)
    if idcg == 0:
        return 0.0
    return dcg(system_relevance) / idcg


def kendall_tau_on_labeled(ranked_ids: list[str], labels: dict[str, dict]) -> float:
    """Kendall's tau between the system's order and the label-sorted order,
    restricted to candidates that were actually labeled."""
    labeled_ranked = [cid for cid in ranked_ids if cid in labels]
    if len(labeled_ranked) < 2:
        return float("nan")
    system_rank = {cid: i for i, cid in enumerate(labeled_ranked)}
    # "pure label rank": sort labeled candidates by relevance desc, tie-break by candidate_id
    label_sorted = sorted(
        labeled_ranked,
        key=lambda cid: (-int(labels[cid]["relevance_label"]), cid),
    )
    label_rank = {cid: i for i, cid in enumerate(label_sorted)}
    ids = labeled_ranked
    tau, _ = kendalltau(
        [system_rank[cid] for cid in ids],
        [label_rank[cid] for cid in ids],
    )
    return float(tau)


def disqualifier_leakage(ranked_ids: list[str], labels: dict[str, dict], top_n: int = 50) -> int:
    top_n_ids = set(ranked_ids[:top_n])
    return sum(
        1 for cid, row in labels.items()
        if row.get("disqualifier_flag") == "true" and cid in top_n_ids
    )


def evaluate(variant_name: str, ranked_csv: str, labels_csv: str) -> dict:
    ranked_rows = load_ranked_csv(ranked_csv)
    ranked_ids = [r["candidate_id"] for r in ranked_rows]
    labels = load_labels(labels_csv)

    unlabeled_in_top50 = sum(1 for cid in ranked_ids[:50] if cid not in labels)

    metrics = {
        "variant": variant_name,
        "ndcg@10": round(ndcg_at_k(ranked_ids, labels, 10), 4),
        "ndcg@50": round(ndcg_at_k(ranked_ids, labels, 50), 4),
        "precision@10": round(precision_at_k(ranked_ids, labels, 10), 4),
        "precision@50": round(precision_at_k(ranked_ids, labels, 50), 4),
        "mean_relevance@10": round(mean_relevance_at_k(ranked_ids, labels, 10), 4),
        "mean_relevance@50": round(mean_relevance_at_k(ranked_ids, labels, 50), 4),
        "kendall_tau": round(kendall_tau_on_labeled(ranked_ids, labels), 4),
        "disqualifier_leakage_count": disqualifier_leakage(ranked_ids, labels, top_n=50),
        "unlabeled_in_top50": unlabeled_in_top50,
    }
    return metrics


def append_to_results(metrics: dict, results_csv: str) -> None:
    path = Path(results_csv)
    fieldnames = list(metrics.keys()) + ["notes"]
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({**metrics, "notes": ""})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a candidate-ranker output CSV against labeled ground truth")
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--ranked-csv", required=True)
    parser.add_argument("--labels-csv", default="data/eval/relevance_labels_v1.csv")
    parser.add_argument("--results-csv", default="data/eval/experiment_results.csv")
    args = parser.parse_args()

    metrics = evaluate(args.variant_name, args.ranked_csv, args.labels_csv)
    append_to_results(metrics, args.results_csv)

    print(f"\n=== {args.variant_name} ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
