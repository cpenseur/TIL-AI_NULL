import json
import math
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

# IMPORTANT: this import assumes nlp_manger.py is on PYTHONPATH
from nlp_manager import NLPManager


DATA_DIR = Path("/home/jupyter/advanced/nlp")
DOC_DIR = DATA_DIR / "documents"
JSONL_PATH = DATA_DIR / "nlp.jsonl"

OUT_DIR = Path("/home/jupyter/TIL-AI_NULL")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_documents():
    docs = []
    for p in sorted(DOC_DIR.glob("*.txt")):
        docs.append({"id": p.stem, "document": p.read_text()})
    return docs


def load_instances():
    instances = []
    with open(JSONL_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instances.append(json.loads(line))
    return instances


def reciprocal_rank(pred_docs, gold_docs):
    """MRR contribution for one query (1/rank of first relevant)."""
    gold = set(gold_docs)
    for i, d in enumerate(pred_docs, start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


def recall_at_k(pred_docs, gold_docs, k):
    gold = set(gold_docs)
    if not gold:
        return None  # undefined; skip
    return 1.0 if any(d in gold for d in pred_docs[:k]) else 0.0


def precision_at_k(pred_docs, gold_docs, k):
    gold = set(gold_docs)
    if k == 0:
        return None
    if not pred_docs:
        return 0.0
    top = pred_docs[:k]
    return sum(1 for d in top if d in gold) / float(k)


def is_unanswerable(diff):
    return diff in ("L4", "L5")


def classify_error(gt, pred):
    """
    Lightweight "what went wrong" categorization.
    (This does NOT use the AE model; it focuses on abstain + retrieval.)
    """
    diff = gt.get("difficulty")
    gold_docs = gt.get("source_docs", [])
    pred_docs = pred.get("documents", [])
    pred_ans = (pred.get("answer") or "").strip()

    # Unanswerables:
    if diff == "L4":
        if pred_ans == "" and len(pred_docs) == 0:
            return "ok"
        if pred_ans != "":
            return "L4_should_abstain_but_answered"
        return "L4_should_be_empty_docs_but_returned_docs"

    if diff == "L5":
        if pred_ans == "" and len(pred_docs) > 0:
            return "ok"
        if pred_ans != "":
            return "L5_should_abstain_but_answered"
        return "L5_should_return_docs_but_empty_docs"

    # Answerables:
    if pred_ans == "":
        # either retrieval too low, or model said NOT_FOUND, etc.
        return "answerable_abstained"

    # Retrieval overlap:
    if gold_docs:
        if not any(d in set(gold_docs) for d in pred_docs[:3]):
            return "retrieval_miss_top3"

    return "answer_mismatch_or_partial"  # needs AE evaluator to know for sure


def main():
    print("Loading corpus...")
    docs = load_documents()

    print("Loading dataset...")
    instances = load_instances()
    print(f"Loaded {len(docs)} documents and {len(instances)} questions")

    manager = NLPManager()
    t0 = time.perf_counter()
    manager.load_corpus(docs)
    t_load = time.perf_counter() - t0
    print(f"Corpus loaded in {t_load:.2f}s")

    preds = []
    latencies = []

    # Retrieval metrics accumulators (answerable only)
    mrr_sum = 0.0
    mrr_n = 0
    r_at = {1: 0.0, 3: 0.0, 5: 0.0}
    r_n = {1: 0, 3: 0, 5: 0}
    p_at = {1: 0.0, 3: 0.0, 5: 0.0}
    p_n = {1: 0, 3: 0, 5: 0}

    # difficulty buckets
    diff_counts = Counter()
    diff_correct = Counter()  # "format-correct" only here; see note below
    error_counts = Counter()

    # For “what was wrong” examples
    wrong_examples = defaultdict(list)  # error_type -> list of (key, question)

    print("Running QA...")
    for gt in instances:
        q = gt["question"]
        key = gt.get("key")
        diff = gt.get("difficulty", "UNKNOWN")

        t1 = time.perf_counter()
        pred = manager.qa(q)
        dt = time.perf_counter() - t1

        latencies.append(dt)
        preds.append(pred)

        diff_counts[diff] += 1

        # Retrieval metrics: only meaningful when gt has source_docs (usually answerable)
        gold_docs = gt.get("source_docs", [])
        if gold_docs:
            mrr_sum += reciprocal_rank(pred.get("documents", []), gold_docs)
            mrr_n += 1
            for k in (1, 3, 5):
                rv = recall_at_k(pred.get("documents", []), gold_docs, k)
                if rv is not None:
                    r_at[k] += rv
                    r_n[k] += 1
                pv = precision_at_k(pred.get("documents", []), gold_docs, k)
                if pv is not None:
                    p_at[k] += pv
                    p_n[k] += 1

        # Per-difficulty "format correctness" (NOT semantic answer correctness)
        err = classify_error(gt, pred)
        error_counts[err] += 1
        if err == "ok":
            diff_correct[diff] += 1
        else:
            if len(wrong_examples[err]) < 10:
                wrong_examples[err].append((key, q))

    # Save raw predictions
    out_path = OUT_DIR / "nlp_results.json"
    out_path.write_text(json.dumps(preds))
    print(f"Saved predictions to {out_path}")

    lat = np.array(latencies, dtype=np.float64)
    total = float(lat.sum())
    qps = len(lat) / total if total > 0 else 0.0

    print("\n=== SPEED ===")
    print(f"Questions: {len(lat)}")
    print(f"Total QA time: {total:.2f}s (excluding corpus load)")
    print(f"QPS: {qps:.2f}")
    print(f"Latency p50: {np.percentile(lat, 50):.3f}s")
    print(f"Latency p90: {np.percentile(lat, 90):.3f}s")
    print(f"Latency p99: {np.percentile(lat, 99):.3f}s")

    print("\n=== RETRIEVAL METRICS (on items with source_docs) ===")
    if mrr_n:
        print(f"MRR: {mrr_sum / mrr_n:.4f} (n={mrr_n})")
    for k in (1, 3, 5):
        if r_n[k]:
            print(f"Recall@{k}: {r_at[k] / r_n[k]:.4f} (n={r_n[k]})")
        if p_n[k]:
            print(f"Precision@{k}: {p_at[k] / p_n[k]:.4f} (n={p_n[k]})")

    print("\n=== PER-DIFFICULTY (format correctness only) ===")
    for diff in sorted(diff_counts.keys()):
        acc = diff_correct[diff] / diff_counts[diff] if diff_counts[diff] else 0.0
        print(f"{diff}: {acc:.3f} ({diff_correct[diff]}/{diff_counts[diff]})")

    print("\n=== ERROR BREAKDOWN ===")
    for k, v in error_counts.most_common():
        print(f"{k}: {v}")

    print("\n=== EXAMPLES (first 10 per error type) ===")
    for err, exs in wrong_examples.items():
        print(f"\n[{err}]")
        for key, q in exs:
            print(f"- key={key} q={q[:160]!r}")


if __name__ == "__main__":
    main()