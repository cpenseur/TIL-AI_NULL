import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RETRIEVAL_ONLY_SCORE = 0.4
MAX_CANDIDATE_TOKEN_LENGTH = 64

DATA_JSONL = Path("/home/jupyter/advanced/nlp/nlp.jsonl")
PREDS_JSON = Path("/home/jupyter/TIL-AI_NULL/nlp_results.json")

# CHANGE THIS if your extracted folder is elsewhere
AE_MODEL_DIR = Path("/home/jupyter/TIL-AI_NULL/test/models/nlp_eval_512")

THRESHOLD = 0.9
MAX_LENGTH = 512


def _format_input(tokenizer, question: str, reference: str, candidate: str) -> str:
    candidate = candidate or ""
    # truncate candidate like the official script
    tokens = tokenizer.tokenize(
        candidate, max_length=MAX_CANDIDATE_TOKEN_LENGTH, truncation=True
    )
    reconstructed = tokenizer.convert_tokens_to_string(tokens)
    return f"Question: {question} Reference: {reference} Candidate: {reconstructed}"


@torch.no_grad()
def batch_ae_scores(tokenizer, model, rows, batch_size=64):
    device = next(model.parameters()).device

    empty_scores = {}
    to_model = []

    for i, row in enumerate(rows):
        docs = row["docs"]
        pred_docs = row["pred_docs"]
        q = row["question"]
        r = row["reference"]
        c = row["candidate"]

        overlap = len(set(docs).intersection(set(pred_docs))) >= 1

        # L4 perfect case
        if len(docs) == 0 and len(pred_docs) == 0 and r == "" and c == "":
            empty_scores[i] = 1.0
        # L5 case: overlap + empty string(s)
        elif (r == "" or c == "") and overlap:
            empty_scores[i] = 1.0 if (r == c) else RETRIEVAL_ONLY_SCORE
        # normal answerable with overlap -> model
        elif overlap:
            to_model.append((i, _format_input(tokenizer, q, r, c)))
        # retrieval failure
        else:
            empty_scores[i] = 0.0

    scores = [0.0] * len(rows)
    for i, s in empty_scores.items():
        scores[i] = s

    for start in trange(0, len(to_model), batch_size):
        batch = to_model[start : start + batch_size]
        idxs, texts = zip(*batch)
        enc = tokenizer(
            list(texts),
            max_length=MAX_LENGTH,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        probs = F.softmax(model(**enc).logits, dim=-1)
        for j, p in enumerate(probs):
            prob_eq = p[1].item()
            scores[idxs[j]] = 1.0 if (prob_eq >= THRESHOLD) else RETRIEVAL_ONLY_SCORE

    return scores


def main():
    gt = []
    with open(DATA_JSONL, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                gt.append(json.loads(line))

    preds = json.loads(PREDS_JSON.read_text())
    assert len(gt) == len(preds), f"len(gt)={len(gt)} len(preds)={len(preds)}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(AE_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(AE_MODEL_DIR)).to(device)
    model.eval()

    rows = []
    for p, g in zip(preds, gt):
        rows.append(
            {
                "docs": g.get("source_docs", []),
                "pred_docs": (p.get("documents") or [])[:3],
                "question": g["question"],
                "reference": g["answer"] if g.get("answer") is not None else "",
                "candidate": (p.get("answer") or ""),
                "difficulty": g.get("difficulty", "UNKNOWN"),
            }
        )

    scores = batch_ae_scores(tokenizer, model, rows, batch_size=64)

    by_diff = defaultdict(list)
    for s, r in zip(scores, rows):
        by_diff[r["difficulty"]].append(s)

    print(f"AE equiv_rate overall: {sum(scores)/len(scores):.4f} (n={len(scores)})")
    for d in sorted(by_diff.keys()):
        vals = by_diff[d]
        print(f"{d}: {sum(vals)/len(vals):.4f} (n={len(vals)})")


if __name__ == "__main__":
    main()