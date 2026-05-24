"""Manages the NLP model."""

import logging
import re
import faiss
import torch
import numpy as np
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)


class NLPManager:

    loaded = False

    def __init__(self):

        MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

        # =====================================
        # DEVICE
        # =====================================
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"NLPManager using device: {self.device}")

        # =====================================
        # EMBEDDING MODEL — on GPU if available
        # =====================================
        self.embedding_model = SentenceTransformer(
            str(MODELS_DIR / "bge-base-en-v1.5"),
            device=self.device
        )

        # =====================================
        # RERANKER — on GPU if available
        # This is the #1 speed bottleneck; GPU makes it ~10x faster
        # =====================================
        self.reranker = CrossEncoder(
            str(MODELS_DIR / "bge-reranker-base"),
            device=self.device,
            max_length=512
        )

        # =====================================
        # TEXT SPLITTER
        # =====================================
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # =====================================
        # LLM
        # =====================================
        self.MODEL_NAME = str(MODELS_DIR / "Qwen2.5-1.5B-Instruct")

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.tokenizer.padding_side = "left"  # required for batch generation

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # =====================================
        # STORAGE
        # =====================================
        self.chunks = []
        self.chunk_to_doc = []
        self.chunk_embeddings = None
        self.faiss_index = None
        self.bm25 = None

        self._token_re = re.compile(r"[a-zA-Z0-9]+")

    # =====================================
    # TOKENIZER
    # =====================================
    def bm25_tokenize(self, text: str) -> list[str]:
        return self._token_re.findall(text.lower())

    # =====================================
    # LOAD CORPUS
    # =====================================
    def load_corpus(self, documents: list[dict[str, str]]) -> None:

        logger.info(f"load_corpus: received {len(documents)} documents")

        chunks = []
        chunk_to_doc = []

        for doc in documents:
            doc_id = doc["id"]
            text = doc["document"]
            for chunk in self.text_splitter.split_text(text):
                chunks.append(chunk)
                chunk_to_doc.append(doc_id)

        logger.info(f"load_corpus: {len(chunks)} total chunks")

        self.chunks = chunks
        self.chunk_to_doc = chunk_to_doc

        # =====================================
        # EMBEDDINGS
        # Large batch size on GPU; fall back to 32 on CPU
        # =====================================
        encode_batch = 256 if self.device == "cuda" else 32
        embeddings = self.embedding_model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=encode_batch
        ).astype("float32")

        self.chunk_embeddings = embeddings

        # =====================================
        # FAISS — GPU index if faiss-gpu is installed, else CPU
        # =====================================
        dim = embeddings.shape[1]
        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(embeddings)

        if self.device == "cuda" and hasattr(faiss, "StandardGpuResources"):
            try:
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                logger.info("FAISS: using GPU index")
            except Exception as e:
                logger.warning(f"FAISS GPU failed, falling back to CPU: {e}")
                self.faiss_index = cpu_index
        else:
            self.faiss_index = cpu_index
            logger.info("FAISS: using CPU index")

        # =====================================
        # BM25
        # =====================================
        tokenized_chunks = [self.bm25_tokenize(c) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        self.loaded = True
        logger.info("load_corpus: complete")

    # =====================================
    # HYBRID RETRIEVAL
    # =====================================
    def hybrid_retrieve(
        self,
        question: str,
        dense_k: int = 50,
        sparse_k: int = 50,
        top_k: int = 20,
        alpha: float = 0.65
    ) -> list[dict]:

        q_emb = self.embedding_model.encode(
            [question],
            normalize_embeddings=True
        ).astype("float32")

        D, I = self.faiss_index.search(q_emb, dense_k)

        dense_scores = {
            int(idx): float(score)
            for score, idx in zip(D[0], I[0])
            if idx >= 0
        }

        q_tok = self.bm25_tokenize(question)
        bm25_scores = self.bm25.get_scores(q_tok)

        if sparse_k >= len(bm25_scores):
            top_sparse_idx = np.argsort(bm25_scores)[::-1]
        else:
            top_sparse_idx = np.argpartition(bm25_scores, -sparse_k)[-sparse_k:]

        sparse_scores = {
            int(i): float(bm25_scores[i])
            for i in top_sparse_idx
        }

        candidates = sorted(set(dense_scores.keys()) | set(sparse_scores.keys()))

        if not candidates:
            return []

        svals = np.array(
            [sparse_scores.get(i, 0.0) for i in candidates], dtype="float32"
        )
        smin, smax = float(svals.min()), float(svals.max())

        def norm_sparse(v: float) -> float:
            return (v - smin) / (smax - smin + 1e-9)

        combined = []
        for idx in candidates:
            dense = dense_scores.get(idx, 0.0)
            sparse = norm_sparse(sparse_scores.get(idx, 0.0))
            combined.append({
                "chunk": self.chunks[idx],
                "doc_id": self.chunk_to_doc[idx],
                "score": float(alpha * dense + (1 - alpha) * sparse)
            })

        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]

    # =====================================
    # RERANK
    # =====================================
    def rerank(self, question: str, retrieved: list[dict], top_k: int = 5) -> list[dict]:

        if not retrieved:
            return []

        pairs = [[question, r["chunk"]] for r in retrieved]
        scores = self.reranker.predict(pairs, batch_size=64, show_progress_bar=False)

        reranked = sorted(
            [{**r, "rerank_score": float(s)} for r, s in zip(retrieved, scores)],
            key=lambda x: x["rerank_score"],
            reverse=True
        )
        return reranked[:top_k]

    # =====================================
    # PROMPT
    # =====================================
    def build_prompt(self, question: str, contexts: list[str]) -> str:
        context_text = "\n\n".join(
            [f"[{i+1}] {c}" for i, c in enumerate(contexts)]
        )
        return (
            "Answer the question using only the contexts below.\n"
            "Output the exact answer only, as short as possible.\n"
            "If the answer is not in the contexts, output: NOT_FOUND\n\n"
            f"{context_text}\n\n"
            f"Q: {question}\n"
            "A:"
        )

    # =====================================
    # QA BATCH — primary entry point
    # Call this from your server with the full batch of questions
    # instead of looping qa() one at a time
    # =====================================
    def qa_batch(self, questions: list[str]) -> list[dict]:
        """
        Full RAG pipeline over a batch of questions.
        Reranker and LLM are called in bulk — much faster than qa() in a loop.
        """
        # ── Stage 1: embed all questions at once ──────────────────────────
        q_embs = self.embedding_model.encode(
            questions,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False
        ).astype("float32")

        # ── Stage 2: FAISS + BM25 per question ────────────────────────────
        all_candidates = []   # list of list[dict] or []

        for q_idx, question in enumerate(questions):
            q_emb = q_embs[q_idx : q_idx + 1]
            D, I = self.faiss_index.search(q_emb, 50)

            dense_scores = {
                int(idx): float(score)
                for score, idx in zip(D[0], I[0])
                if idx >= 0
            }

            q_tok = self.bm25_tokenize(question)
            bm25_scores = self.bm25.get_scores(q_tok)
            top_sparse_idx = np.argpartition(bm25_scores, -50)[-50:] \
                if len(bm25_scores) >= 50 else np.argsort(bm25_scores)[::-1]
            sparse_scores = {int(i): float(bm25_scores[i]) for i in top_sparse_idx}

            candidates = sorted(set(dense_scores.keys()) | set(sparse_scores.keys()))
            if not candidates:
                all_candidates.append([])
                continue

            svals = np.array([sparse_scores.get(i, 0.0) for i in candidates], dtype="float32")
            smin, smax = float(svals.min()), float(svals.max())

            retrieved = []
            for idx in candidates:
                dense = dense_scores.get(idx, 0.0)
                sparse = (sparse_scores.get(idx, 0.0) - smin) / (smax - smin + 1e-9)
                retrieved.append({
                    "chunk": self.chunks[idx],
                    "doc_id": self.chunk_to_doc[idx],
                    "score": float(0.65 * dense + 0.35 * sparse)
                })

            retrieved.sort(key=lambda x: x["score"], reverse=True)
            all_candidates.append(retrieved[:20])

        # ── Stage 3: batch rerank — ONE reranker call for all questions ───
        # Flatten all pairs, call predict once, then re-slice
        all_pairs = []
        pair_map = []   # (q_idx, chunk_idx_within_question)

        for q_idx, candidates in enumerate(all_candidates):
            for c_idx, c in enumerate(candidates):
                all_pairs.append([questions[q_idx], c["chunk"]])
                pair_map.append((q_idx, c_idx))

        if not all_pairs:
            return [{"documents": [], "answer": ""} for _ in questions]

        # Single batched reranker call — the big speedup
        all_scores = self.reranker.predict(
            all_pairs,
            batch_size=256,          # large batch = GPU fully utilised
            show_progress_bar=False
        )

        # Re-attach scores and pick top 5 per question
        per_question_scored: list[list[dict]] = [[] for _ in questions]
        for (q_idx, c_idx), score in zip(pair_map, all_scores):
            entry = {**all_candidates[q_idx][c_idx], "rerank_score": float(score)}
            per_question_scored[q_idx].append(entry)

        reranked_all = []
        for scored in per_question_scored:
            scored.sort(key=lambda x: x["rerank_score"], reverse=True)
            reranked_all.append(scored[:5])

        # ── Stage 4: collect documents + build prompts ────────────────────
        prompts = []
        active_indices = []
        doc_lists = []

        for q_idx, reranked in enumerate(reranked_all):
            if not reranked:
                doc_lists.append([])
                continue

            documents = []
            for r in reranked:
                if r["doc_id"] not in documents:
                    documents.append(r["doc_id"])
                if len(documents) == 3:
                    break

            doc_lists.append(documents)
            contexts = [r["chunk"] for r in reranked]
            prompts.append(self.build_prompt(questions[q_idx], contexts))
            active_indices.append(q_idx)

        # ── Stage 5: batched LLM generation ──────────────────────────────
        results = [{"documents": [], "answer": ""} for _ in questions]

        if not prompts:
            return results

        outputs = self.generator(
            prompts,
            max_new_tokens=80,
            do_sample=False,
            repetition_penalty=1.1,
            return_full_text=False,
            batch_size=8            # increase to 16 if VRAM allows
        )

        # ── Stage 6: assemble ─────────────────────────────────────────────
        for out_idx, q_idx in enumerate(active_indices):
            raw = outputs[out_idx][0]["generated_text"].strip()
            answer = raw.split("\n")[0].strip()
            answer = answer.replace("Answer:", "").strip()

            if answer.upper().startswith("NOT_FOUND"):
                answer = ""

            results[q_idx] = {
                "documents": doc_lists[q_idx],
                "answer": answer
            }

        return results

    # =====================================
    # QA — single question wrapper (backward compat)
    # =====================================
    def qa(self, question: str) -> dict:
        return self.qa_batch([question])[0]