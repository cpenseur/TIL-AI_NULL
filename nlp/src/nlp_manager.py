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
        # EMBEDDING MODEL
        # =====================================
        self.embedding_model = SentenceTransformer(
            str(MODELS_DIR / "bge-base-en-v1.5")
        )

        # =====================================
        # RERANKER
        # =====================================
        self.reranker = CrossEncoder(
            str(MODELS_DIR / "bge-reranker-base")
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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id  # suppress padding warning
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

            split_chunks = self.text_splitter.split_text(text)

            for chunk in split_chunks:
                chunks.append(chunk)
                chunk_to_doc.append(doc_id)

        logger.info(f"load_corpus: {len(chunks)} chunks after splitting")

        self.chunks = chunks
        self.chunk_to_doc = chunk_to_doc

        # =====================================
        # EMBEDDINGS
        # batch_size=32 is safe for CPU; won't OOM
        # show_progress_bar=False avoids tqdm overhead in server context
        # =====================================
        embeddings = self.embedding_model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32
        ).astype("float32")

        self.chunk_embeddings = embeddings

        # =====================================
        # FAISS (CPU only — safe on all envs)
        # =====================================
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)

        # =====================================
        # BM25
        # =====================================
        tokenized_chunks = [self.bm25_tokenize(c) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        self.loaded = True
        logger.info("load_corpus: done")

    # =====================================
    # HYBRID RETRIEVAL
    # Wider funnel than original (was dense_k=20, sparse_k=20, top_k=8)
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
            top_sparse_idx = np.argpartition(
                bm25_scores, -sparse_k
            )[-sparse_k:]

        sparse_scores = {
            int(i): float(bm25_scores[i])
            for i in top_sparse_idx
        }

        candidates = sorted(
            set(dense_scores.keys()) | set(sparse_scores.keys())
        )

        if not candidates:
            return []

        svals = np.array(
            [sparse_scores.get(i, 0.0) for i in candidates],
            dtype="float32"
        )

        smin = float(svals.min())
        smax = float(svals.max())

        def normalize_sparse(v: float) -> float:
            return (v - smin) / (smax - smin + 1e-9)

        combined = []

        for idx in candidates:
            dense = dense_scores.get(idx, 0.0)
            sparse = normalize_sparse(sparse_scores.get(idx, 0.0))
            score = alpha * dense + (1 - alpha) * sparse

            combined.append({
                "chunk": self.chunks[idx],
                "doc_id": self.chunk_to_doc[idx],
                "score": float(score)
            })

        combined.sort(key=lambda x: x["score"], reverse=True)

        return combined[:top_k]

    # =====================================
    # RERANK
    # top_k raised from 3 → 5 so reranker sees more candidates
    # =====================================
    def rerank(self, question: str, retrieved: list[dict], top_k: int = 5) -> list[dict]:

        if not retrieved:
            return []

        pairs = [[question, r["chunk"]] for r in retrieved]
        scores = self.reranker.predict(pairs)

        reranked = []

        for r, s in zip(retrieved, scores):
            reranked.append({
                **r,
                "rerank_score": float(s)
            })

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]

    # =====================================
    # PROMPT
    # Simplified vs original — shorter prompts work better on 1.5B models
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
    # QA — single question (used by server per-question calls)
    # =====================================
    def qa(self, question: str) -> dict:

        # 1. RETRIEVE — wider funnel than original
        retrieved = self.hybrid_retrieve(
            question,
            dense_k=50,
            sparse_k=50,
            top_k=20
        )

        if not retrieved:
            return {"documents": [], "answer": ""}

        # 2. RERANK
        reranked = self.rerank(question, retrieved, top_k=5)

        if not reranked:
            return {"documents": [], "answer": ""}

        # =====================================
        # ALWAYS collect documents before any answer decision.
        # Returning docs with empty answer = 0.4 pts.
        # Returning nothing = 0.0 pts.
        # =====================================
        documents = []
        for r in reranked:
            if r["doc_id"] not in documents:
                documents.append(r["doc_id"])
            if len(documents) == 3:
                break

        # =====================================
        # SOFT ABSTENTION — only skip generation if reranker
        # is extremely unconfident. Still return documents.
        # =====================================
        top1 = reranked[0]["rerank_score"]
        if top1 < -5.0:  # effectively never fires; safety net only
            return {"documents": documents, "answer": ""}

        # 3. BUILD CONTEXT + PROMPT
        contexts = [r["chunk"] for r in reranked]
        prompt = self.build_prompt(question, contexts)

        # 4. GENERATE
        output = self.generator(
            prompt,
            max_new_tokens=80,       # was 40 — enough for multi-part answers
            do_sample=False,
            repetition_penalty=1.1,  # prevents looping/repetition
            return_full_text=False
        )

        raw = output[0]["generated_text"].strip()
        answer = raw.split("\n")[0].strip()
        answer = answer.replace("Answer:", "").strip()

        # 5. NOT_FOUND → empty answer but keep documents for partial credit
        if answer.upper().startswith("NOT_FOUND"):
            return {"documents": documents, "answer": ""}

        return {
            "documents": documents,
            "answer": answer
        }

    # =====================================
    # QA BATCH — call this from your server when handling a
    # batch of questions for much better LLM throughput
    # =====================================
    def qa_batch(self, questions: list[str]) -> list[dict]:
        """
        Process a list of questions together, batching the LLM call.
        Drop-in replacement for calling qa() in a loop.
        """
        # Stage 1: retrieve + rerank for each question
        batch_meta = []  # None or (documents, contexts)

        for question in questions:
            retrieved = self.hybrid_retrieve(question, dense_k=50, sparse_k=50, top_k=20)

            if not retrieved:
                batch_meta.append(None)
                continue

            reranked = self.rerank(question, retrieved, top_k=5)

            if not reranked:
                batch_meta.append(None)
                continue

            documents = []
            for r in reranked:
                if r["doc_id"] not in documents:
                    documents.append(r["doc_id"])
                if len(documents) == 3:
                    break

            contexts = [r["chunk"] for r in reranked]
            batch_meta.append((documents, contexts))

        # Stage 2: collect prompts for non-empty entries
        prompts = []
        active_indices = []

        for i, meta in enumerate(batch_meta):
            if meta is not None:
                documents, contexts = meta
                prompts.append(self.build_prompt(questions[i], contexts))
                active_indices.append(i)

        if not prompts:
            return [{"documents": [], "answer": ""} for _ in questions]

        # Stage 3: single batched LLM call
        outputs = self.generator(
            prompts,
            max_new_tokens=80,
            do_sample=False,
            repetition_penalty=1.1,
            return_full_text=False,
            batch_size=4  # safe for 1.5B; increase to 8 if GPU VRAM allows
        )

        # Stage 4: assemble results
        results = [{"documents": [], "answer": ""} for _ in questions]

        for out_idx, q_idx in enumerate(active_indices):
            documents, _ = batch_meta[q_idx]

            raw = outputs[out_idx][0]["generated_text"].strip()
            answer = raw.split("\n")[0].strip()
            answer = answer.replace("Answer:", "").strip()

            if answer.upper().startswith("NOT_FOUND"):
                answer = ""

            results[q_idx] = {"documents": documents, "answer": answer}

        return results