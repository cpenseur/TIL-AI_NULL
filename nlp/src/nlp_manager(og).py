"""Manages the NLP model."""

import re
import faiss
import torch
import numpy as np
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


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
            tokenizer=self.tokenizer
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

        chunks = []
        chunk_to_doc = []

        for doc in documents:
            doc_id = doc["id"]
            text = doc["document"]

            split_chunks = self.text_splitter.split_text(text)

            for chunk in split_chunks:
                chunks.append(chunk)
                chunk_to_doc.append(doc_id)

        self.chunks = chunks
        self.chunk_to_doc = chunk_to_doc

        # embeddings
        embeddings = self.embedding_model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=True
        ).astype("float32")

        self.chunk_embeddings = embeddings

        # FAISS
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)

        # BM25
        tokenized_chunks = [self.bm25_tokenize(c) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        self.loaded = True

    # =====================================
    # HYBRID RETRIEVAL
    # =====================================
    def hybrid_retrieve(
        self,
        question: str,
        dense_k: int =20,
        sparse_k: int = 20,
        top_k: int = 8,
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
    # =====================================
    def rerank(self, question: str, retrieved: list[dict], top_k: int = 3):

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
    # =====================================

    def build_prompt(self, question: str, contexts: list[str]) -> str:

        context_text = "\n\n".join(
        [f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)]
    )

        return f"""
You are a precise question-answering system.

You MUST follow these rules:

1. Use ONLY the provided contexts.
2. Do NOT guess or use outside knowledge.
3. If the answer is not explicitly supported, output exactly:
NOT_FOUND

4. If the question requires a FACT (L1):
   - extract the exact value from context
   - do NOT rephrase incorrectly

5. If the question requires MULTIPLE facts (L2):
   - combine information from contexts carefully
   - show final computed result only

6. If the question requires CROSS-CONTEXT reasoning (L3):
   - use multiple contexts together
   - ensure all parts of the answer are supported

7. Do NOT include explanations.
8. Output ONLY the final answer.

CONTEXTS:
{context_text}

QUESTION:
{question}

ANSWER:
"""

    # =====================================
    # QA PIPELINE
    # =====================================
    def qa(self, question: str):

        # 1. RETRIEVE
        retrieved = self.hybrid_retrieve(
            question,
            dense_k=8,
            sparse_k=8,
            top_k=4
        )

        if not retrieved:
            return {"documents": [], "answer": ""}

        # 2. RERANK (ONLY SOURCE OF TRUTH NOW)
        reranked = self.rerank(question, retrieved, top_k=3)

        if not reranked:
            return {"documents": [], "answer": ""}

        # =====================================
        # L4/L5 SOFT ABSTENTION (YOUR FIX)
        # =====================================
        top1 = reranked[0]["rerank_score"]
        top3_avg = sum(
            r["rerank_score"] for r in reranked[:3]
        ) / len(reranked[:3])

        if top1 < 0.05 and top3_avg < 0.05:
            return {"documents": [], "answer": ""}

        # =====================================
        # DOCUMENT IDS
        # =====================================
        documents = []
        for r in reranked:
            if r["doc_id"] not in documents:
                documents.append(r["doc_id"])

        documents = documents[:3]

        # =====================================
        # CONTEXT
        # =====================================
        contexts = [r["chunk"] for r in reranked]

        prompt = self.build_prompt(question, contexts)

        # =====================================
        # GENERATION
        # =====================================
        output = self.generator(
            prompt,
            max_new_tokens=40,
            do_sample=False,
            temperature=0.1,
            return_full_text=False
        )
        answer = output[0]["generated_text"].strip()

        answer = answer.split("\n")[0].strip()
        answer = answer.replace("Answer:", "").strip()

        # =====================================
        # FINAL L5 CHECK
        # =====================================
        if answer.upper().startswith("NOT_FOUND"):
            return {"documents": documents, "answer": ""}

        return {
            "documents": documents,
            "answer": answer
        }