from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder

BASE = Path("models")
BASE.mkdir(parents=True, exist_ok=True)

# 1) Embedding model
emb_dir = BASE / "bge-base-en-v1.5"
print("Downloading embedding ->", emb_dir)
SentenceTransformer("BAAI/bge-base-en-v1.5").save(str(emb_dir))

# 2) Reranker
rerank_dir = BASE / "bge-reranker-base"
print("Downloading reranker ->", rerank_dir)
CrossEncoder("BAAI/bge-reranker-base").save(str(rerank_dir))

# 3) LLM (Qwen)
llm_dir = BASE / "Qwen2.5-1.5B-Instruct"
print("Downloading LLM ->", llm_dir)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tok.save_pretrained(str(llm_dir))

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=dtype,
    device_map="auto",
)
model.save_pretrained(str(llm_dir))

print("Done.")
