import json
import re

RESULTS_PATH = (
    "data/evaluation/500_docs_parent_child/metrics_500/"
    "retrieval_evaluation_results_with_reranker_withchunkchange.json"
)


def to_doc_id(chunk_id: str) -> str:
  return chunk_id.split(":")[0]


def has_specific_anchor(question: str) -> bool:
  # Heuristic check for capitalized proper nouns/places, numbers/codes/dates, or quoted terms
  has_number = bool(re.search(r"\b\d+[a-zA-Z0-9/-]*\b", question))
  has_caps = bool(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", question[1:]))
  return has_number or has_caps


with open(RESULTS_PATH, "r", encoding="utf-8") as f:
  data = json.load(f)

rows = data["detailed_results"]["Hybrid + BGE Reranker"]

doc_hits, doc_misses = [], []
for item in rows:
  gold_docs = {to_doc_id(cid) for cid in item.get("relevant_chunk_ids", [])}
  ret_docs = [to_doc_id(cid) for cid in item.get("retrieved_chunk_ids", [])[:15]]
  hit = bool(set(ret_docs) & gold_docs)
  (doc_hits if hit else doc_misses).append(item)

print(f"Doc@15 Hits:   {len(doc_hits)} / {len(rows)} ({len(doc_hits)/len(rows):.2%})")
print(f"Doc@15 Misses: {len(doc_misses)} / {len(rows)} ({len(doc_misses)/len(rows):.2%})")

anchored_hits = sum(1 for x in doc_hits if has_specific_anchor(x["question"]))
anchored_misses = sum(1 for x in doc_misses if has_specific_anchor(x["question"]))
print(f"Heuristic anchored among Hits:   {anchored_hits}/{len(doc_hits)}")
print(f"Heuristic anchored among Misses: {anchored_misses}/{len(doc_misses)}")

print("\n--- Sample 15 Doc@15 Misses ---")
for item in doc_misses[:15]:
  q = item["question"]
  tag = "ANCHORED" if has_specific_anchor(q) else "GENERIC "
  print(f"[{tag}] ({item['doc_id']}) {q}")