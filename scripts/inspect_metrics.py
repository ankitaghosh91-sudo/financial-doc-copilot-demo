import json

RESULTS_PATH = (
    "data/evaluation/500_docs_parent_child/metrics_500/"
    "retrieval_evaluation_results_with_reranker_withchunkchange.json"
)


def to_parent_id(chunk_id: str) -> str:
  # Converts "8065.jpeg:child:1:1" -> "8065.jpeg:parent:1"
  parts = chunk_id.split(":")
  if len(parts) >= 4 and parts[1] == "child":
    return f"{parts[0]}:parent:{parts[2]}"
  return chunk_id


def to_doc_id(chunk_id: str) -> str:
  # Converts "8065.jpeg:child:1:1" -> "8065.jpeg"
  return chunk_id.split(":")[0]


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
  if not gold_ids:
    return 0.0
  return len(set(retrieved_ids[:k]) & gold_ids) / len(gold_ids)


with open(RESULTS_PATH, "r", encoding="utf-8") as f:
  data = json.load(f)

detailed = data["detailed_results"]

levels = {
    "child": lambda x: x,
    "parent": to_parent_id,
    "doc": to_doc_id,
}

for method_name, rows in detailed.items():
  print(f"\n=== {method_name} (n={len(rows)}) ===")
  for level_name, mapper in levels.items():
    r5, r15 = [], []
    for item in rows:
      gold_mapped = {mapper(cid) for cid in item.get("relevant_chunk_ids", [])}
      ret_mapped = [mapper(cid) for cid in item.get("retrieved_chunk_ids", [])]
      r5.append(recall_at_k(ret_mapped, gold_mapped, 5))
      r15.append(recall_at_k(ret_mapped, gold_mapped, 15))

    print(
        f"  [{level_name.upper():6s}] Recall@5: {sum(r5)/len(r5):.4f} | "
        f"Recall@15: {sum(r15)/len(r15):.4f}"
    )