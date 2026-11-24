import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "500_docs_parent_child"
    / "metrics_500"
    / "e2e_report_full_v4.txt"
)
DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "500_docs_parent_child"
    / "metrics_500"
    / "matched_gold_retrieved_citations.txt"
)

DIAG_ROW_RE = re.compile(
    r"^\s*(?P<q_num>\d+)\s*\|\s*(?P<q_id>[^|]+?)\s*\|\s*(?P<doc_id>[^|]+?)\s*\|"
    r"\s*(?P<max_tok>[^|]+?)\s*\|\s*(?P<in_tok>[^|]+?)\s*\|\s*(?P<out_tok>[^|]+?)\s*\|"
    r"\s*(?P<cost>[^|]+?)\s*\|\s*(?P<lat_ms>[^|]+?)\s*\|"
    r"\s*(?P<recall>[^|]+?)\s*\|\s*(?P<mrr>[^|]+?)\s*\|\s*(?P<faith>[^|]+?)\s*\|"
    r"\s*(?P<ans_rel>[^|]+?)\s*\|\s*(?P<ctx_prec>[^|]+?)\s*\|"
    r"\s*(?P<flags>[^|]+?)\s*\|\s*(?P<question>.*)$"
)

DETAIL_ROW_RE = re.compile(
    r"^\s*Q#(?P<q_num>\d+)\s*\((?P<q_id>[^)]+)\)\s*\|\s*"
    r"GoldAns:\s*(?P<gold_ans>.*?)\s*\|\s*"
    r"GoldChunkID:\s*(?P<gold_chunks>.*?)\s*\|\s*"
    r"Retrieved\(Rank:ChunkID\):\s*\[(?P<retrieved>.*?)\]\s*\|\s*"
    r"CitedChunkIDs:\s*\[(?P<cited>.*?)\]\s*$"
)


def parse_chunk_id_list(raw: str) -> list[str]:
    text = raw.strip()
    if not text or text == "-":
        return []
    ids: list[str] = []
    for part in text.split(","):
        token = part.strip()
        if not token or token == "-":
            continue
        # Strip rank prefix like "#1:7767.jpeg:child:0:1" -> "7767.jpeg:child:0:1"
        if token.startswith("#") and ":" in token:
            _, chunk_id = token.split(":", 1)
            token = chunk_id.strip()
        if token and token != "-":
            ids.append(token)
    return ids


def parse_report_text(text: str) -> tuple[dict[int, dict], dict[int, dict]]:
    diagnostics: dict[int, dict] = {}
    details: dict[int, dict] = {}

    for line in text.splitlines():
        diag_match = DIAG_ROW_RE.match(line)
        if diag_match:
            q_num = int(diag_match.group("q_num"))
            diagnostics[q_num] = {
                "q_num": q_num,
                "q_id": diag_match.group("q_id").strip(),
                "doc_id": diag_match.group("doc_id").strip(),
                "lat_ms": diag_match.group("lat_ms").strip(),
                "recall": diag_match.group("recall").strip(),
                "mrr": diag_match.group("mrr").strip(),
                "faith": diag_match.group("faith").strip(),
                "ans_rel": diag_match.group("ans_rel").strip(),
                "ctx_prec": diag_match.group("ctx_prec").strip(),
                "flags": diag_match.group("flags").strip(),
                "question": diag_match.group("question").strip(),
            }
            continue

        detail_match = DETAIL_ROW_RE.match(line)
        if detail_match:
            q_num = int(detail_match.group("q_num"))
            gold_ids = parse_chunk_id_list(detail_match.group("gold_chunks"))
            retrieved_ids = parse_chunk_id_list(detail_match.group("retrieved"))
            cited_ids = parse_chunk_id_list(detail_match.group("cited"))
            details[q_num] = {
                "q_num": q_num,
                "q_id": detail_match.group("q_id").strip(),
                "gold_ans": detail_match.group("gold_ans").strip(),
                "gold_chunk_ids": gold_ids,
                "retrieved_chunk_ids": retrieved_ids,
                "cited_chunk_ids": cited_ids,
            }

    return diagnostics, details


def extract_matched_questions(
    diagnostics: dict[int, dict], details: dict[int, dict]
) -> list[dict]:
    matched: list[dict] = []
    all_q_nums = sorted(set(diagnostics.keys()) | set(details.keys()))

    for q_num in all_q_nums:
        diag = diagnostics.get(q_num, {})
        det = details.get(q_num, {})

        gold_ids = det.get("gold_chunk_ids", [])
        retrieved_ids = det.get("retrieved_chunk_ids", [])
        cited_ids = det.get("cited_chunk_ids", [])

        retrieved_set = set(retrieved_ids)
        cited_set = set(cited_ids)

        matched_ids = [cid for cid in gold_ids if cid in retrieved_set and cid in cited_set]
        matched_ids = list(dict.fromkeys(matched_ids))

        if matched_ids:
            matched.append(
                {
                    "q_num": q_num,
                    "q_id": det.get("q_id") or diag.get("q_id", ""),
                    "doc_id": diag.get("doc_id", ""),
                    "question": diag.get("question", ""),
                    "gold_ans": det.get("gold_ans", ""),
                    "matched_chunk_ids": matched_ids,
                    "gold_chunk_ids": gold_ids,
                    "retrieved_chunk_ids": retrieved_ids,
                    "cited_chunk_ids": cited_ids,
                    "recall": diag.get("recall", ""),
                    "mrr": diag.get("mrr", ""),
                    "faith": diag.get("faith", ""),
                    "ans_rel": diag.get("ans_rel", ""),
                    "ctx_prec": diag.get("ctx_prec", ""),
                    "flags": diag.get("flags", ""),
                    "lat_ms": diag.get("lat_ms", ""),
                }
            )

    return matched


def write_matched_report(
    matched: list[dict], output_path: Path, total_queries: int
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "MATCHED QUESTIONS: GOLD CHUNK ID IN BOTH RETRIEVED & CITED CHUNKS",
        f"Total Queries Scanned : {total_queries}",
        f"Total Matched Queries : {len(matched)}",
        "=" * 100,
        "",
    ]

    for idx, item in enumerate(matched, start=1):
        lines.extend(
            [
                f"[{idx}] Q#{item['q_num']} ({item['q_id']}) | Doc_ID: {item['doc_id']}",
                f"Question           : {item['question']}",
                f"Gold Answer        : {item['gold_ans']}",
                f"Matched Chunk IDs  : {item['matched_chunk_ids']}",
                f"Gold Chunk IDs     : {item['gold_chunk_ids']}",
                f"Retrieved Chunk IDs: {item['retrieved_chunk_ids']}",
                f"Cited Chunk IDs    : {item['cited_chunk_ids']}",
                f"Scores & Flags     : Recall={item['recall']}, MRR={item['mrr']}, "
                f"Faith={item['faith']}, AnsRel={item['ans_rel']}, "
                f"CtxPrec={item['ctx_prec']}, Flags={item['flags']}, Lat(ms)={item['lat_ms']}",
                "-" * 100,
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    input_path = DEFAULT_INPUT_FILE
    if not input_path.exists():
        raise FileNotFoundError(f"Report file not found: {input_path}")

    raw_text = input_path.read_text(encoding="utf-8", errors="replace").lstrip("\ufeff")
    diagnostics, details = parse_report_text(raw_text)
    matched = extract_matched_questions(diagnostics, details)
    total_queries = len(set(diagnostics.keys()) | set(details.keys()))

    write_matched_report(matched, DEFAULT_OUTPUT_FILE, total_queries=total_queries)
    print(f"Parsed diagnostic rows : {len(diagnostics)}")
    print(f"Parsed detail rows     : {len(details)}")
    print(f"Matched questions      : {len(matched)}")
    print(f"Saved output to        : {DEFAULT_OUTPUT_FILE}")


if __name__ == "__main__":
    main()