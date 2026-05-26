import csv
from typing import Any

RESULT_FIELDNAMES = [
    "id",
    "entity_type",
    "keyword",
    "keyword_normalized",
    "model_name",
    "request",
    "triton_input",
    "chunking_strategy",
    "field_count",
    "token_length",
    "tokens",
    "cosine_similarity",
    "relevance_score",
    "baseline_relevance_score",
]


def write_results_to_tsv(output_path: str, rows: list[dict[str, Any]]) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as tsv_file:
        writer = csv.DictWriter(tsv_file, fieldnames=RESULT_FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
