from typing import Any

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from triton_input_simulator import simulate_triton_encode
from utils import (
    cleansed_normalize_labeled_list_field,
    cleansed_normalize_scalar_field,
    normalize_keyword,
)


def build_request(json_data: dict[str, Any]) -> str:
    parts = [
        cleansed_normalize_scalar_field(json_data.get("name")),
        cleansed_normalize_labeled_list_field(json_data.get("sponsors")),
        cleansed_normalize_scalar_field(
            json_data.get("description") or json_data.get("name")
        ),
        cleansed_normalize_labeled_list_field(json_data.get("tagCategories")),
        cleansed_normalize_labeled_list_field(json_data.get("tagTopics")),
    ]
    return " ".join(part for part in parts if part)


def process_row(
    row: dict[str, Any],
    model: SentenceTransformer,
    *,
    model_id: str,
    strategy: str,
) -> dict[str, Any]:
    keyword = str(row.get("keyword", ""))
    keyword_normalized = normalize_keyword(keyword)

    entity_type = str(row.get("entity_type", ""))
    json_data = row.get("json_data") if isinstance(row.get("json_data"), dict) else {}

    relevance_score = row.get("relevance_score")
    baseline_relevance_score = row.get("baseline_relevance_score")
    row_id = str(row.get("id", ""))

    request_text = build_request(json_data)

    request_tokens = model.tokenizer.tokenize(request_text)

    label_emb, triton_input = simulate_triton_encode(model, [request_text], model_id)
    keyword_emb, _ = simulate_triton_encode(model, [keyword], model_id)
    cosine_score = float(cosine_similarity(keyword_emb, label_emb)[0][0])

    return {
        "id": row_id,
        "entity_type": entity_type,
        "keyword": keyword,
        "keyword_normalized": keyword_normalized,
        "model_name": model_id,
        "request": request_text,
        "triton_input": triton_input[0] if triton_input else "",
        "chunking_strategy": strategy,
        "field_count": None,
        "token_length": len(request_tokens),
        "tokens": " ".join(request_tokens),
        "cosine_similarity": cosine_score,
        "relevance_score": relevance_score,
        "baseline_relevance_score": baseline_relevance_score,
    }
