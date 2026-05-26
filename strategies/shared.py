from __future__ import annotations

from typing import Any

import numpy as np

from utils import normalize_scalar_field, normalize_unlabeled_list_field


def build_field_map(json_data: dict[str, Any]) -> dict[str, str]:
    return {
        "name": normalize_scalar_field(json_data.get("name")),
        "sponsors": normalize_unlabeled_list_field(json_data.get("sponsors")),
        "description": normalize_scalar_field(
            json_data.get("description") or json_data.get("name")
        ),
        "tagCategories": normalize_unlabeled_list_field(json_data.get("tagCategories")),
        "tagTopics": normalize_unlabeled_list_field(json_data.get("tagTopics")),
    }


def build_request(field_map: dict[str, str]) -> str:
    return " ".join(value for value in field_map.values() if value)


def mean_normalize_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    mean_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)
    norm = np.linalg.norm(mean_embedding)
    if norm == 0:
        return mean_embedding
    return mean_embedding / norm


def build_result_dict(
    *,
    row_id: str,
    entity_type: str,
    keyword: str,
    keyword_normalized: str,
    model_id: str,
    request_text: str,
    triton_input: str,
    strategy: str,
    field_count: int | None,
    token_length: int,
    tokens: str,
    cosine_score: float,
    relevance_score: Any,
    baseline_relevance_score: Any,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "entity_type": entity_type,
        "keyword": keyword,
        "keyword_normalized": keyword_normalized,
        "model_name": model_id,
        "request": request_text,
        "triton_input": triton_input,
        "chunking_strategy": strategy,
        "field_count": field_count,
        "token_length": token_length,
        "tokens": tokens,
        "cosine_similarity": cosine_score,
        "relevance_score": relevance_score,
        "baseline_relevance_score": baseline_relevance_score,
    }
