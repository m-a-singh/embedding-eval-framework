from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from strategies.shared import (
    build_field_map,
    build_request,
    build_result_dict,
    mean_normalize_embeddings,
)
from triton_input_simulator import simulate_triton_encode
from utils import normalize_keyword


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

    field_map = build_field_map(json_data)
    request_text = build_request(field_map)

    chunk_texts: list[str] = []
    chunk_tokens_list: list[list[str]] = []

    field_embeddings: list[np.ndarray] = []
    for field_name, field_value in field_map.items():
        if not field_value:
            continue

        chunk_texts.append(field_value)
        tokens = model.tokenizer.tokenize(field_value)
        chunk_tokens_list.append(tokens)

        field_emb, _ = simulate_triton_encode(
            model, [field_value], f"{model_id}:{field_name}"
        )
        field_embeddings.append(field_emb[0])

    if field_embeddings:
        label_emb = np.array([mean_normalize_embeddings(field_embeddings)])
    else:
        label_emb, _ = simulate_triton_encode(model, [request_text], model_id)

    triton_input = "\n".join(
        [f"Chunk {i + 1}: {text}" for i, text in enumerate(chunk_texts)]
    )
    keyword_emb, _ = simulate_triton_encode(model, [keyword], model_id)
    cosine_score = float(cosine_similarity(keyword_emb, label_emb)[0][0])

    return build_result_dict(
        row_id=row_id,
        entity_type=entity_type,
        keyword=keyword,
        keyword_normalized=keyword_normalized,
        model_id=model_id,
        request_text=request_text,
        triton_input=triton_input if triton_input else "",
        strategy=strategy,
        field_count=len([value for value in field_map.values() if value]),
        token_length=sum(len(t) for t in chunk_tokens_list),
        tokens="\n".join(
            [
                f"Chunk {i + 1}: {' '.join(tokens)}"
                for i, tokens in enumerate(chunk_tokens_list)
            ]
        ),
        cosine_score=cosine_score,
        relevance_score=relevance_score,
        baseline_relevance_score=baseline_relevance_score,
    )
