import math

import numpy as np
import pandas as pd

from strategies.shared import (
    build_field_map,
    build_request,
    mean_normalize_embeddings,
)
from utils import (
    cleanse_text,
    cleansed_normalize_labeled_list_field,
    cleansed_normalize_scalar_field,
    normalize_keyword,
    normalize_labeled_list_field,
)


def test_normalize_keyword_strips_and_lowercases():
    assert normalize_keyword("  Hello WORLD  ") == "hello world"


def test_normalize_labeled_list_field_sorts_and_commas():
    assert normalize_labeled_list_field(["b", "A", " c "]) == "A,b,c"


def test_normalize_labeled_list_field_handles_sets_and_empty_items():
    # order doesn't matter in the input, but output should be sorted + cleaned
    assert normalize_labeled_list_field({"  ", "B", "a", "a"}) == "a,B"


def test_cleanse_text_ascii_whitespace_and_strip():
    assert cleanse_text("  café\n\t  au  lait  ") == "caf au lait"


def test_build_request_all_empty_returns_empty_string():
    field_map = {
        "name": "",
        "sponsors": "",
        "description": "",
        "tagCategories": "",
        "tagTopics": "",
    }
    assert build_request(field_map) == ""


def test_build_result_dict_schema_and_values():
    # Ensures shared result schema stays stable.
    from strategies.shared import build_result_dict

    out = build_result_dict(
        row_id="1",
        entity_type="episode",
        keyword="K",
        keyword_normalized="k",
        model_id="m",
        request_text="req",
        triton_input="t",
        strategy="chunking",
        field_count=3,
        token_length=10,
        tokens="a b",
        cosine_score=0.5,
        relevance_score=2.0,
        baseline_relevance_score=1.0,
    )

    # key set matches results_writer.RESULT_FIELDNAMES
    from results_writer import RESULT_FIELDNAMES

    assert list(out.keys()) == RESULT_FIELDNAMES
    assert out["model_name"] == "m"
    assert out["chunking_strategy"] == "chunking"


def test_build_field_map_normalizes_and_falls_back_description_to_name():
    json_data = {
        "name": "  Name ",
        "sponsors": ["b", "A"],
        "description": None,
        "tagCategories": ["  z ", "y"],
        "tagTopics": None,
    }

    fm = build_field_map(json_data)
    assert fm["name"] == "Name"
    assert fm["description"] == "Name"  # fallback
    assert fm["sponsors"] == "A b"  # unlabeled list field joins with spaces
    assert fm["tagCategories"] == "y z"
    assert fm["tagTopics"] == ""


def test_build_request_joins_nonempty_values_in_order():
    field_map = {
        "name": "N",
        "sponsors": "",
        "description": "D",
        "tagCategories": "C",
        "tagTopics": "",
    }
    assert build_request(field_map) == "N D C"


def test_mean_normalize_embeddings_unit_norm_when_nonzero():
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    out = mean_normalize_embeddings([e1, e2])
    assert out.shape == (3,)
    assert math.isclose(float(np.linalg.norm(out)), 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_mean_normalize_embeddings_all_zero_returns_zero_vector():
    e1 = np.zeros(3)
    e2 = np.zeros(3)
    out = mean_normalize_embeddings([e1, e2])
    assert np.all(out == 0.0)


def test_cleansed_normalize_labeled_list_field_dedupes_casefold_and_ascii():
    # 'café' -> 'caf', and duplicates should be removed case-insensitively
    out = cleansed_normalize_labeled_list_field(["café", "CAFÉ", "  test  "])
    assert out == "caf,test"


def test_cleansed_normalize_scalar_field_handles_none_and_strips():
    assert cleansed_normalize_scalar_field(None) == ""
    assert cleansed_normalize_scalar_field("  hi\nthere ") == "hi there"


def test_build_report_aggregations_smoke():
    # Mirror build_report.py logic in a minimal, deterministic way
    df = pd.DataFrame(
        [
            {
                "model_name": "m1",
                "chunking_strategy": "current",
                "cosine_similarity": 0.9,
                "relevance_score": 2.0,
            },
            {
                "model_name": "m1",
                "chunking_strategy": "current",
                "cosine_similarity": 0.1,
                "relevance_score": 0.0,
            },
            {
                "model_name": "m1",
                "chunking_strategy": "chunking",
                "cosine_similarity": 0.2,
                "relevance_score": 2.0,
            },
        ]
    )

    relevant_threshold = 1.0
    df["is_relevant"] = df["relevance_score"] >= relevant_threshold

    group_cols = ["model_name", "chunking_strategy"]

    def safe_corr(g: pd.DataFrame) -> float:
        x = g["cosine_similarity"]
        y = g["relevance_score"]
        if x.notna().sum() < 2 or y.notna().sum() < 2:
            return float("nan")
        if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
            return float("nan")
        return float(x.corr(y))

    summary = (
        df.groupby(group_cols, dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n": int(len(g)),
                    "mean_cosine": float(g["cosine_similarity"].mean()),
                    "median_cosine": float(g["cosine_similarity"].median()),
                    "mean_cosine_relevant": float(
                        g.loc[g["is_relevant"], "cosine_similarity"].mean()
                    ),
                    "mean_cosine_nonrelevant": float(
                        g.loc[~g["is_relevant"], "cosine_similarity"].mean()
                    ),
                    "delta_relevant_minus_nonrelevant": float(
                        g.loc[g["is_relevant"], "cosine_similarity"].mean()
                        - g.loc[~g["is_relevant"], "cosine_similarity"].mean()
                    ),
                    "pearson_corr_cosine_vs_relevance": safe_corr(g),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    # Assertions: has both groups and expected columns
    assert set(summary["chunking_strategy"]) == {"current", "chunking"}
    assert "delta_relevant_minus_nonrelevant" in summary.columns
    assert summary.loc[summary["chunking_strategy"] == "current", "n"].item() == 2
