import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from results_writer import write_results_to_tsv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Generic, public model IDs (downloaded automatically by sentence-transformers)
DEFAULT_MODELS = [
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
]


def iter_jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from e


def load_models(model_ids: list[str]) -> dict[str, SentenceTransformer]:
    models: dict[str, SentenceTransformer] = {}
    for model_id in model_ids:
        models[model_id] = SentenceTransformer(model_id)
    return models


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generic embedding evaluation runner for comparing request-building strategies across two models."
    )
    parser.add_argument(
        "--data", type=str, default="data/sample.jsonl", help="Path to JSONL dataset"
    )
    parser.add_argument(
        "--out", type=str, default="results/results.tsv", help="Path to output TSV"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="One or more SentenceTransformer model IDs (e.g. BAAI/bge-base-en-v1.5)",
    )
    parser.add_argument(
        "--entity-type",
        type=str,
        default="",
        help="Optional filter: only evaluate rows where entity_type matches",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Strategy modules (generic) live under strategies/
    from strategies import chunking, cleansed, current, weighted_chunking

    strategy_processors = {
        "current": current.process_row,
        "chunking": chunking.process_row,
        "weighted_chunking": weighted_chunking.process_row,
        "cleansed": cleansed.process_row,
    }

    models = load_models(list(args.models))

    results: list[dict[str, Any]] = []
    rows = list(iter_jsonl_rows(data_path))
    if args.entity_type:
        rows = [r for r in rows if str(r.get("entity_type", "")) == args.entity_type]

    for row in tqdm(rows, desc="Evaluating rows"):
        for model_id, model in models.items():
            for strategy_name, processor in strategy_processors.items():
                results.append(
                    processor(row, model, model_id=model_id, strategy=strategy_name)
                )

    write_results_to_tsv(str(out_path), results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
