import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize embedding evaluation TSV outputs."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        type=str,
        default="results/results.tsv",
        help="Input TSV produced by run_evaluation.py",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        type=str,
        default="results/summary.csv",
        help="Output CSV",
    )
    parser.add_argument(
        "--relevant-threshold",
        type=float,
        default=1.0,
        help="Treat relevance_score >= threshold as relevant for classification-style metrics",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, sep="\t")

    # Coerce numeric columns
    for col in ["cosine_similarity", "relevance_score", "baseline_relevance_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_relevant"] = df["relevance_score"] >= float(args.relevant_threshold)

    group_cols = ["model_name", "chunking_strategy"]

    def safe_corr(g: pd.DataFrame) -> float:
        # Pearson correlation between cosine and relevance labels (graded)
        x = g["cosine_similarity"]
        y = g["relevance_score"]
        if x.notna().sum() < 2 or y.notna().sum() < 2:
            return float("nan")
        if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
            return float("nan")
        return float(x.corr(y))

    def summarize_group(g: pd.DataFrame) -> pd.Series:
        relevant = g.loc[g["is_relevant"], "cosine_similarity"]
        nonrelevant = g.loc[~g["is_relevant"], "cosine_similarity"]
        mean_relevant = float(relevant.mean())
        mean_nonrelevant = float(nonrelevant.mean())

        return pd.Series(
            {
                "n": int(len(g)),
                "mean_cosine": float(g["cosine_similarity"].mean()),
                "median_cosine": float(g["cosine_similarity"].median()),
                "mean_cosine_relevant": mean_relevant,
                "mean_cosine_nonrelevant": mean_nonrelevant,
                "delta_relevant_minus_nonrelevant": float(
                    mean_relevant - mean_nonrelevant
                ),
                "pearson_corr_cosine_vs_relevance": safe_corr(g),
            }
        )

    # NOTE: include_groups=False avoids a pandas FutureWarning where grouping columns
    # will be excluded from the applied frame in future versions.
    summary = (
        df.groupby(group_cols, dropna=False)
        .apply(summarize_group, include_groups=False)
        .reset_index()
        .sort_values(
            ["delta_relevant_minus_nonrelevant", "pearson_corr_cosine_vs_relevance"],
            ascending=False,
        )
    )

    summary.to_csv(out_path, index=False)
    print(f"Wrote summary to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
