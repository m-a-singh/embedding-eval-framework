import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# -----------------------------
# Small data helpers
# -----------------------------


def _load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


def _load_tsv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t")
    except FileNotFoundError:
        return pd.DataFrame()


def _read_jsonl(path: Path, max_rows: int = 5000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
            if len(rows) >= max_rows:
                break
    return rows


# -----------------------------
# App
# -----------------------------

st.set_page_config(page_title="Embedding Eval UI", layout="wide")

st.title("Embedding Eval Framework — Import → Run → View")

page = st.sidebar.radio(
    "Navigation",
    options=["1) Import dataset", "2) View / Run results (data_viewer.py)"],
    index=0,
)

# -----------------------------
# 1) Import
# -----------------------------

if page == "1) Import dataset":
    st.header("Import dataset")
    st.write(
        "Upload a CSV/JSON/JSONL, map columns to the framework schema, and write `data/sample.jsonl`."
    )

    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV, JSON, or JSONL)", type=["csv", "json", "jsonl"]
    )

    if uploaded_file is None:
        st.stop()

    try:
        ext = Path(uploaded_file.name).suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(uploaded_file)
        elif ext == ".jsonl":
            df = pd.read_json(uploaded_file, lines=True)
        else:
            df = pd.read_json(uploaded_file)
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        st.stop()

    st.success(f"Loaded '{uploaded_file.name}' with {len(df)} records")
    with st.expander("Preview"):
        st.dataframe(df.head(10))

    columns = ["-- Select Column --"] + list(df.columns)

    c1, c2 = st.columns(2)
    with c1:
        id_col = st.selectbox("id", options=columns)
        entity_col = st.selectbox("entity_type", options=columns)
        keyword_col = st.selectbox("keyword (required)", options=columns)
        relevance_col = st.selectbox("relevance_score (required)", options=columns)
        baseline_col = st.selectbox(
            "baseline_relevance_score (optional)", options=columns
        )

    with c2:
        st.caption(
            "json_data mapping: either map a prebuilt json_data column OR map primitive subfields."
        )
        json_data_col = st.selectbox("json_data column (optional)", options=columns)
        st.markdown("**OR** map json_data subfields:")
        name_col = st.selectbox("json_data.name", options=columns)
        desc_col = st.selectbox("json_data.description", options=columns)
        sponsors_col = st.selectbox("json_data.sponsors", options=columns)
        tag_categories_col = st.selectbox("json_data.tagCategories", options=columns)
        tag_topics_col = st.selectbox("json_data.tagTopics", options=columns)

    st.subheader("Downsample")
    sample_size = st.slider(
        "Max rows to export",
        min_value=1,
        max_value=max(1, len(df)),
        value=min(50, len(df)) if len(df) else 1,
    )

    st.sidebar.header("Weighted chunking: field weights")
    st.sidebar.caption(
        "Optional: write `field_weights` into each row (your weighted_chunking can consume this)."
    )
    weights_enabled = st.sidebar.checkbox("Include field_weights in output", value=True)

    default_fields = ["name", "description", "sponsors", "tagCategories", "tagTopics"]
    weights: Dict[str, float] = {}
    if weights_enabled:
        raw_fields = st.sidebar.text_area(
            "Field names (one per line)", value="\n".join(default_fields)
        )
        field_names = [f.strip() for f in raw_fields.splitlines() if f.strip()]
        for f in field_names:
            weights[f] = float(
                st.sidebar.slider(
                    f"weight: {f}", min_value=0.0, max_value=5.0, value=1.0, step=0.1
                )
            )

    # Use your existing importer logic by delegating to the module script via subprocess.
    # This avoids duplicating the parsing quirks in two places.
    st.info(
        "This unified UI reuses the existing importer code by calling `dataset_importer.py` logic indirectly is not available; "
        "so this page writes `data/sample.jsonl` directly using a minimal, safe mapping approach."
    )

    def _cell(v: Any) -> Any:
        if v is None:
            return None
        try:
            if bool(pd.isna(v)):
                return None
        except Exception:
            pass
        return v

    def _to_str(row: Dict[str, Any], col: str, default: str = "") -> str:
        if col == "-- Select Column --":
            return default
        v = _cell(row.get(col))
        if v is None or isinstance(v, (dict, list)):
            return default
        return str(v).strip()

    def _to_int(row: Dict[str, Any], col: str, default: int = 0) -> int:
        if col == "-- Select Column --":
            return default
        v = _cell(row.get(col))
        if v is None:
            return default
        try:
            return int(float(v))
        except Exception:
            return default

    def _to_list(row: Dict[str, Any], col: str) -> List[str]:
        if col == "-- Select Column --":
            return []
        v = _cell(row.get(col))
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, dict):
            return []
        return [s.strip() for s in str(v).split(",") if s.strip()]

    def _parse_pythonish(obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, str):
            s = obj.strip()
            if (s.startswith("{") and s.endswith("}")) or (
                s.startswith("[") and s.endswith("]")
            ):
                try:
                    return json.loads(s)
                except Exception:
                    pass
                try:
                    import ast

                    return ast.literal_eval(s)
                except Exception:
                    return obj
        return obj

    def _coerce_json_data_block(v: Any) -> Any:
        # Handles dict OR stringified dict. Also unwraps the sample_input.jsonl pattern.
        v = _parse_pythonish(v)
        if not isinstance(v, dict):
            return None

        # parse one-level nested pythonish strings
        v1 = {k: _parse_pythonish(val) for k, val in v.items()}

        # sample_input.jsonl pattern: json_data['name'] is a stringified dict with the full payload,
        # and inside that payload, ['name'] is another stringified dict containing real name/description.
        name_val = v1.get("name")
        if (
            isinstance(name_val, str)
            and name_val.strip().startswith("{")
            and name_val.strip().endswith("}")
        ):
            lvl1 = _parse_pythonish(name_val)
            if isinstance(lvl1, dict):
                lvl1 = {k: _parse_pythonish(val) for k, val in lvl1.items()}
                inner_name = lvl1.get("name")
                if (
                    isinstance(inner_name, str)
                    and inner_name.strip().startswith("{")
                    and inner_name.strip().endswith("}")
                ):
                    lvl2 = _parse_pythonish(inner_name)
                    if isinstance(lvl2, dict):
                        lvl2 = {k: _parse_pythonish(val) for k, val in lvl2.items()}
                        return lvl2
                return lvl1

        return v1

    def _build_json_data_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
        # If user mapped a prebuilt json_data column, parse + normalize.
        if json_data_col != "-- Select Column --":
            block = _coerce_json_data_block(row.get(json_data_col))
            if isinstance(block, dict):

                def _try_parse_nested_dict_string(s: str) -> Any:
                    ss = s.strip()
                    if ss.startswith("{") and ss.endswith("}"):
                        try:
                            import ast

                            return ast.literal_eval(ss)
                        except Exception:
                            return s
                    return s

                def _as_list(x: Any) -> List[str]:
                    if x is None:
                        return []
                    if isinstance(x, list):
                        out: List[str] = []
                        for item in x:
                            # If list items are fragments like "'sponsors': ['Acme Bank']", salvage the value.
                            if isinstance(item, str) and ":" in item:
                                _, rhs = item.split(":", 1)
                                rhs = rhs.strip().strip(",")
                                parsed_rhs = _parse_pythonish(rhs)
                                if isinstance(parsed_rhs, list):
                                    out.extend(
                                        [
                                            str(v).strip()
                                            for v in parsed_rhs
                                            if str(v).strip()
                                        ]
                                    )
                                    continue
                                if isinstance(parsed_rhs, str) and parsed_rhs:
                                    out.append(parsed_rhs.strip().strip('"').strip("'"))
                                    continue

                            # If list item is a stringified dict, parse it.
                            if isinstance(item, str):
                                parsed_item = _try_parse_nested_dict_string(item)
                                if isinstance(parsed_item, dict):
                                    # If it contains known keys, extract any list-ish values
                                    for k in ("sponsors", "tagCategories", "tagTopics"):
                                        vv = parsed_item.get(k)
                                        if isinstance(vv, list):
                                            out.extend(
                                                [
                                                    str(v).strip()
                                                    for v in vv
                                                    if str(v).strip()
                                                ]
                                            )
                                    continue

                            s = str(item).strip()
                            if s:
                                out.append(s)
                        # de-dupe preserving order
                        seen: set[str] = set()
                        deduped: List[str] = []
                        for v in out:
                            if v not in seen:
                                seen.add(v)
                                deduped.append(v)
                        return deduped

                    if isinstance(x, dict):
                        return []

                    parsed = _parse_pythonish(str(x))
                    if isinstance(parsed, list):
                        return [str(v).strip() for v in parsed if str(v).strip()]

                    return [s.strip() for s in str(x).split(",") if s.strip()]

                name = block.get("name")
                desc = block.get("description")

                # If name/description are not plain strings, attempt to unwrap nested dict-as-string.
                name_u = _parse_pythonish(name) if isinstance(name, str) else name
                desc_u = _parse_pythonish(desc) if isinstance(desc, str) else desc

                # sample_input.jsonl: name is a dict-like string whose 'name' is another dict-like string
                if isinstance(name_u, dict) and isinstance(name_u.get("name"), str):
                    inner = _parse_pythonish(name_u.get("name"))
                    if isinstance(inner, dict):
                        name_u = inner
                if isinstance(desc_u, dict) and isinstance(
                    desc_u.get("description"), str
                ):
                    inner = _parse_pythonish(desc_u.get("description"))
                    if isinstance(inner, dict):
                        desc_u = inner

                name_s = (
                    name_u
                    if isinstance(name_u, str)
                    else (name_u.get("name") if isinstance(name_u, dict) else "")
                )
                desc_s = (
                    desc_u
                    if isinstance(desc_u, str)
                    else (desc_u.get("description") if isinstance(desc_u, dict) else "")
                )

                if not desc_s and name_s:
                    desc_s = name_s

                # If sponsors/tag lists are broken fragments, prefer extracting from the name_u/desc_u dict payload if present.
                sponsors_src = block.get("sponsors")
                tag_cat_src = block.get("tagCategories")
                tag_top_src = block.get("tagTopics")

                if isinstance(name_u, dict):
                    sponsors_src = name_u.get("sponsors", sponsors_src)
                    tag_cat_src = name_u.get("tagCategories", tag_cat_src)
                    tag_top_src = name_u.get("tagTopics", tag_top_src)

                return {
                    "name": name_s or "",
                    "description": desc_s or "",
                    "sponsors": _as_list(sponsors_src),
                    "tagCategories": _as_list(tag_cat_src),
                    "tagTopics": _as_list(tag_top_src),
                }

        name = _to_str(row, name_col, "")
        desc = _to_str(row, desc_col, "")
        if not desc and name:
            desc = name

        return {
            "name": name,
            "description": desc,
            "sponsors": _to_list(row, sponsors_col),
            "tagCategories": _to_list(row, tag_categories_col),
            "tagTopics": _to_list(row, tag_topics_col),
        }

    if st.button("Write data/sample.jsonl", type="primary"):
        if (
            keyword_col == "-- Select Column --"
            or relevance_col == "-- Select Column --"
        ):
            st.error("keyword and relevance_score are required")
            st.stop()

        sampled_df = (
            df.sample(n=min(sample_size, len(df)), random_state=42)
            if len(df) > sample_size
            else df
        )
        records: List[Dict[str, Any]] = sampled_df.to_dict(orient="records")

        out_rows: List[Dict[str, Any]] = []
        for i, row in enumerate(records, start=1):
            rel = _to_int(row, relevance_col, 0)
            base = (
                _to_int(row, baseline_col, rel)
                if baseline_col != "-- Select Column --"
                else rel
            )

            out: Dict[str, Any] = {
                "id": _to_str(row, id_col, default=f"rec_{i:03d}"),
                "entity_type": _to_str(row, entity_col, default="episode"),
                "json_data": _build_json_data_from_row(row),
                "keyword": _to_str(row, keyword_col, default=""),
                "relevance_score": rel,
                "baseline_relevance_score": base,
            }
            if weights_enabled:
                out["field_weights"] = weights
            out_rows.append(out)

        target_path = Path("data/sample.jsonl")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        st.success(f"Wrote {len(out_rows)} rows to {target_path}")
        with st.expander("First row"):
            st.json(out_rows[0] if out_rows else {})

# -----------------------------
# 2) View/Run results (delegate to data_viewer.py)
# -----------------------------

if page == "2) View / Run results (data_viewer.py)":
    st.header("View / Run results")
    st.write(
        "This project already has a stable dashboard at `data_viewer.py`. Use it for running the pipeline and viewing results."
    )

    st.subheader("Recommended")
    st.code("streamlit run data_viewer.py", language="bash")

    st.subheader("One-click (starts a second Streamlit server on port 8502)")
    st.caption("If something is already using 8502, change the port below.")
    port = st.number_input("Port", min_value=8502, max_value=8999, value=8502, step=1)

    if st.button("Start data_viewer.py", type="primary"):
        # Best-effort: launch another Streamlit process.
        # Note: this is mainly for local dev; in hosted environments it may not work.
        cmd = [
            "python3",
            "-m",
            "streamlit",
            "run",
            "data_viewer.py",
            "--server.port",
            str(int(port)),
            "--server.headless",
            "true",
        ]
        subprocess.Popen(cmd)  # noqa: S603,S607
        st.success(f"Started data_viewer.py on http://localhost:{int(port)}")
        st.info("If it doesn't open automatically, copy/paste the URL above.")
