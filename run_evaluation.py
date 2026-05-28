import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dataset Importer Pipeline", layout="centered")

st.title("📦 Dataset Importer & Mapping Pipeline")
st.write(
    "Upload any custom CSV or JSON dataset, map your columns to the evaluation framework features, and export a compatible `sample.jsonl` file instantly."
)

# --- File Upload Section ---
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV, JSON, or JSONL)", type=["csv", "json", "jsonl"]
)

if uploaded_file is not None:
    # 1. Safely load the data into a Pandas DataFrame for mapping
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()
        if file_extension == ".csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension == ".jsonl":
            df = pd.read_json(uploaded_file, lines=True)
        else:
            # Standard JSON
            df = pd.read_json(uploaded_file)

        st.success(
            f"Successfully loaded '{uploaded_file.name}' with {len(df)} records!"
        )

        # Preview raw structure
        with st.expander("🔍 Preview Uploaded Raw Data"):
            st.dataframe(df.head(3))

    except Exception as e:
        st.error(f"Error parsing file: {e}")
        st.stop()

    st.markdown("---")
    st.subheader("🎯 Map Columns to Framework Fields")

    # Available columns from user's dataset
    columns = ["-- Select Column --"] + list(df.columns)

    # Grid UI layout for mappings
    col1, col2 = st.columns(2)

    with col1:
        id_col = st.selectbox("🔑 Unique Row Identifier (id):", options=columns)
        keyword_col = st.selectbox("🔎 Search Query Term (keyword):", options=columns)
        relevance_col = st.selectbox(
            "📊 Ground-Truth Target Label (relevance_score):", options=columns
        )
        baseline_col = st.selectbox(
            "📉 Baseline Score Option (baseline_relevance_score):", options=columns
        )
        entity_col = st.selectbox(
            "🏷️ Category/Filter String (entity_type):", options=columns
        )

    with col2:
        st.info(
            "**🧩 Nested Metadata Block (`json_data` mapping)**\n\nSelect the structural text fields to combine inside the evaluator block:"
        )
        meta_name = st.selectbox("📌 Context Item Name:", options=columns)
        meta_desc = st.selectbox("📝 Context Item Description:", options=columns)
        meta_tags_cat = st.selectbox(
            "📂 Category Tags Column (Array or Comma Separated):", options=columns
        )
        meta_tags_top = st.selectbox(
            "💬 Topic Tags Column (Array or Comma Separated):", options=columns
        )
        meta_sponsors = st.selectbox("🤝 Sponsors Column:", options=columns)

    st.markdown("---")

    # --- Sample Downsampling Range Slider ---
    st.subheader("⚙️ Downsample Preferences")
    max_sample = len(df) if len(df) > 15 else 15
    default_sample = len(df) if len(df) < 50 else 50
    sample_size = st.slider(
        "Select maximum rows to extract for evaluation:",
        min_value=10,
        max_value=max_sample,
        value=default_sample,
    )

    # --- Processing Action Button ---
    if st.button("🚀 Process, Map and Save Dataset", type="primary"):
        if (
            keyword_col == "-- Select Column --"
            or relevance_col == "-- Select Column --"
        ):
            st.error(
                "❌ 'keyword' and 'relevance_score' mappings are absolutely required by the framework evaluation loop."
            )
        else:
            with st.spinner("Refactoring dataset structure..."):
                if len(df) > sample_size:
                    sampled_df = df.sample(n=sample_size, random_state=42).copy()
                else:
                    sampled_df = df.copy()

                records: list[dict[str, Any]] = sampled_df.to_dict(orient="records")
                output_rows = []

                for idx, row in enumerate(records, start=1):
                    # Core normalizer to safely extract either standalone values OR nested dictionary sub-keys
                    # without repeating or mapping fields recursively
                    def extract_field(col_name: str, subkey: str) -> Any:
                        if col_name == "-- Select Column --":
                            return None

                        val = row.get(col_name)
                        if val is None or (isinstance(val, float) and val != val):
                            return None

                        parsed_obj = None
                        # Check if the value is a native dict or a stringified JSON/Python dictionary literal
                        if isinstance(val, dict):
                            parsed_obj = val
                        elif isinstance(val, str):
                            stripped = val.strip()
                            if (
                                stripped.startswith("{") and stripped.endswith("}")
                            ) or (stripped.startswith("[") and stripped.endswith("]")):
                                try:
                                    parsed_obj = ast.literal_eval(stripped)
                                except Exception:
                                    try:
                                        parsed_obj = json.loads(stripped)
                                    except Exception:
                                        pass

                        # LOGIC FIX: If the target column represents an entire object structure (like mapping subfields to a single 'json_data' column)
                        # perform an explicit key lookup. Otherwise, return the clean raw scalar column value directly.
                        if isinstance(parsed_obj, dict):
                            if subkey in parsed_obj:
                                target_val = parsed_obj[subkey]
                                # Handle deep stringified objects nested inside parsed keys cleanly
                                if isinstance(
                                    target_val, str
                                ) and target_val.strip().startswith("{"):
                                    try:
                                        deep_parsed = ast.literal_eval(
                                            target_val.strip()
                                        )
                                        if (
                                            isinstance(deep_parsed, dict)
                                            and subkey in deep_parsed
                                        ):
                                            return deep_parsed[subkey]
                                        return deep_parsed
                                    except Exception:
                                        pass
                                return target_val
                            # If we explicitly mapped to a 'json_data' column but the exact subkey isn't matched,
                            # return None to prevent the entire row's object from being duplicated inside a single string field.
                            if (
                                col_name == "json_data"
                                or "name" in parsed_obj
                                or "description" in parsed_obj
                            ):
                                return None

                        return val

                    # Clean independent text field parsing helper
                    def get_field_string(
                        col_name: str, subkey: str, default_val: str = ""
                    ) -> str:
                        val = extract_field(col_name, subkey)
                        if val is None:
                            return default_val
                        return str(val).strip()

                    # Clean independent array list parsing helper
                    def get_field_list(col_name: str, subkey: str) -> list[str]:
                        val = extract_field(col_name, subkey)
                        if val is None:
                            return []
                        if isinstance(val, list):
                            return [
                                str(item).strip() for item in val if str(item).strip()
                            ]
                        return [
                            item.strip() for item in str(val).split(",") if item.strip()
                        ]

                    # Clean independent integer parsing helper
                    def get_field_int(
                        col_name: str, subkey: str, default_val: int = 0
                    ) -> int:
                        # For direct outer fields like relevance or baseline, map directly if not mapping to 'json_data'
                        val = (
                            extract_field(col_name, subkey)
                            if col_name == "json_data"
                            else row.get(col_name)
                        )
                        if val is None or (isinstance(val, float) and val != val):
                            return default_val
                        try:
                            return int(float(val))
                        except (ValueError, TypeError):
                            return default_val

                    # Resolve metadata attributes separately using exact distinct key extractions
                    name_str = get_field_string(meta_name, "name")
                    desc_str = get_field_string(meta_desc, "description")

                    if desc_str == "" and name_str != "":
                        desc_str = name_str

                    # Assemble the sub-metadata block explicitly flat
                    json_data = {
                        "name": name_str,
                        "description": desc_str,
                        "sponsors": get_field_list(meta_sponsors, "sponsors"),
                        "tagCategories": get_field_list(meta_tags_cat, "tagCategories"),
                        "tagTopics": get_field_list(meta_tags_top, "tagTopics"),
                    }

                    rel_val = get_field_int(
                        relevance_col, "relevance_score", default_val=0
                    )
                    base_val = (
                        get_field_int(
                            baseline_col,
                            "baseline_relevance_score",
                            default_val=rel_val,
                        )
                        if baseline_col != "-- Select Column --"
                        else rel_val
                    )

                    framework_row = {
                        "id": get_field_string(
                            id_col, "id", default_val=f"rec_{idx:03d}"
                        ),
                        "entity_type": get_field_string(
                            entity_col, "entity_type", default_val="episode"
                        ),
                        "json_data": json_data,
                        "keyword": get_field_string(keyword_col, "keyword"),
                        "relevance_score": rel_val,
                        "baseline_relevance_score": base_val,
                    }
                    output_rows.append(framework_row)

                # Write cleanly to target data/sample.jsonl
                target_path = Path("data/sample.jsonl")
                target_path.parent.mkdir(parents=True, exist_ok=True)

                with open(target_path, "w", encoding="utf-8") as f:
                    for r in output_rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

                st.success(
                    f"🎉 Success! Processed records successfully saved to `{target_path}`."
                )
                st.info(
                    "💡 You can now safely run evaluations through the Dashboard or via terminal."
                )

                if len(output_rows) > 0:
                    with st.expander("👀 View Transformed File Snapshot"):
                        st.json(output_rows[0])
