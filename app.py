import streamlit as st
import pandas as pd
from itertools import combinations
from rapidfuzz import fuzz
import swifter
import re
from concurrent.futures import ThreadPoolExecutor
import io

# ----------------------------
# Core classification helpers
# ----------------------------
def _normalize_name(raw_text: str, stopwords: list[str]) -> str:
    text = str(raw_text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [tok for tok in text.split() if tok and tok not in stopwords]
    return " ".join(tokens).strip()


def classify(
    names,
    threshold: int = 85,
    metric: str = "token_set_ratio",
    decision_rule: str = "any_pair",  # any_pair | all_pairs | proportion_at_least
    proportion: float = 0.5,
    normalize_text: bool = True,
    stopwords: list[str] | None = None,
):
    if not isinstance(names, list):
        names = list(names)

    # Treat both real NULLs and empty/whitespace strings as missing
    def _is_missing(value) -> bool:
        if value is None:
            return True
        try:
            if pd.isna(value):
                return True
        except Exception:
            pass
        if isinstance(value, str) and value.strip() == "":
            return True
        return False

    non_null_values = [v for v in names if not _is_missing(v)]
    has_null = len(non_null_values) != len(names)
    # Distinct non-null values (case-insensitive, trimmed) for decision around NULL behavior
    distinct_non_null = list({str(v).strip().lower() for v in non_null_values})
    num_distinct_non_null = len(distinct_non_null)

    # If there are no non-null values at all, consider it a normalization issue
    if num_distinct_non_null == 0:
        return "Normalization Issue"

    # If there is exactly one non-null value and NULLs present → normalization issue
    if num_distinct_non_null == 1:
        return "Normalization Issue" if has_null else "Clean"

    # Choose similarity function
    metric_map = {
        "token_set_ratio": fuzz.token_set_ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
        "partial_ratio": fuzz.partial_ratio,
        "WRatio": fuzz.WRatio,
    }
    similarity = metric_map.get(metric, fuzz.token_set_ratio)

    # Prepare names
    default_stopwords = ["browser", "server", "database", "db"]
    active_stopwords = default_stopwords if stopwords is None else stopwords
    # Use only non-null original values for similarity checks
    original_unique_non_null_values = list({str(v) for v in non_null_values})
    prepared_names = (
        [_normalize_name(n, active_stopwords) for n in original_unique_non_null_values]
        if normalize_text
        else [str(n) for n in original_unique_non_null_values]
    )

    # If normalization collapses all to one token, it's a normalization issue
    if normalize_text and len(set(prepared_names)) == 1:
        return "Normalization Issue"

    # Compute pairwise scores
    scores = []
    for a, b in combinations(prepared_names, 2):
        score = similarity(a, b)
        scores.append(score)

    if not scores:
        return "Clean"

    # Decide based on rule
    if decision_rule == "all_pairs":
        decision = min(scores) >= threshold
    elif decision_rule == "proportion_at_least":
        fraction = sum(s >= threshold for s in scores) / len(scores)
        decision = fraction >= proportion
    else:  # any_pair
        decision = max(scores) >= threshold

    # Apply NULL-aware overrides
    if has_null and num_distinct_non_null > 1:
        return "Multi-Software + Normalization Issue"

    return "Normalization Issue" if decision else "True Multi-Software"


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(layout="wide")
st.title("🔎 Software Normalization Checker")


@st.cache_data(show_spinner=False)
def load_and_combine_files(file_payloads):
    def _read_single_csv_payload(payload):
        name, content = payload
        try:
            return pd.read_csv(io.BytesIO(content))
        except Exception:
            return pd.DataFrame()

    if not file_payloads:
        return pd.DataFrame()

    if len(file_payloads) == 1:
        return _read_single_csv_payload(file_payloads[0])

    max_workers = min(8, len(file_payloads))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        dfs = list(executor.map(_read_single_csv_payload, file_payloads))
    dfs = [d for d in dfs if not d.empty]
    if len(dfs) == 0:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True, sort=False)


# Build groups with row-level progress updates for very large datasets
def build_groups_with_progress(df: pd.DataFrame, group_fields: list[str], messy_field: str, on_progress=None, approx_steps: int = 200) -> pd.DataFrame:
    from collections import defaultdict

    total_rows = len(df)
    if total_rows == 0:
        return pd.DataFrame(columns=[*group_fields, messy_field])

    groups = defaultdict(list)
    # Choose chunk size to yield about `approx_steps` updates
    chunk_size = max(1, total_rows // max(1, approx_steps))
    processed = 0

    cols_to_use = list(group_fields) + [messy_field]
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunk = df.iloc[start:end][cols_to_use]

        if len(group_fields) == 0:
            # Should not happen due to UI guard, but handle gracefully
            groups[()].extend(chunk[messy_field].tolist())
        else:
            chunk_grouped = chunk.groupby(group_fields)[messy_field].apply(list)
            for key, lst in chunk_grouped.items():
                # Ensure tuple keys for consistent dict usage
                if not isinstance(key, tuple):
                    key = (key,)
                groups[key].extend(lst)

        processed = end
        if on_progress is not None:
            try:
                on_progress(processed, total_rows, len(groups))
            except Exception:
                # Do not fail computation due to UI callback errors
                pass

    if len(groups) == 0:
        return pd.DataFrame(columns=[*group_fields, messy_field])

    keys = list(groups.keys())
    values = list(groups.values())
    keys_df = pd.DataFrame(keys, columns=group_fields)
    keys_df[messy_field] = values
    return keys_df


uploaded_files = st.file_uploader("Upload your CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    # Create a lightweight signature of the current upload set
    file_keys = tuple(sorted((getattr(f, "name", str(i)), getattr(f, "size", None)) for i, f in enumerate(uploaded_files)))

    if (
        "uploaded_df" in st.session_state
        and st.session_state.get("loaded_file_keys") == file_keys
    ):
        df = st.session_state["uploaded_df"]
    else:
        with st.spinner("Loading and combining files..."):
            payloads = [
                (getattr(f, "name", f"file_{i}"), f.getbuffer().tobytes())
                for i, f in enumerate(uploaded_files)
            ]
            df = load_and_combine_files(payloads)
            if df.empty:
                st.error("No valid CSV data loaded.")
                st.stop()
            st.session_state["uploaded_df"] = df
            st.session_state["loaded_file_keys"] = file_keys

    st.write("### Preview of Combined Data")
    st.dataframe(df.head())

    # Session state for persisting results across reruns
    if "grouped_results" not in st.session_state:
        st.session_state["grouped_results"] = None
    # Persist uploaded data for summary stats across reruns
    st.session_state["uploaded_df"] = df

    # Choose the column to analyze for normalization
    st.write("### Choose field to analyze")
    columns_list = list(df.columns)
    default_messy_index = columns_list.index("software_product") if "software_product" in columns_list else 0
    messy_field = st.selectbox(
        "Column to check for normalization issues:",
        options=columns_list,
        index=default_messy_index,
        help="Within each group, values from this column will be compared"
    )

    # User selects grouping fields
    st.write("### Choose fields to group by")
    group_fields = st.multiselect(
        "Select keys for grouping (e.g., source_p_id, target_p_id):",
        options=columns_list,
        default=[
            c for c in [
                "source_p_id",
                "target_p_id",
            ] if c in columns_list
        ]
    )

    # Matching settings
    st.write("### Matching Settings")
    threshold = st.slider(
        "Fuzzy matching threshold",
        min_value=0,
        max_value=100,
        value=85,
        step=1,
        help="Minimum similarity score to consider two names equivalent"
    )
    metric = st.selectbox(
        "Similarity metric",
        options=["token_set_ratio", "token_sort_ratio", "partial_ratio", "WRatio"],
        index=0,
        help=(
            "How to compare two names: token_set_ratio ignores word order and extra words; "
            "token_sort_ratio compares sorted words; partial_ratio checks substrings; "
            "WRatio automatically chooses a robust combo."
        ),
    )
    decision_rule = st.selectbox(
        "Decision rule",
        options=["any_pair", "all_pairs", "proportion_at_least"],
        index=0,
        help=(
            "How to decide within a group: any_pair is sensitive, all_pairs is strict, "
            "proportion_at_least lets you choose a required fraction."
        ),
    )
    proportion = 0.5
    if decision_rule == "proportion_at_least":
        proportion = st.slider(
            "Required fraction of similar pairs",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="If this fraction of pairs exceeds the threshold, mark as normalization issue"
        )
    normalize_text = st.checkbox(
        "Normalize strings (lowercase, remove punctuation, drop generic words)",
        value=True,
    )
    extra_stopwords = st.text_input(
        "Extra generic words to ignore (comma-separated)", value=""
    )
    stopwords = [w.strip().lower() for w in extra_stopwords.split(",") if w.strip()]
    use_swifter = st.checkbox(
        "Accelerate classification with Swifter (chunked progress)",
        value=True,
        help=(
            "Uses swifter to speed up apply in chunks while still updating the Streamlit progress bar. "
            "Disable if you encounter environment-specific issues."
        ),
    )

    with st.expander("What do 'Similarity metric' and 'Decision rule' mean?"):
        st.markdown(
            """
            - **Similarity metric**: how we score how close two names are (0–100).
              - **token_set_ratio**: Ignores word order and extra words. Best default for cases like `Google Chrome` vs `Chrome Browser` or `google-chrome`.
              - **token_sort_ratio**: Compares the same words after sorting. Extra/missing words lower the score more than with set.
              - **partial_ratio**: Good when one name is contained in the other (substring), e.g., `PostgreSQL` vs `PostgreSQL Database`.
              - **WRatio**: Tries several strategies and picks the best. General-purpose and robust.
            - **Decision rule**: how we decide for a whole group of names.
              - **any_pair**: If any pair of names reaches the threshold → flag as a normalization issue (sensitive).
              - **all_pairs**: Only flag if every pair reaches the threshold → strict; mixed groups usually won't trigger.
              - **proportion_at_least**: Flag if at least the chosen fraction of pairs reach the threshold. Use the slider to set the fraction.
            """
        )

    if len(group_fields) == 0:
        st.warning("Please select at least one field to group by.")
    else:
        if st.button("🚀 Run Normalization Analysis"):
            # First, prepare groups with row-level progress
            st.write("Preparing groups...")
            group_progress = st.progress(0)
            group_status = st.empty()

            def on_group_progress(processed_rows: int, total_rows: int, discovered_groups: int):
                pct = int(processed_rows / total_rows * 100) if total_rows else 100
                group_progress.progress(pct)
                group_status.text(
                    f"Reading rows {processed_rows:,}/{total_rows:,} ({pct}%). Groups discovered so far: {discovered_groups:,}"
                )

            grouped = build_groups_with_progress(
                df=df,
                group_fields=group_fields,
                messy_field=messy_field,
                on_progress=on_group_progress,
                approx_steps=200,
            )
            group_status.text("Grouping complete.")

            # Classify with progress updates
            total_groups = len(grouped)
            progress_bar = st.progress(0)
            status_placeholder = st.empty()

            if use_swifter and total_groups > 0:
                # Chunked swifter apply: balanced between speed and UI progress
                desired_updates = 100
                chunk_size = max(1, total_groups // desired_updates)
                processed = 0
                # Ensure column exists
                grouped["issue_type"] = None
                for start in range(0, total_groups, chunk_size):
                    end = min(start + chunk_size, total_groups)
                    subset_idx = grouped.index[start:end]
                    series_subset = grouped.loc[subset_idx, messy_field]
                    result_subset = series_subset.swifter.apply(
                        lambda names: classify(
                            names,
                            threshold=threshold,
                            metric=metric,
                            decision_rule=decision_rule,
                            proportion=proportion,
                            normalize_text=normalize_text,
                            stopwords=stopwords if stopwords else None,
                        )
                    )
                    grouped.loc[subset_idx, "issue_type"] = result_subset.values
                    processed = end
                    pct = int(processed / total_groups * 100)
                    progress_bar.progress(pct)
                    status_placeholder.text(
                        f"Processed {processed}/{total_groups} groups ({pct}%). Remaining: {total_groups - processed}"
                    )
            else:
                # Fallback: per-group loop for progress
                issue_types = []
                for idx, names in enumerate(grouped[messy_field].tolist(), start=1):
                    issue_types.append(
                        classify(
                            names,
                            threshold=threshold,
                            metric=metric,
                            decision_rule=decision_rule,
                            proportion=proportion,
                            normalize_text=normalize_text,
                            stopwords=stopwords if stopwords else None,
                        )
                    )
                    pct = int(idx / total_groups * 100) if total_groups else 100
                    progress_bar.progress(pct)
                    status_placeholder.text(
                        f"Processed {idx}/{total_groups} groups ({pct}%). Remaining: {total_groups - idx}"
                    )
                grouped["issue_type"] = issue_types

            # Flag groups that contain both NULL and non-NULL values for the selected field
            grouped["has_null_and_value"] = grouped[messy_field].apply(
                lambda vals: (
                    any((v is None) or (pd.isna(v)) or (isinstance(v, str) and v.strip() == "") for v in vals)
                    and any(not ((v is None) or (pd.isna(v)) or (isinstance(v, str) and v.strip() == "")) for v in vals)
                )
            )

            st.session_state["grouped_results"] = grouped
            st.session_state["selected_messy_field"] = messy_field
            st.success("✅ Processing Complete!")

    # If results exist, show summary, filters, table, and download regardless of reruns
    results_df = st.session_state.get("grouped_results")
    if results_df is not None:
        # Summary
        st.write("### Summary Counts")
        # Top-line metrics
        total_records = len(st.session_state.get("uploaded_df", results_df))
        selected_messy_field = st.session_state.get("selected_messy_field", None)
        null_records = 0
        if selected_messy_field and selected_messy_field in st.session_state.get("uploaded_df", results_df).columns:
            series_orig = st.session_state.get("uploaded_df", results_df)[selected_messy_field]
            null_mask = series_orig.isna() | (series_orig.astype(str).str.strip() == "")
            null_records = int(null_mask.sum())
        # Mixed NULL/non-NULL groups count for the selected field (treat empty as NULL)
        mixed_count = 0
        if selected_messy_field and selected_messy_field in results_df.columns:
            mixed_count = int(
                results_df[selected_messy_field]
                .apply(lambda vals: (
                    any((v is None) or (pd.isna(v)) or (isinstance(v, str) and v.strip() == "") for v in vals)
                    and any(not ((v is None) or (pd.isna(v)) or (isinstance(v, str) and v.strip() == "")) for v in vals)
                ))
                .sum()
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Total records", total_records)
        m2.metric(
            label=(f"Records without '{selected_messy_field}'" if selected_messy_field else "Records without selected field"),
            value=null_records,
        )
        m3.metric(
            label=(
                f"Groups with NULL and non-NULL in '{selected_messy_field}'" if selected_messy_field else "Groups with NULL and non-NULL"
            ),
            value=mixed_count,
        )
        total_groups = len(results_df)
        summary = results_df["issue_type"].value_counts().reset_index()
        summary.columns = ["Issue Type", "Count"]
        summary["Percentage"] = (summary["Count"] / total_groups * 100).round(1)
        # Append a total row
        total_row = pd.DataFrame([
            {"Issue Type": "Total", "Count": int(summary["Count"].sum()), "Percentage": ""}
        ])
        summary_with_total = pd.concat([summary, total_row], ignore_index=True)
        st.dataframe(summary_with_total)

        # Filters and detailed results (no dropdown)
        st.write("### Detailed Results")
        filter_columns = st.multiselect(
            "Select columns to filter",
            options=list(results_df.columns),
            key="filter_columns",
        )

        filtered_df = results_df.copy()

        for col in filter_columns:
            series = results_df[col]
            # Numeric range filter
            if pd.api.types.is_numeric_dtype(series):
                min_value = float(series.min()) if pd.notna(series.min()) else 0.0
                max_value = float(series.max()) if pd.notna(series.max()) else 0.0
                selected_min, selected_max = st.slider(
                    f"{col} range",
                    min_value=min_value,
                    max_value=max_value,
                    value=(min_value, max_value),
                    key=f"filter_{col}_numrange",
                )
                filtered_df = filtered_df[filtered_df[col].between(selected_min, selected_max)]

            # Datetime range filter
            elif pd.api.types.is_datetime64_any_dtype(series):
                series_dt = pd.to_datetime(series, errors="coerce")
                min_date = series_dt.min().date() if not series_dt.isna().all() else None
                max_date = series_dt.max().date() if not series_dt.isna().all() else None
                if min_date is not None and max_date is not None:
                    start_date, end_date = st.date_input(
                        f"{col} date range",
                        value=(min_date, max_date),
                        key=f"filter_{col}_daterange",
                    )
                    col_dt = pd.to_datetime(filtered_df[col], errors="coerce").dt.date
                    filtered_df = filtered_df[(col_dt >= start_date) & (col_dt <= end_date)]

            # Categorical / text filter
            else:
                unique_values = series.dropna().astype(str).unique().tolist()
                if len(unique_values) <= 100:
                    selected_values = st.multiselect(
                        f"{col} values",
                        options=sorted(unique_values),
                        default=sorted(unique_values),
                        key=f"filter_{col}_values",
                    )
                    filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected_values)]
                else:
                    query = st.text_input(f"{col} contains", key=f"filter_{col}_query")
                    if query:
                        filtered_df = filtered_df[
                            filtered_df[col].astype(str).str.contains(query, case=False, na=False)
                        ]

        display_columns = st.multiselect(
            "Columns to display",
            options=list(filtered_df.columns),
            default=list(filtered_df.columns),
            key="display_columns",
        )
        st.dataframe(filtered_df[display_columns])

        # Download option (full results)
        @st.cache_data
        def convert_df(_df):
            return _df.to_csv(index=False).encode("utf-8")

        csv = convert_df(results_df)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="normalization_analysis.csv",
            mime="text/csv",
        )