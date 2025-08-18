import streamlit as st
import pandas as pd
from itertools import combinations
from rapidfuzz import fuzz
import swifter
import re

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
    original_unique_names = list(set(names))

    # Single unique name → clearly clean
    if len(original_unique_names) == 1:
        return "Clean"

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
    prepared_names = (
        [_normalize_name(n, active_stopwords) for n in original_unique_names]
        if normalize_text
        else [str(n) for n in original_unique_names]
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

    return "Normalization Issue" if decision else "True Multi-Software"


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(layout="wide")
st.title("🔎 Software Normalization Checker")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data")
    st.dataframe(df.head())

    # Choose the column to analyze for normalization
    st.write("### Choose field to analyze")
    columns_list = list(df.columns)
    default_messy_index = columns_list.index("software_name") if "software_name" in columns_list else 0
    messy_field = st.selectbox(
        "Column to check for normalization issues:",
        options=columns_list,
        index=default_messy_index,
        help="Within each group, values from this column will be compared"
    )

    # User selects grouping fields
    st.write("### Choose fields to group by")
    group_fields = st.multiselect(
        "Select keys for grouping (e.g., source_p_id, target_p_id, relationship_first_seen_date, software_version):",
        options=columns_list,
        default=[
            c for c in [
                "source_p_id",
                "target_p_id",
                "relationship_first_seen_date",
                "software_version",
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
            with st.spinner("Processing groups... this may take a while for large datasets"):
                grouped = (
                    df.groupby(group_fields)
                    .agg({messy_field: lambda x: list(x)})
                    .reset_index()
                )

                # Apply classification with swifter using selected settings
                grouped["issue_type"] = grouped[messy_field].swifter.apply(
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

            st.success("✅ Processing Complete!")

            # Summary
            st.write("### Summary Counts")
            total_groups = len(grouped)
            summary = grouped["issue_type"].value_counts().reset_index()
            summary.columns = ["Issue Type", "Count"]
            summary["Percentage"] = (summary["Count"] / total_groups * 100).round(1)
            st.dataframe(summary)

            # Show results with filtering controls
            st.write("### Detailed Results")
            with st.expander("Filter detailed results"):
                filter_columns = st.multiselect(
                    "Select columns to filter",
                    options=list(grouped.columns),
                )

                filtered_df = grouped.copy()

                for col in filter_columns:
                    series = grouped[col]
                    # Numeric range filter
                    if pd.api.types.is_numeric_dtype(series):
                        min_value = float(series.min()) if pd.notna(series.min()) else 0.0
                        max_value = float(series.max()) if pd.notna(series.max()) else 0.0
                        selected_min, selected_max = st.slider(
                            f"{col} range",
                            min_value=min_value,
                            max_value=max_value,
                            value=(min_value, max_value),
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
                            )
                            filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected_values)]
                        else:
                            query = st.text_input(f"{col} contains")
                            if query:
                                filtered_df = filtered_df[
                                    filtered_df[col].astype(str).str.contains(query, case=False, na=False)
                                ]

                display_columns = st.multiselect(
                    "Columns to display",
                    options=list(filtered_df.columns),
                    default=list(filtered_df.columns),
                )
                st.dataframe(filtered_df[display_columns])

            # Download option
            @st.cache_data
            def convert_df(_df):
                return _df.to_csv(index=False).encode("utf-8")

            csv = convert_df(grouped)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="normalization_analysis.csv",
                mime="text/csv",
            )