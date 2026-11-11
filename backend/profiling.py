"""Column Profiling.

First substep of Rule Suggestion. Summarises each column into a compact
profile the model can read. No model is called here.

profile_column is the original implementation, unchanged.
"""

import re

import numpy as np
import pandas as pd


def mean(s):
    return sum(s) / len(s) if len(s) else 0

#: profiling thresholds, collected here so none is buried in the body
TOP_VALUES = 5                  # most frequent values listed for a categorical
MAX_CARDINALITY = 20            # distinct values under which a column is categorical
N_EXAMPLES = 3                  # example values shown for a free-text column
HIGH_MISSING = 0.05             # missing share above which the profile says HIGH
CATEGORICAL_RATIO = 0.05        # distinct/rows under which a wider column is still categorical
CATEGORICAL_MAX = 100           # but never above this many distinct values
IQR_FENCE = 1.5                 # interquartile fence for candidate outliers
MIN_FORMAT_CONSISTENCY = 0.9    # dominant-pattern share below which formats compete
NUMERIC_TEXT_SHARE = 0.8        # share of numeric-looking values to call a column numeric text


def profile_column(s, k=TOP_VALUES, max_card=MAX_CARDINALITY, examples=N_EXAMPLES):
    """Return a short profile dict for one column."""
    SENTINELS = {"n/a", "na", "null", "none", "unknown", "-", "?", "nan", ""}
    n_total = len(s)
    s = s.dropna()
    n_miss = n_total - len(s)
    miss = f"{n_miss}/{n_total}"
    if n_total and n_miss / n_total > HIGH_MISSING:
        miss += f" ({100 * n_miss / n_total:.0f}% — HIGH, these are errors unless the column is genuinely optional)"
    p = {
        "type": str(s.dtype) if len(s) else "empty",
        "missing": miss,
    }
    if len(s) == 0:
        return p

    n_sent = s.astype(str).str.strip().str.lower().isin(SENTINELS).sum()
    if n_sent:
        p["placeholders"] = f"{n_sent}/{n_total} (missing values written as text)"

    name = str(s.name).lower()
    is_identifier = (
        bool(re.search(r"(^|_)(id|ids|key|keys|code|uuid|guid|index|idx)($|_)", name))
        or name.endswith("_id") or name == "id"
        or (pd.api.types.is_numeric_dtype(s) and s.nunique() == len(s))
    )
    if is_identifier:
        p["role"] = "identifier (check only for missing values or formatting; magnitude carries no meaning, so outlier and range checks do not apply)"

    if pd.api.types.is_numeric_dtype(s) and not is_identifier:
        p["type"] = str(s.dtype)
        p["range"] = f"{s.min():g} to {s.max():g}"
        p["mean"] = f"{s.mean():.4g}"
        p["std"] = f"{s.std():.4g}" if len(s) > 1 else "n/a"
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        p["typical_range"] = f"{q1 - IQR_FENCE * iqr:g} to {q3 + IQR_FENCE * iqr:g} (values outside this are candidate outliers)"
        if (s == s.round()).all():
            p["integral"] = "yes"

    elif pd.api.types.is_numeric_dtype(s):
        p["type"] = str(s.dtype)
        p["range"] = f"{s.min():g} to {s.max():g}"

    elif pd.api.types.is_datetime64_any_dtype(s):
        p["range"] = f"{s.min()} to {s.max()}"
        p["distinct"] = s.nunique()

    else:
        s = s.astype(str)
        u = s.nunique()
        p["distinct"] = u
        if u <= max_card or (u / len(s) < CATEGORICAL_RATIO and u <= CATEGORICAL_MAX):
            p["type"] = "categorical"
            vc = s.value_counts().head(k)
            p["values"] = ", ".join(f"{v} ({c})" for v, c in vc.items())
        else:
            p["type"] = "string"
            lens = s.str.len()
            p["length"] = f"{lens.min()}-{lens.max()}"
            p["examples"] = ", ".join(s.head(examples))

        pat = (s.str.replace(r"\d+", "9", regex=True)
                .str.replace(r"[A-Za-z]", "a", regex=True))
        top_pat = pat.value_counts()
        numeric_text = s.str.match(r"^\s*[-+]?\d*\.?\d+").mean() > NUMERIC_TEXT_SHARE
        # only meaningful for structured values; free text always varies
        if re.search(r"[^a ]", top_pat.index[0]):
            cons = top_pat.iloc[0] / len(s)
            if cons < MIN_FORMAT_CONSISTENCY:
                rivals = [s[pat == ptn].iloc[0] for ptn in top_pat.head(3).index]
                p["competing_formats"] = " vs ".join(repr(v) for v in rivals) + \
                    f" ({len(top_pat)} distinct formats in all)"
                p["format_consistency"] = (
                    f"{cons:.2f} (LOW). The values above are rival encodings of the "
                    f"same concept, NOT a list of accepted formats. At most one can be "
                    f"canonical, so a value matching one of them may still be an error. "
                    f"Do not treat the observed formats as a whitelist."
                )
                if numeric_text:
                    p["format_consistency"] += (
                        " These are numbers carrying inconsistent unit text, so the "
                        "canonical form is the numeric value alone and every value "
                        "carrying extra text is an error."
                    )
            else:
                p["patterns"] = " | ".join(
                    f"{ptn} ({100 * cnt / len(s):.0f}%)"
                    for ptn, cnt in top_pat.head(3).items()
                )
                p["format_consistency"] = f"{cons:.2f} (high)"
        elif p["type"] == "string":
            p["format"] = ("free text (value patterns are not a meaningful constraint; "
                           "near-duplicate or typo detection over free text over-flags "
                           "badly and needs a stated similarity threshold to be usable)")
        if numeric_text:
            p["note"] = "numeric values stored as text"
    return p


def profile_table(df, **kw):
    """Profile every column. Returns {column: {field: value}}."""
    return {col: profile_column(df[col], **kw) for col in df.columns}


#: fields shown when only coarse information is wanted
BASIC_FIELDS = ("type", "missing", "distinct", "role", "examples", "values",
                "range")   # numeric columns keep their values under range, not values

#: these carry long explanatory prose, trimmed when shown coarsely
_TERSE = ("missing", "role")


def full_profile_text(profiles, cols=None):
    """Every profile field. Used by the single-column prompts."""
    out = ""
    for col in (cols if cols is not None else profiles):
        if col not in profiles:
            continue
        fields = " | ".join(f"{k} -> {v}" for k, v in profiles[col].items())
        out += f"\t{col}: {fields}\n"
    return out


def basic_profile_text(profiles, cols=None):
    """Name, type and coarse counts only.

    Used for grouping and for the cross-column prompts, where a relation is
    argued from what the columns mean rather than from their distributions.
    """
    out = ""
    for col in (cols if cols is not None else profiles):
        if col not in profiles:
            continue
        fields = []
        for k, v in profiles[col].items():
            if k not in BASIC_FIELDS:
                continue
            v = str(v)
            if k in _TERSE:
                v = v.split(" (")[0].split(" \u2014")[0]
            fields.append(f"{k} -> {v}")
        out += f"\t{col}: {' | '.join(fields)}\n"
    return out


def profile_text_for(profiles, cols, fields):
    """Only the named fields, so each error type is given just what it needs."""
    out = ""
    for col in cols:
        if col not in profiles:
            continue
        shown = [f"{k} -> {v}" for k, v in profiles[col].items() if k in fields]
        if shown:
            out += f"\t{col}: {' | '.join(shown)}\n"
    return out