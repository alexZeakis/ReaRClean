"""Relation Discovery.

Second substep of Rule Suggestion. Reduces the search space before any rule is
proposed: the model groups columns that might be related, the groups are forced
into a partition, and one measured signal is computed per group.

extract_groups, partition_groups and spillover_hints are the original
implementations, unchanged.
"""

import re

import pandas as pd

from profiling import basic_profile_text

SPILLOVER_TOKEN_MARGIN = 0.5

GROUPING_PROMPT = {
    "template": "We have a dataframe with the following columns:\n\n{}\n\n"
                "Find me groups of columns that might be related. Each column must "
                "appear in exactly one group: the groups must not overlap. A column "
                "that is not related to any other belongs in a group of its own. "
                "Example of output: [('column1', 'column2', 'column3'), "
                "('column4', 'column5'), ...]",
    "system": "You are a helpful assistant in detecting groups of columns that might "
              "be related, based on names or domains.",
}


def extract_groups(response, columns):
    valid = set(columns)
    groups = []

    def _add(cols):
        cols = list(dict.fromkeys(cols))
        if len(cols) >= 2 and cols not in groups:
            groups.append(cols)

    for span in re.findall(r"[\[\(](.*?)[\]\)]", response, re.DOTALL):
        cols = [c for c in re.findall(r"['\"]([^'\"]+)['\"]", span) if c in valid]
        if not cols:  # model omitted the quotes
            cols = [t for t in (x.strip(" \t\n[](){}'\"") for x in span.split(","))
                    if t in valid]
        _add(cols)

    if not groups:  # no brackets at all: fall back to per-line column mentions
        for line in response.splitlines():
            cols = [c for c in valid
                    if re.search(rf"(?<!\w){re.escape(str(c))}(?!\w)", line)]
            _add(cols)

    return groups    

FIX_HINTS = [
    ("iteritems", "use .items() instead of .iteritems(); it was removed in pandas 2"),
    ("append", "DataFrame.append was removed; build a list and use pd.concat, or append to a plain list"),
    ("is_categorical", "use isinstance(dtype, pd.CategoricalDtype) instead"),
    ("get_value", "use .at[] or .loc[] instead of .get_value()"),
    ("ix", "use .loc[] for labels or .iloc[] for positions instead of .ix[]"),
    ("SettingWithCopy", "operate on a copy made with .copy() and assign with .loc[]"),
    ("truth value of a", "compare element-wise and use .any() or .all(), not a bare if on a Series"),
    ("could not convert string to float",
     "guard the conversion with a regex or try/except; not every value is numeric"),
    ("list index out of range",
     "check the match succeeded before indexing, e.g. `m = re.findall(...); if m: m[0]`"),
    ("exceeded", "the previous version was too slow; avoid comparing every pair of "
                 "values, work column-wise with vectorised pandas operations instead"),
    ("KeyError", "only reference columns that exist in the profile above"),
    ("NameError", "define or import everything used; only the standard library, pandas, numpy and re are available"),
    ("ModuleNotFoundError", "that package is not installed; use only the standard library, pandas, numpy or re"),
    ("ImportError", "that package is not installed; use only the standard library, pandas, numpy or re"),
]


def partition_groups(groups, columns):
    """Make groups disjoint so every column belongs to exactly one group.

    Groups are taken in the order the model produced them, which tends to
    put the most confident relations first. A column already claimed by an
    earlier group is dropped from later ones; columns never mentioned end up
    in a group of their own.
    """
    assigned, out = set(), []
    for g in groups:
        cols = [c for c in g if c in set(columns) and c not in assigned]
        if cols:
            out.append(cols)
            assigned.update(cols)
    out += [[c] for c in columns if c not in assigned]
    return out


def spillover_hints(df, group):
    """Flag columns whose values look merged into a neighbour."""
    hints = []
    for a in group:
        if a not in df.columns:
            continue
        empty_a = df[a].isna() | df[a].astype(str).str.strip().eq("")
        if not (0 < empty_a.sum() < len(df)):
            continue
        for b in group:
            if b == a or b not in df.columns:
                continue
            toks = df[b].astype(str).str.split().str.len()
            hi, lo = toks[empty_a].mean(), toks[~empty_a].mean()
            if pd.notna(hi) and pd.notna(lo) and hi > lo + SPILLOVER_TOKEN_MARGIN:
                hints.append(
                    f"\twhen {a} is empty ({int(empty_a.sum())} rows), {b} holds more "
                    f"tokens than usual ({hi:.1f} vs {lo:.1f} on average)"
                )
    return hints


def discover_groups(df, profiles, llm):
    """Group related columns, then measure one cross-column signal per group.

    Returns (groups, evidence). evidence maps a group index to observations
    that a relation exists; it does not say what the relation is, and is passed
    to rule generation as evidence rather than as a rule.
    """
    prompt = GROUPING_PROMPT["template"].format(basic_profile_text(profiles, df.columns))
    response = llm(prompt, GROUPING_PROMPT["system"])
    print(response)

    groups = extract_groups(response, df.columns)
    if not groups:
        print("WARNING: no groups parsed; falling back to one group of all columns")
        groups = [list(df.columns)]
    groups = partition_groups(groups, list(df.columns))
    print(groups)

    # identifiers carry no meaning to relate against, so a relation involving
    # one is an artifact; drop them before the cross-column prompts see a group
    ids = {c for c, p in profiles.items() if "identifier" in str(p.get("role", ""))}
    if ids:
        print(f"dropping identifier column(s) from cross-column groups: {sorted(ids)}")
        groups = [[c for c in g if c not in ids] for g in groups]
        groups = [g for g in groups if g]

    evidence = {}
    for i, g in enumerate(groups):
        if len(g) >= 2:
            hints = spillover_hints(df, g)
            if hints:
                evidence[i] = hints
    return groups, evidence