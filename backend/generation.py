"""Rule Generation.

Third substep of Rule Suggestion. Rules are generated one error type at a time:
each entry in ERROR_TYPES has its own prompt, which states the type, the signal
that indicates it and the profile fields relevant to it.

Single-column types get one call with every column's profile. Cross-column
types get one call per group of related columns, with coarse profile
information only, since a relation is argued from what the columns mean rather
than from their value distributions.

Adding an error type is a change to ERROR_TYPES alone: the service reports it
and the frontend builds its checkbox from that report.
"""

import re

from profiling import full_profile_text, basic_profile_text, profile_text_for

# --------------------------------------------------------------------------
# shared prompt fragments
# --------------------------------------------------------------------------

_SINGLE_SYSTEM = ("You are a helpful assistant in generating single-column data "
                  "quality rules from column profiles. You only propose rules "
                  "supported by evidence in the profile.")

_CROSS_SYSTEM = ("You are a helpful assistant in generating rules that relate "
                 "several columns to each other. You state relationships as "
                 "conditions every row should satisfy.")

_CONSTRAINTS = (
    "\n\nFollow these constraints:\n"
    "- Every rule must be justified by a statistic in the profile above. Do not "
    "propose rules based on general domain assumptions alone.\n"
    "- If no column shows this error type, return nothing at all. Never write a "
    "rule stating that no action is required or that a check is unnecessary.\n"
    "- Never derive a threshold from the observed data and then test the data "
    "against it. Ranges and distinct value lists in the profile were computed "
    "from these rows, so no row can fall outside them.\n"
    "- Respect any stated role of a column: checks the profile says do not apply "
    "must not be proposed.\n"
    "- Few well-grounded rules are better than many speculative ones.\n\n"
)

_SINGLE_FORMAT = ("Format each rule as 'Rule N [column_name]: <rule>', naming the "
                  "single column it applies to.")

#: Detect only identifies cells, so a rule that states the condition is enough.
#: Impute and Correct write a value, so a rule that does not say where that
#: value comes from leaves the code generator to invent one, which is where
#: constant fills come from.
_REPAIR_CLAUSE = {
    "Detect": "",
    "Impute": "\n- State where the replacement value comes from: another column "
              "in the same row, a statistic of this column such as its median or "
              "most frequent value, or a fixed rule. A rule that only identifies "
              "the cell without saying how to fill it is not usable.",
    "Correct": "\n- State where the replacement value comes from: another column "
               "in the same row, a statistic of this column such as its median or "
               "most frequent value, or a fixed rule. A rule that only identifies "
               "the cell without saying how to repair it is not usable.",
}

_CROSS_FORMAT = ("Format each rule as 'Rule N [column_a, column_b]: <rule>', "
                 "naming every column it involves.")


# --------------------------------------------------------------------------
# the error types
#
# scope:  "single" -> one call, full profiles
#         "cross"  -> one call per group, coarse profiles
# fields: profile fields passed to the model for this type
# --------------------------------------------------------------------------

ERROR_TYPES = {

    "FI": {
        "label": "Formatting inconsistency",
        "scope": "single",
        "fields": ("type", "patterns", "competing_formats", "format_consistency",
                   "examples", "values", "role"),
        "system": _SINGLE_SYSTEM,
        "template":
            "We have a dataframe whose columns show the following profiles:\n\n{profile}\n\n"
            "Generate rule(s) about FORMATTING INCONSISTENCY, where the same concept "
            "is written in several different ways within one column, regarding {mode}, "
            "i.e. the task of {task}.\n\n"
            "The signal for this error type is a column whose values are split across "
            "competing patterns rather than sharing one form."
            + _CONSTRAINTS +
            "- State the single canonical form all values should take, and treat every "
            "value not already in that form as an error. The patterns listed in the "
            "profile are rival encodings, not accepted formats: a value matching one of "
            "them may still be wrong. Do not build a whitelist of observed formats.\n"
            "- Do not propose missing-value, typo, outlier or cross-column rules here.\n\n"
            + _SINGLE_FORMAT,
    },

    "MV": {
        "label": "Missing values",
        "scope": "single",
        "fields": ("type", "missing", "placeholders", "distinct", "examples",
                   "values", "role"),
        "system": _SINGLE_SYSTEM,
        "template":
            "We have a dataframe whose columns show the following profiles:\n\n{profile}\n\n"
            "Generate rule(s) about MISSING VALUES, regarding {mode}, i.e. the task of "
            "{task}.\n\n"
            "The signal for this error type is a non-zero null count, or placeholder "
            "tokens such as N/A, unknown or none stored as ordinary text."
            + _CONSTRAINTS +
            "- Treat a placeholder token as missing, and say which token the column uses.\n"
            "- Do not propose formatting, typo, outlier or cross-column rules here.\n\n"
            + _SINGLE_FORMAT,
    },

    "TP": {
        "label": "Typos",
        "scope": "single",
        "fields": ("type", "distinct", "values", "patterns", "examples", "role"),
        "system": _SINGLE_SYSTEM,
        "template":
            "We have a dataframe whose columns show the following profiles:\n\n{profile}\n\n"
            "Generate rule(s) about TYPOS, regarding {mode}, i.e. the task of {task}.\n\n"
            "The signal for this error type is a low-cardinality column holding values "
            "that are near-duplicates of one another, for example one frequent value and "
            "a rare variant differing by a character, a doubled letter or a transposition."
            + _CONSTRAINTS +
            "- Name the canonical value and say that rare near-duplicates of it should be "
            "mapped onto it. Do not propose a rule for a column whose values are free "
            "text, where near-duplicates are expected.\n"
            "- Do not propose formatting, missing-value, outlier or cross-column rules "
            "here.\n\n"
            + _SINGLE_FORMAT,
    },

    "OT": {
        "label": "Outliers",
        "scope": "single",
        "fields": ("type", "typical_range", "examples", "values", "role"),
        "system": _SINGLE_SYSTEM,
        "template":
            "We have a dataframe whose columns show the following profiles:\n\n{profile}\n\n"
            "Generate rule(s) about OUTLIERS, regarding {mode}, i.e. the task of {task}.\n\n"
            "The signal for this error type is a numeric column holding values far "
            "outside the range its other values occupy."
            + _CONSTRAINTS +
            "- State the admissible range from what the quantity means in the real world, "
            "not from the observed values, since every observed value lies inside its own "
            "range by construction.\n"
            "- Do not propose a rule for a column the profile marks as an identifier, "
            "where magnitude carries no meaning.\n"
            "- Do not propose formatting, missing-value, typo or cross-column rules here.\n\n"
            + _SINGLE_FORMAT,
    },

    "SP": {
        "label": "Value spillover",
        "scope": "cross",
        "fields": ("type", "missing", "distinct", "role", "examples", "values"),
        "system": _CROSS_SYSTEM,
        "template":
            "We have a dataframe with the following related columns:\n\n{profile}\n"
            "{evidence}\n"
            "Generate rule(s) about VALUE SPILLOVER, where part of one column's value "
            "has been stored in another column, regarding {mode}, i.e. the task of "
            "{task}.\n\n"
            "The signal for this error type is one column holding more content than "
            "usual on exactly the rows where a neighbouring column is empty."
            + _CONSTRAINTS +
            "- State which column the fragment belongs in and how it should be moved.\n"
            "- Do not propose a relationship that only holds because one column is an "
            "identifier.\n"
            "- Do not propose single-column rules here.\n\n"
            + _CROSS_FORMAT,
    },

    "DV": {
        "label": "Derived value",
        "scope": "cross",
        "fields": ("type", "missing", "distinct", "role", "examples", "values"),
        "system": _CROSS_SYSTEM,
        "template":
            "We have a dataframe with the following related columns:\n\n{profile}\n"
            "{evidence}\n"
            "Generate rule(s) about DERIVED VALUES, where one column should be "
            "computable from the others but disagrees with them, regarding {mode}, "
            "i.e. the task of {task}.\n\n"
            "Work out what should hold between these columns from their names, their "
            "example values and what such values mean in the real world. This includes "
            "a value computed from others, and an ordering that must never be broken, "
            "such as one date or quantity that cannot exceed another."
            + _CONSTRAINTS +
            "- State the relationship as a condition every row should satisfy, then say "
            "which column is authoritative when it is broken and how the disagreement "
            "should be resolved.\n"
            "- Do not propose a relationship that only holds because one column is an "
            "identifier.\n"
            "- Do not propose single-column rules here.\n\n"
            + _CROSS_FORMAT,
    },
}

SINGLE_TYPES = [k for k, v in ERROR_TYPES.items() if v["scope"] == "single"]
CROSS_TYPES = [k for k, v in ERROR_TYPES.items() if v["scope"] == "cross"]

TASK_LABEL = {
    "Detect": "Detecting errors",
    "Impute": "Imputing values on missing data",
    "Correct": "Correcting erroneous values in place",
}


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def _clean(text):
    """Strip markdown emphasis and collapse whitespace.

    Models wrap rules in bold, which otherwise ends up inside the rule text and
    then inside the code-generation prompt.
    """
    text = text.strip().replace("\n", " ")
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" *-:")


def parse_rules(response, default_columns, valid_columns):
    """Pull 'Rule N [cols]: text' entries, falling back to 'Rule N: text'."""
    valid = set(valid_columns)
    out = []
    tagged = re.findall(
        r"Rule\s*\d+\s*\[([^\]]*)\]\s*:\s*(.*?)(?=Rule\s*\d+\s*[\[:]|\Z)",
        response or "", flags=re.DOTALL)
    if tagged:
        for cols, text in tagged:
            text = _clean(text)
            if not text:
                continue
            cs = [c.strip().strip("'\"` ") for c in cols.split(",")]
            cs = [c for c in cs if c in valid]
            out.append({"rule": text, "affected_columns": cs or list(default_columns)})
        return out

    plain = re.findall(
        r"(?:\d+\.\s*)?Rule\s+\d+:\s*(.*?)(?=(?:\d+\.\s*)?Rule\s+\d+:|\Z)",
        response or "", flags=re.DOTALL)
    for text in plain:
        text = _clean(text)
        if text:
            out.append({"rule": text, "affected_columns": list(default_columns)})
    return out


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def generate_rules(df, profiles, groups, evidence, mode, llm, error_types=None):
    """Propose rules, one error type at a time.

    error_types selects which entries of ERROR_TYPES to run; None means all,
    which is the interface default.
    """
    types = list(error_types) if error_types else list(ERROR_TYPES)
    task = TASK_LABEL.get(mode, TASK_LABEL["Detect"])
    rules = []

    # ---- single-column types: one call each, full profiles ----
    for code in [t for t in types if t in SINGLE_TYPES]:
        spec = ERROR_TYPES[code]
        print(f"STARTING RULE GENERATION: {spec['label']}")
        text = (profile_text_for(profiles, list(df.columns), spec["fields"])
                or full_profile_text(profiles, df.columns))
        prompt = spec["template"].format(profile=text, mode=mode, task=task,
                                         evidence="") + _REPAIR_CLAUSE.get(mode, "")
        response = llm(prompt, spec["system"])
        found = parse_rules(response, list(df.columns), df.columns)
        if not found:
            print(f"\tno rules parsed. Raw response:\n\t"
                  + (response or "(empty)")[:800].replace("\n", "\n\t"))
        for r in found:
            r["error_type"] = code
        print(f"\t{len(found)} rule(s)")
        rules += found

    # ---- cross-column types: one call per group, coarse profiles ----
    cross = [t for t in types if t in CROSS_TYPES]
    for gi, group in enumerate(groups):
        cols = [c for c in group if c in profiles]
        if len(cols) < 2:
            continue                       # a lone column has nothing to relate to
        ev = ""
        if evidence.get(gi):
            ev = "\nObserved across these columns:\n" + "\n".join(evidence[gi]) + "\n"
        for code in cross:
            # spillover is argued from the token-count observation, so without
            # that evidence there is nothing for the prompt to work from
            if code == "SP" and not evidence.get(gi):
                print(f"skipping Value spillover for group {gi + 1}: no measured signal")
                continue
            spec = ERROR_TYPES[code]
            print(f"STARTING RULE GENERATION: {spec['label']} "
                  f"(group {gi + 1}/{len(groups)}: {', '.join(cols)})")
            text = (profile_text_for(profiles, cols, spec["fields"])
                    or basic_profile_text(profiles, cols))
            prompt = spec["template"].format(profile=text, mode=mode, task=task,
                                             evidence=ev) + _REPAIR_CLAUSE.get(mode, "")
            response = llm(prompt, spec["system"])
            # the valid set is the group, not the whole table: a cross-column call
            # is scoped to these columns, so a rule naming others is out of scope
            # and falls back to the group
            found = parse_rules(response, cols, cols)
            if not found:
                print(f"\tno rules parsed. Raw response:\n\t"
                      + (response or "(empty)")[:800].replace("\n", "\n\t"))
            for r in found:
                r["error_type"] = code
                r["group"] = gi
            print(f"\t{len(found)} rule(s)")
            rules += found

    print(f"TOTAL: {len(rules)} rule(s)")
    return rules