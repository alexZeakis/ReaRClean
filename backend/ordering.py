"""Rule Ordering.

Induces a TARGET <- SOURCE relation for each generated function, decouples a
function that writes several columns into single-target functions, then orders
and executes them under one of three strategies.

Linear runs the functions in the order the rules were listed. Path selects
node-disjoint paths of maximum weight over a column graph, which is a selection
rather than a permutation, so some approved rules may be left unexecuted.
Topological sorts the functions themselves, so every approved rule runs and
only the sequence changes.

Detection is order-invariant, since no rule changes what another rule reads, so
ordering applies only to Impute and Correct.
"""

import traceback

import numpy as np
import pandas as pd

from codegen import (enable_function, extract_function_code, extract_function_name,
                     extract_affected_columns, split_functions)
from execution import run_with_timeout, run_code_with_timeout, \
                       normalize_results, relation_sets
from paths import find_path, topological_order

STRATEGIES = ("Linear", "Path", "Topological")

RELATION_PROMPT = {
    "template": "Given the following function, you should give me a list of affected "
                "columns and their dependent columns. \nEXAMPLE:\n#Function\n"
                "def foo(df):\ndf['Column C'] = df['Column A'] + df['Column B'] \n\n"
                "#Output\n{{'Column C': ['Column A', 'Column B']}}\n\nA column can also "
                "be dependent on itself, e.g. if using the average of the column C to "
                "fill missing values of Column C, then {{'Column C': ['Column C']}}\n."
                "Return only this list, no text or comments.\n\nFunction: def foo(df):\n"
                "{}\n\n#Output:",
    "system": "You are a helpful assistant in detecting affected columns from Python "
              "functions.",
}

DECOUPLE_PROMPT = {
    "template": "The following function affects more than one columns. You must create "
                "equivalent separate functions, so each function affects only one "
                "column. \nORIGINAL FUNCTION:\n{}\n",
    "system": "You are an assistant that will help decouple functions that affect more "
              "than one columns.",
}


# --------------------------------------------------------------------------
# Relation Induction
# --------------------------------------------------------------------------

def _induce(code, llm):
    response = llm(RELATION_PROMPT["template"].format(code), RELATION_PROMPT["system"])
    return extract_affected_columns(response)


def _add_edges(edges, affected, values, function_index):
    """One edge per source -> target pair, weighted by the cells affected.

    When induction gives nothing usable we fall back to self-edges on the
    columns the function actually wrote, so a working function is never dropped
    from execution.
    """
    added = False
    for target, sources in (affected or {}).items():
        for source in sources:
            edges.append({"target": target, "source": source,
                          "weight": len(values), "function": function_index})
            added = True
    if not added:
        written = {e["column"] for e in normalize_results(values)
                   if e["column"] is not None}
        print(f"\t\tno relation induced; self-edges for {sorted(written)}")
        for col in written:
            edges.append({"target": col, "source": col,
                          "weight": len(values), "function": function_index})


def analyse_relations(df, functions, llm, func_timeout=60, decouple=True):
    """Induce relations, decoupling any function that writes several columns.

    Returns (functions, edges, relations, weights). The function list may be
    longer than the input, since a decoupled function becomes several.
    """
    print("STARTING RELATION ANALYSIS IN FUNCTIONS:")
    new_funcs, edges, relations, weights = [], [], [], []

    for nof, entry in enumerate(functions):
        code = entry.get("code")
        if not code:
            continue
        print(f"\tFunction {nof + 1}/{len(functions)}: {entry.get('name')}")

        try:
            affected = _induce(code, llm)
            print(f"\t\taffected columns: {affected}")
        except Exception:
            traceback.print_exc()
            continue

        # Path needs one target per function, since its graph is over columns.
        # Topological orders the functions themselves, so a multi-target function
        # is a valid node; splitting it there turns a single atomic edit, such as
        # moving a token from one column to another, into two functions with a
        # write-after-read conflict the sort cannot see.
        if decouple and len(affected) > 1:           # decouple
            print("\t\tdecoupling")
            try:
                response = llm(DECOUPLE_PROMPT["template"].format(code),
                               DECOUPLE_PROMPT["system"])
                parts = split_functions(extract_function_code(response))
            except Exception as e:
                print(f"\t\tdecoupling failed: {type(e).__name__}: {e}")
                parts = []

            for noff, part_code in enumerate(parts):
                try:
                    # foo = enable_function(part_code)
                    # values = normalize_results(
                        # run_with_timeout(foo, df.copy(), func_timeout))
                    values = normalize_results(
                        run_code_with_timeout(part_code, df.copy(), func_timeout))                        
                except Exception as e:
                    print(f"\t\t\tpart {noff} failed: {type(e).__name__}: {e}")
                    continue

                part = dict(entry)
                part.update({"code": part_code,
                             "name": extract_function_name(part_code),
                             "values": values, "weight": len(values)})
                new_funcs.append(part)
                idx = len(new_funcs) - 1

                try:
                    sub = _induce(part_code, llm)
                except Exception:
                    sub = {}
                print(f"\t\t\tpart {noff} affects: {sub}")
                _add_edges(edges, sub, values, idx)
                relations.append(relation_sets(sub, values))
                weights.append(len(values))
            continue

        new_funcs.append(entry)
        idx = len(new_funcs) - 1
        values = entry.get("values") or []
        _add_edges(edges, affected, values, idx)
        relations.append(relation_sets(affected, values))
        weights.append(len(values))

    return new_funcs, edges, relations, weights


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

def run(df, functions, mode, llm, ordering="Linear", func_timeout=60):
    """Order the functions and execute them, writing according to the task."""
    if mode == "Detect":
        ordering = "Linear"          # detection is order-invariant
    if ordering not in STRATEGIES:
        raise ValueError(f"unknown ordering strategy: {ordering}")

    runnable = [f for f in functions if f.get("code")]
    edges, relations, optimal_path, topo_order = [], [], None, None

    if ordering in ("Path", "Topological") and mode in ("Impute", "Correct"):
        runnable, edges, relations, weights = analyse_relations(
            df, runnable, llm, func_timeout, decouple=(ordering == "Path"))
        if ordering == "Topological":
            # topo_order, edges, optimal_path = topological_order(relations, weights)
            topo_order, edges, _ = topological_order(relations, weights)
        else:
            edges = [e for e in edges if e["weight"] > 0]   # drop redundant edges
            optimal_path = find_path(edges)

    # ---------------- Detect: dedup flagged cells, then blank ----------------
    if mode == "Detect":
        flagged, coverage = {}, {}
        for f in runnable:
            per_col = {}
            for e in normalize_results(f.get("values") or []):
                if e["column"] is None:
                    continue
                flagged[(e["index"], e["column"])] = \
                    flagged.get((e["index"], e["column"]), 0) + 1
                per_col[e["column"]] = per_col.get(e["column"], 0) + 1
            coverage[f.get("name") or "rule"] = per_col

        out = df.copy()
        detected = []
        for (idx, col) in flagged:
            if col in out.columns and idx in out.index:
                out.loc[idx, col] = np.nan
                detected.append({"index": int(idx), "column": col})
        print(f"DETECTED: {len(detected)} cell(s)")
        return _result(out, detected, coverage,
                       [f.get("name") for f in runnable], [], edges, relations,
                       [], ordering)

    # ---------------- Impute / Correct: execute in order ----------------
    out = df.copy()
    detected, failures, coverage = [], [], {}

    def _execute(index, column=None):
        f = runnable[index]
        try:
            # foo = enable_function(f["code"])
            # results = run_with_timeout(foo, out, func_timeout)
            results = run_code_with_timeout(f["code"], out, func_timeout)
        except Exception as e:
            # the function ran during generation against the original data, but
            # earlier rules have since changed it; skip rather than abandoning
            # every remaining rule
            msg = f"{type(e).__name__}: {e}"
            print(f"\tFunction {index} failed at execution: {msg}")
            failures.append({"name": f.get("name"), "error": msg})
            return
        per_col = coverage.setdefault(f.get("name") or f"rule_{index}", {})
        for res in normalize_results(results, default_column=column):
            v = res["value"]
            if isinstance(v, (list, dict)) or pd.isna(v):
                continue
            col = column or res["column"]
            if col not in out.columns or res["index"] not in out.index:
                continue
            if mode == "Impute":
                cur = out.loc[res["index"], col]
                # Impute writes only where the cell is already empty, so no
                # existing value is lost
                if not (pd.isna(cur) or str(cur).strip() == ""):
                    continue
            if not pd.api.types.is_numeric_dtype(out[col]) and not isinstance(v, str):
                v = str(v)
            detected.append({"index": int(res["index"]), "column": col})
            per_col[col] = per_col.get(col, 0) + 1
            out.loc[res["index"], col] = v

    if topo_order is not None:                      # every function runs
        order = list(topo_order)
        print(f"TOPOLOGICAL ORDER: {order}")
        for index in order:
            _execute(index)
        dropped = []
    elif optimal_path:                              # a selection, not a permutation
        order = []
        for path in optimal_path:                   # path -> (source, target, info)
            index = path[2]["function"]
            order.append(index)
            _execute(index, column=path[1])
        dropped = [i for i in range(len(runnable)) if i not in set(order)]
        if dropped:
            print(f"PATH ORDERING left {len(dropped)} function(s) unexecuted")
    else:                                           # linear
        order = list(range(len(runnable)))
        for index in order:
            _execute(index)
        dropped = []

    print(f"CHANGED: {len(detected)} cell(s)")
    return _result(out, detected, coverage,
                   [runnable[i].get("name") for i in order],
                   [runnable[i].get("name") for i in dropped],
                   edges, relations, failures, ordering)


def _result(table, detected, coverage, order, dropped, edges, relations,
            failures, ordering):
    return {
        "table": table,
        "detected": detected,
        "coverage": coverage,
        "order": order,
        "dropped": dropped,
        "edges": edges,
        "relations": [{"writes": sorted(r["writes"]), "reads": sorted(r["reads"])}
                      for r in relations],
        "failures": failures,
        "strategy": ordering,
    }