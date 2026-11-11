"""ReaRClean backend.

All three components are implemented: Rule Suggestion, Code Generation and
Rule Ordering.

Run from inside the rearclean directory:

    python service.py
"""

import io
import math
import os

import pandas as pd
from flask import Flask, jsonify, request


from llm import LLM
from profiling import profile_table
from discovery import discover_groups
from generation import generate_rules, ERROR_TYPES
from codegen import generate_code, ALLOWED_IMPORTS
from ordering import run as run_ordering, STRATEGIES

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("REARCLEAN_MAX_UPLOAD_MB", "256")) * 1024 * 1024


# --------------------------------------------------------------------------
# what this backend can do
#
# The frontend builds its controls from this, so adding a task, an ordering or
# an error type is a change here alone.
# --------------------------------------------------------------------------

TASKS = ["Detect", "Impute", "Correct"]


def error_type_list():
    """Reported to the frontend, which builds a checkbox per entry.

    Read from generation.ERROR_TYPES so adding a type is a change there alone.
    """
    return [{"code": k, "label": v["label"], "scope": v["scope"]}
            for k, v in ERROR_TYPES.items()]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def read_table(payload):
    """Accept the table as CSV text. Raises if it is missing or unreadable."""
    if "csv" not in payload:
        raise ValueError("no table supplied: expected 'csv'")
    return pd.read_csv(io.StringIO(payload["csv"]))


def make_llm(payload):
    """Build the model wrapper from the request. Credentials never persist."""
    cfg = payload.get("llm") or {}
    missing = [k for k in ("model", "endpoint", "token") if not cfg.get(k)]
    if missing:
        raise ValueError(f"llm config missing: {', '.join(missing)}")
    return LLM(cfg["model"], cfg["endpoint"], cfg["token"])


def safe(o):
    """Strip NaN and infinity before serialising.

    Flask emits them as bare NaN, which is not valid JSON, and a strict client
    parser rejects the whole response. Generated functions produce them
    routinely on tables with missing numeric values.
    """
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [safe(v) for v in o]
    return o


def fail(e, code=400):
    return jsonify({"error": f"{type(e).__name__}: {e}"}), code


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/capabilities")
def capabilities():
    return jsonify({
        "tasks": TASKS,
        "orderings": list(STRATEGIES),
        "error_types": error_type_list(),
        "allowed_imports": sorted(ALLOWED_IMPORTS),
    })


@app.post("/suggest")
def suggest():
    """Rule Suggestion, in its three substeps."""
    try:
        payload = request.get_json(force=True)
        df = read_table(payload)
        task = payload.get("task", "Detect")
        types = payload.get("error_types") or list(ERROR_TYPES)
        llm = make_llm(payload)
        print(f"/suggest on {len(df)} rows x {len(df.columns)} columns, "
              f"task={task}, types={types}")

        # 1. Column Profiling
        profiles = profile_table(df)

        # 2. Relation Discovery
        groups, evidence = discover_groups(df, profiles, llm)

        # 3. Rule Generation
        rules = generate_rules(df, profiles, groups, evidence, task, llm,
                               error_types=types)

        return jsonify(safe({
            "groups": [list(g) for g in groups],
            "evidence": {str(k): v for k, v in evidence.items()},
            "rules": rules,
            "calls": len(llm.calls),
            "implemented": True,
        }))
    except Exception as e:
        return fail(e)


@app.post("/profile")
def profile():
    """Column Profiling on its own, for a client that wants the profiles
    without paying for rule suggestion. No model is called."""
    try:
        df = read_table(request.get_json(force=True))
        return jsonify(safe({"profiles": profile_table(df), "implemented": True}))
    except Exception as e:
        return fail(e)


@app.post("/generate")
def generate():
    """Code Generation: one Python function per rule."""
    try:
        payload = request.get_json(force=True)
        df = read_table(payload)
        rules = payload.get("rules") or []
        if not rules:
            raise ValueError("no rules supplied")
        task = payload.get("task", "Detect")
        llm = make_llm(payload)
        print(f"/generate {len(rules)} rule(s), task={task}")

        functions = generate_code(df, rules, task, llm,
                                  notries=payload.get("tries", 10),
                                  func_timeout=payload.get("timeout", 60),
                                  functions=payload.get("reuse"))
        return jsonify(safe({"functions": functions,
                             "calls": len(llm.calls),
                             "implemented": True}))
    except Exception as e:
        return fail(e)


@app.post("/run")
def run():
    """Rule Ordering and sandboxed execution."""
    try:
        payload = request.get_json(force=True)
        df = read_table(payload)
        functions = payload.get("functions") or []
        if not functions:
            raise ValueError("no functions supplied")
        task = payload.get("task", "Detect")
        ordering = payload.get("ordering", "Linear")
        llm = make_llm(payload)
        print(f"/run {len(functions)} function(s), task={task}, ordering={ordering}")

        out = run_ordering(df, functions, task, llm, ordering=ordering,
                           func_timeout=payload.get("timeout", 60))
        table = out.pop("table")
        out["csv"] = table.to_csv(index=False)
        out["calls"] = len(llm.calls)
        out["implemented"] = True
        return jsonify(safe(out))
    except Exception as e:
        return fail(e)


if __name__ == "__main__":
    # loopback by default: the service executes generated code, so it should
    # not be reachable from the network unless that is asked for explicitly
    host = os.environ.get("REARCLEAN_HOST", "127.0.0.1")
    port = int(os.environ.get("REARCLEAN_PORT", "8000"))
    print(f"ReaRClean backend on http://{host}:{port}")
    print("Implemented: /profile, /suggest, /generate, /run.")
    app.run(host=host, port=port, threaded=True)