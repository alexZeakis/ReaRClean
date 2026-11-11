"""ReaRClean frontend.

Two tabs, following the order of the work rather than the order of the
components: Configuration, where the run is set up and the suggested rules are
refined, and Results, where the outcome is exposed.

Run from inside the rearclean directory:

    streamlit run app.py
"""

import io
import os
import json
import time

import pandas as pd
import streamlit as st

from client import Backend, DEFAULT_BASE

st.set_page_config(page_title="ReaRClean", layout="wide")

S = st.session_state
S.setdefault("csv", None)
S.setdefault("df", None)
S.setdefault("rules", [])
S.setdefault("groups", [])
S.setdefault("evidence", {})
S.setdefault("functions", [])
S.setdefault("result", None)


# ==========================================================================
# sidebar
# ==========================================================================

st.sidebar.title("ReaRClean")

base_url = st.sidebar.text_input("Backend URL", DEFAULT_BASE)
api = Backend(base_url)

if not api.health():
    st.sidebar.error("Backend unreachable")
    st.error(f"No backend at {base_url}. Start it with `python service.py`.")
    st.stop()

st.sidebar.success("Backend connected")
caps = api.capabilities()

st.sidebar.subheader("Model")
cfg = {
    "model": st.sidebar.text_input("Model",
                                   os.environ.get("REARCLEAN_MODEL", "gpt-4o-mini")),
    "endpoint": st.sidebar.text_input("Endpoint", "https://api.openai.com/v1/"),
    "token": st.sidebar.text_input("Token",
                                   os.environ.get("OPENAI_API_KEY", ""), type="password"),
}
tries = st.sidebar.number_input(
    "No. tries", 1, 20, 10,
    help="Times a function is regenerated before it is abandoned.")
timeout = st.sidebar.number_input(
    "Function timeout (s)", 5, 600, 60,
    help="Wall-clock cap on each generated function.")


tab_cfg, tab_res = st.tabs(["Configuration", "Results"])


# ==========================================================================
# Configuration
# ==========================================================================

with tab_cfg:
    st.header("Input")
    left, right = st.columns([3, 2])

    with left:
        upload = st.file_uploader("Upload a CSV file", type="csv")
        if upload is not None:
            raw = upload.getvalue().decode("utf-8", errors="replace")
            if raw != S.csv:
                S.csv = raw
                S.df = pd.read_csv(io.StringIO(raw))
                S.rules, S.groups, S.evidence = [], [], {}
                S.functions, S.result = [], None
        if S.df is not None:
            st.dataframe(S.df.head(20), use_container_width=True)
            st.caption(f"{len(S.df):,} rows x {len(S.df.columns)} columns")

    with right:
        task = st.radio(
            "Task", caps["tasks"], horizontal=True,
            help="Detect flags cells and blanks them. Impute fills cells that are "
                 "already empty. Correct repairs values in place.")

        if task == "Detect":
            ordering = "Linear"
            st.caption("Detection is order-invariant, so ordering does not apply.")
        else:
            ordering = st.selectbox("Rule ordering", caps["orderings"])
            if ordering == "Path":
                st.caption("Path selects node-disjoint paths, so some approved "
                           "rules may be left unexecuted.")

        st.markdown("**Error types**")
        chosen = [et["code"] for et in caps["error_types"]
                  if st.checkbox(f"{et['label']}  ({et['scope']})", value=True,
                                 key=f"et_{et['code']}")]

    st.divider()
    st.header("Rules")

    def _clear_rule_widgets():
        """Streamlit keeps keyed widget state, so stale rule text would survive
        a load and overwrite what was just read in."""
        for k in [k for k in S.keys() if k.startswith(("rt", "rc"))]:
            del S[k]

    c1, c2, c3, c4 = st.columns(4)
    do_suggest = c1.button("Generate rules", disabled=S.df is None,
                           use_container_width=True)
    do_add = c2.button("Add rule", disabled=S.df is None, use_container_width=True)

    c3.download_button(
        "Export rules", json.dumps(
            {"task": task, "error_types": chosen, "groups": S.groups,
             "evidence": S.evidence, "rules": S.rules}, indent=2),
        "rearclean_rules.json", "application/json",
        disabled=not S.rules, use_container_width=True,
        help="Save this rule set so an identical run can be reproduced later.")

    with c4.popover("Import rules", disabled=S.df is None,
                    use_container_width=True):
        up = st.file_uploader("Rule set", type="json", key="rules_upload",
                              label_visibility="collapsed")
        if up is not None and st.button("Load", use_container_width=True):
            try:
                data = json.loads(up.getvalue().decode("utf-8"))
                S.rules = data.get("rules", [])
                S.groups = data.get("groups", [])
                S.evidence = data.get("evidence", {})
                S.functions, S.result = [], None
                _clear_rule_widgets()
                st.success(f"Loaded {len(S.rules)} rule(s)")
                st.rerun()
            except Exception as e:
                st.error(f"Could not read that file: {e}")

    if do_suggest:
        if not cfg["token"]:
            st.error("A token is required.")
        elif not chosen:
            st.error("Select at least one error type.")
        else:
            with st.spinner("Suggesting rules"):
                t0 = time.time()
                try:
                    out = api.suggest(S.csv, task, cfg, chosen)
                    S.rules = out.get("rules", [])
                    S.groups = out.get("groups", [])
                    S.evidence = out.get("evidence", {})
                    S.functions, S.result = [], None
                    _clear_rule_widgets()
                    if not out.get("implemented", True):
                        st.warning("Rule suggestion is not implemented yet.")
                    else:
                        st.success(f"{len(S.rules)} rule(s) in {time.time() - t0:.1f}s")
                except Exception as e:
                    st.error(str(e))

    if do_add:
        S.rules.append({"rule": "", "affected_columns": [], "error_type": None})

    if S.groups:
        with st.expander(f"Discovered groups ({len(S.groups)})"):
            for i, g in enumerate(S.groups):
                st.write(f"**{i + 1}.** " + ", ".join(g))
                for line in S.evidence.get(str(i), []):
                    st.caption(line.strip())

    if S.rules:
        labels = {e["code"]: e["label"] for e in caps["error_types"]}
        remove = None
        for i, rule in enumerate(S.rules):
            with st.container(border=True):
                head, drop = st.columns([9, 1])
                head.caption(f"Rule {i + 1} - {labels.get(rule.get('error_type'), 'manual')}")
                if drop.button("Remove", key=f"rm{i}", use_container_width=True):
                    remove = i
                rule["rule"] = st.text_area("Rule", rule.get("rule", ""), key=f"rt{i}",
                                            height=70, label_visibility="collapsed")
                rule["affected_columns"] = st.multiselect(
                    "Affected columns", list(S.df.columns),
                    default=[c for c in rule.get("affected_columns", [])
                             if c in S.df.columns],
                    key=f"rc{i}",
                    help="These determine the dependencies used for ordering.")
        if remove is not None:
            S.rules.pop(remove)
            st.rerun()

        st.divider()
        if st.button("Generate code", type="primary", use_container_width=True):
            valid = [r for r in S.rules if r.get("rule", "").strip()]
            if not valid:
                st.error("No rules to compile.")
            else:
                with st.spinner("Generating code"):
                    try:
                        out = api.generate(S.csv, valid, task, cfg, tries, timeout)
                        S.functions = out.get("functions", [])
                        if not out.get("implemented", True):
                            st.warning("Code generation is not implemented yet.")
                    except Exception as e:
                        st.error(str(e))
                        S.functions = []

                if S.functions:
                    with st.spinner(f"Running under {ordering} ordering"):
                        try:
                            S.result = api.run(S.csv, S.functions, task, cfg,
                                               ordering, timeout)
                            st.success("Done. See the Results tab.")
                        except Exception as e:
                            st.error(str(e))


# ==========================================================================
# Results
# ==========================================================================

with tab_res:
    if not S.functions and not S.result:
        st.info("Nothing to show yet. Generate rules and code in the "
                "Configuration tab.")
    else:
        if S.functions:
            st.header("Generated code")
            for f in S.functions:
                with st.expander(f"{f.get('name') or 'failed'}  -  {f.get('rule','')[:70]}"):
                    if f.get("code"):
                        st.code(f["code"], language="python")
                        st.caption(
                            f"columns: {', '.join(f.get('affected_columns', []))} | "
                            f"cells affected: {f.get('weight', 0)} | "
                            f"attempts: {f.get('attempts', '-')}")
                    else:
                        st.error(f"Not compiled: {f.get('error')}")

        res = S.result
        if res:
            st.divider()
            st.header("Rule coverage")
            cov = res.get("coverage") or {}
            if cov:
                st.caption("Cells each rule affected, per column. A rule firing on "
                           "the wrong column, or on nothing, is visible here.")
                frame = pd.DataFrame(cov).T.fillna(0).astype(int)
                st.dataframe(frame.style.background_gradient(cmap="Blues"),
                             use_container_width=True)
            else:
                st.caption("No cells were affected.")

            if res.get("strategy") != "Linear":
                st.divider()
                st.header("Execution order")
                st.write(" -> ".join(str(n) for n in res.get("order", []))
                         or "nothing ran")
                if res.get("dropped"):
                    st.warning("Left unexecuted by Path ordering: "
                               + ", ".join(str(n) for n in res["dropped"]))
                if res.get("relations"):
                    with st.expander("Induced relations"):
                        st.json(res["relations"])

            if res.get("failures"):
                st.divider()
                st.subheader("Execution failures")
                for f in res["failures"]:
                    st.error(f"{f.get('name')}: {f.get('error')}")

            st.divider()
            st.header("Result")
            out = pd.read_csv(io.StringIO(res["csv"]))
            touched = {(d["column"], d["index"]) for d in res.get("detected", [])}

            def highlight(frame):
                style = pd.DataFrame("", index=frame.index, columns=frame.columns)
                for col, idx in touched:
                    if col in style.columns and idx in style.index:
                        style.at[idx, col] = "background-color: #ffe08a"
                return style

            st.caption(f"{len(touched)} cell(s) affected out of "
                       f"{out.shape[0] * out.shape[1]:,}.")
            st.dataframe(out.style.apply(highlight, axis=None),
                         use_container_width=True)

            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "Download result", res["csv"], "rearclean_output.csv", "text/csv",
                use_container_width=True,
                help="Feed this back as input to chain tasks, for instance "
                     "detection then imputation.")

            # a cell that was already empty shows no change in the result table,
            # so scoring from the table alone undercounts recall; this is the
            # list of cells the rules actually touched
            cells = pd.DataFrame(res.get("detected", []), columns=["index", "column"])
            dl2.download_button(
                "Download flagged cells", cells.to_csv(index=False),
                "rearclean_cells.csv", "text/csv", use_container_width=True,
                help="One row per cell the rules touched, including cells that "
                     "were already empty.")