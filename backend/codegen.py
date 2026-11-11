"""Code Generation.

Compiles each rule into a Python function through a prompting loop that retries
until the function executes, and enforces the import restriction before any
generated code is run.

Each function is also executed once against the input frame during generation.
That doubles as a smoke test and records how many cells it affects, which is
the edge weight the Path ordering uses.

The extraction helpers are the originals: extract_function_code in particular
re-attaches imports found anywhere in the response and adds pandas when the
body needs it, which is why a generated function has the modules it refers to.
"""

import ast
import os
import re

from profiling import profile_column, full_profile_text
from discovery import spillover_hints
from execution import run_with_timeout, run_code_with_timeout, normalize_results

#: modules a generated function may use. Anything else is rejected before the
#: function is executed, so nothing reaches the filesystem or the network.
ALLOWED_IMPORTS = {"pandas", "numpy", "re", "math", "datetime", "collections",
                   "itertools", "statistics", "string", "decimal", "fractions"}

DEFAULT_TRIES = int(os.environ.get("REARCLEAN_TRIES", "10"))
DEFAULT_FUNC_TIMEOUT = int(os.environ.get("REARCLEAN_FUNC_TIMEOUT", "60"))

_BANNED_CALLS = ("__import__", "eval(", "exec(", "open(", "compile(",
                 "globals(", "locals(", "setattr(", "getattr(",
                 "read_csv(", "read_table(", "read_json(", "read_excel(",
                 "read_parquet(", "read_pickle(", "read_sql(", "read_html(",
                 "to_csv(", "to_pickle(", ".load(", ".save(")                 

_IMPORT_RULE = ("Use only the Python standard library and pandas, numpy or re; do "
                "not import any package that would need to be installed separately. ")

_NO_WHITELIST = ("If the rule concerns formatting consistency, do not build a "
                 "whitelist of the formats already present in the data: those "
                 "formats conflict with one another, so a value can match one of "
                 "them and still be wrong. ")

#: one entry per task, differing in what the generated function may write
TASK_PROMPTS = {

    "Detect": {
        "template": "Give me a function in python that will take each row as input "
                    "and return a list of the values that confirm this error. Each "
                    "column should have its own entry in the output. " + _NO_WHITELIST
                    + "Flag every value that is not already in the single canonical "
                    "form the rule asks for. " + _IMPORT_RULE +
                    "Example of output: [{{'column': <col_name>, 'index': <index_no>}}]",
        "system": "You are an assistant that will help in error detection in tabular "
                  "data.",
    },

    "Impute": {
        "template": "Give me a function in python that will take each row as input "
                    "and return a list of imputed values. Perform this action only "
                    "on missing values. Each column should have its own entry in the "
                    "output. " + _IMPORT_RULE +
                    "Example of output: [{{'column': <col_name>, 'index': <index_no>, "
                    "'value': <imputed_value>}}]",
        "system": "You are an assistant that will help in data imputation on missing "
                  "values in tabular data.",
    },

    "Correct": {
        "template": "Give me a function in python that will take each row as input, "
                    "find the values that violate this rule, and return the corrected "
                    "value for each of them. Return an entry only for values that "
                    "actually violate the rule and whose corrected value differs from "
                    "the current one; leave every other value alone. Do not return an "
                    "entry when you cannot determine a correct replacement. "
                    + _NO_WHITELIST + "Each column should have its own entry in the "
                    "output. " + _IMPORT_RULE +
                    "Example of output: [{{'column': <col_name>, 'index': <index_no>, "
                    "'value': <corrected_value>}}]",
        "system": "You are an assistant that will help in error correction in tabular "
                  "data. You repair values in place, leaving valid values untouched.",
    },
}


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


def fix_hint(failure):
    """A corrective suggestion for a known failure signature, if we have one."""
    for signature, hint in FIX_HINTS:
        if signature.lower() in failure.lower():
            return hint
    return None


def check_imports(code):
    """Reasons the code is not safe to run, empty if it is.

    Checked before execution, so the restriction stated in the prompt is
    enforced rather than merely requested.
    """
    problems = []
    for mod in re.findall(r"^\s*(?:import|from)\s+([\w.]+)", code, re.M):
        root = mod.split(".")[0]
        if root not in ALLOWED_IMPORTS:
            problems.append(f"imports '{root}', which is not permitted")
    for call in _BANNED_CALLS:
        if call in code:
            problems.append(f"calls {call.rstrip('(')}, which is not permitted")
    return problems


def extract_function_code(response):
    foo_pattern = r"```python(.*?)```"
    match = re.search(foo_pattern, response, re.DOTALL)

    if match is None:
        # no fenced block: take everything from the first def onwards, so a
        # response that omits the fence is still usable
        m2 = re.search(r"(def\s+\w+\s*\(.*)", response, re.DOTALL)
        if m2 is None:
            raise ValueError("no function found in response")
        code_str = m2.group(1).strip()
    else:
        code_str = match.group(1).strip()
    
    #adding missing imports
    # only ones the body does not already carry: the regex also matches imports
    # inside the code block itself, which would otherwise be duplicated
    imports = re.findall(r'^\s*import\b.*$', response, flags=re.MULTILINE)
    for imp in imports:
        if imp.strip() not in code_str:
            code_str = imp + '\n' + code_str
    
    #adding manual imports (pandas)
    if "pd." in code_str and "import pandas" not in code_str:
        code_str = "import pandas as pd\n"+code_str    
        
    return code_str


def extract_function_name(response):
    def_pattern = r"def (.*?)\("
    match = re.search(def_pattern, response, re.DOTALL)
    foo_name = match.group(1).strip()   
    
    return foo_name


def enable_function(function):
    foo_name = extract_function_name(function)         

    namespace = {}
    exec(function, namespace)

    foo = namespace[foo_name]

    return foo


def split_functions(code_str):
    parts = re.split(r'(?=^def\s+)', code_str, flags=re.MULTILINE)
    
    # Extract top imports
    imports = []
    for line in code_str.splitlines():
        if line.startswith(("import", "from")):
            imports.append(line)
        else:
            break
    import_block = "\n".join(imports)

    functions = []
    for part in parts:
        part = part.strip()
        
        if not part.startswith("def"):
            continue
        
        # Prepend imports once
        if import_block:
            part = import_block + "\n\n" + part
        
        functions.append(part)

    return functions


def extract_affected_columns(response):
    col_pattern = "({.*?})"
    match = re.search(col_pattern, response, re.DOTALL)
    
    affected_columns = match.group(1).strip()
    # affected_columns = eval(affected_columns)    
    affected_columns = ast.literal_eval(affected_columns)
    return affected_columns


def generate_code(df, rules, mode, llm, profiles=None, notries=DEFAULT_TRIES,
                  func_timeout=DEFAULT_FUNC_TIMEOUT, functions=None):
    """Compile one function per rule.

    functions, when given, are reused instead of generated: they depend on the
    rules and the task but not on the ordering, so comparing two orderings over
    the same rules costs no further model calls.
    """
    spec = TASK_PROMPTS.get(mode, TASK_PROMPTS["Detect"])
    if profiles is None:
        profiles = {c: profile_column(df[c]) for c in df.columns}
    full_profile = full_profile_text(profiles, df.columns)

    prompt_template = ("We have a dataframe with the following columns:\n\n{}\n\n"
                       "The following rule:{}\n") + spec["template"]

    out = []

    if functions:
        print(f"REUSING {len(functions)} function(s) from a previous run")
        for i, code_str in enumerate(functions):
            rule = rules[i] if i < len(rules) else {}
            values, error = [], None
            try:
                # foo = enable_function(code_str)
                # values = normalize_results(run_with_timeout(foo, df.copy(), func_timeout))
                values = normalize_results(
                    run_code_with_timeout(code_str, df.copy(), func_timeout))
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                print(f"\tReused function {i} failed: {error}")
            out.append({"rule": rule.get("rule", ""),
                        "error_type": rule.get("error_type"),
                        "affected_columns": rule.get("affected_columns", []),
                        "name": extract_function_name(code_str),
                        "code": code_str, "values": values, "weight": len(values),
                        "error": error, "failures": [], "attempts": 0})
        return out

    print("STARTING CODE GENERATION:")
    for nor, entry in enumerate(rules):
        rule = entry.get("rule", "") if isinstance(entry, dict) else str(entry)
        cols = [c for c in (entry.get("affected_columns", [])
                            if isinstance(entry, dict) else []) if c in profiles]
        print(f"\tRule {nor + 1}/{len(rules)}: {rule[:70]}")

        if cols:
            profile_str = full_profile_text(profiles, cols)
            if len(cols) > 1:
                hints = spillover_hints(df, cols)
                if hints:
                    profile_str += ("\nObserved across these columns:\n"
                                    + "\n".join(hints) + "\n")
            print(f"\t\taffected columns: {', '.join(cols)}")
        else:
            profile_str = full_profile
            print("\t\tno affected columns given, sending full profile")

        prompt = prompt_template.format(profile_str, rule)
        seen, code_str, values, error = [], None, [], None
        tries = -1

        for tries in range(notries):
            system = spec["system"]
            if seen:
                system += "\nKNOWN FAILURE MODE:\n"
                for f in seen:
                    system += f"- {f}\n"
                    fix = fix_hint(f)
                    if fix:
                        system += f"  Fix: {fix}\n"
                system += ("The previous attempt failed for the reason(s) above. "
                           "Write a different implementation; repeating the same "
                           "approach will fail again.\n")

            # first try deterministic, then sample so retries actually explore
            temperature = 0.0 if tries == 0 else min(0.2 * tries, 0.8)
            response = llm(prompt, system, temperature=temperature)

            try:
                candidate = extract_function_code(response)
                unsafe = check_imports(candidate)
                if unsafe:
                    raise ValueError("; ".join(unsafe))
                # foo = enable_function(candidate)
                # values = normalize_results(run_with_timeout(foo, df.copy(), func_timeout))
                values = normalize_results(
                    run_code_with_timeout(candidate, df.copy(), func_timeout))
                code_str, error = candidate, None
                break
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                if error not in seen:      # repeats teach the model nothing
                    seen.append(error)
                print(f"\t\tattempt {tries + 1} failed: {error}")

        if code_str is None:
            print(f"\t\tgiving up after {notries} tries, "
                  f"{len(seen)} distinct failure(s)")

        out.append({
            "rule": rule,
            "error_type": entry.get("error_type") if isinstance(entry, dict) else None,
            "affected_columns": cols,
            "name": extract_function_name(code_str) if code_str else None,
            "code": code_str,
            "values": values,
            "weight": len(values),     # cells affected: the Path edge weight
            "error": error,
            "failures": seen,
            "attempts": tries + 1,
        })

    ok = sum(1 for f in out if f["code"])
    print(f"TOTAL: {ok}/{len(out)} function(s) compiled")
    return out