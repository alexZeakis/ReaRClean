"""Sandboxed execution.

Generated code is executed, so this is the only place that runs it: each
function runs in a separate process under a wall-clock cap, and whatever shape
it returns is normalised before anything is written.

All three routines are the original implementations, unchanged.
"""

import multiprocessing
import queue
import re
import threading
import traceback
import warnings

import numpy as np
import pandas as pd

# generated code triggers these constantly; everything else stays visible
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")


def run_with_timeout(foo, frame, seconds=60):
    """Run a generated function, giving up after `seconds`.

    Generated code can be accidentally quadratic — a fuzzy near-duplicate rule
    over a few thousand distinct free-text values is tens of millions of string
    comparisons — and would otherwise hang the whole run. A forked child can
    actually be killed; the thread fallback can only be abandoned.
    """
    try:
        ctx = multiprocessing.get_context('fork')
    except (ValueError, RuntimeError):
        ctx = None

    if ctx is not None:
        q = ctx.Queue()

        def worker():
            try:
                q.put(('ok', foo(frame)))
            except Exception as e:
                q.put(('err', f'{type(e).__name__}: {e}'))

        p = ctx.Process(target=worker, daemon=True)
        p.start()
        try:
            status, payload = q.get(timeout=seconds)
        except queue.Empty:
            p.terminate()
            p.join(5)
            raise TimeoutError(f'exceeded {seconds}s and was terminated')
        finally:
            if p.is_alive():
                p.terminate()
            p.join(5)
        if status == 'err':
            raise RuntimeError(payload)
        return payload

    # no fork available: run in a thread we can abandon but not kill
    box = {}

    def worker():
        try:
            box['ok'] = foo(frame)
        except Exception as e:
            box['err'] = f'{type(e).__name__}: {e}'

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError(f'exceeded {seconds}s (left running in the background)')
    if 'err' in box:
        raise RuntimeError(box['err'])
    return box.get('ok')

def run_code_with_timeout(code, frame, seconds=60):
    """Compile and call generated code inside the sandboxed child.

    exec() of the generated string runs module-level statements, so it belongs
    in the child under the timeout rather than in the parent.
    """
    def call(df):
        from codegen import enable_function     # imported here to avoid a cycle
        return enable_function(code)(df)
    return run_with_timeout(call, frame, seconds)

def normalize_results(results, default_column=None):
    """Expand LLM output entries so index, column and value are scalars.

    Generated functions sometimes return {'column': [...], 'index': [...]}
    instead of one entry per cell. Parallel value lists are paired with the
    indices; a scalar value is reused for every index in the entry.
    """
    out = []
    # a generated function may return a DataFrame, Series, ndarray or a bare
    # dict instead of a list; `results or []` truth-tests those and raises
    if results is None:
        results = []
    elif isinstance(results, dict):
        results = [results]
    elif isinstance(results, pd.DataFrame):
        results = results.to_dict('records')
    elif isinstance(results, (pd.Series, np.ndarray)):
        results = results.tolist()
    elif not isinstance(results, (list, tuple, set)):
        try:
            results = list(results)
        except TypeError:
            print(f'\tIgnoring unusable function output of type '
                  f'{type(results).__name__}')
            results = []
    for res in results:
        if not isinstance(res, dict):
            continue
        idxs = res.get('index')
        idxs = idxs if isinstance(idxs, list) else [idxs]
        cols = res.get('column', default_column)
        cols = cols if isinstance(cols, list) else [cols]
        vals = res.get('value')
        for n, i in enumerate(idxs):
            v = (vals[n] if n < len(vals) else None) if isinstance(vals, list) else vals
            for c in cols:
                out.append({'index': i,
                            'column': c if c is not None else default_column,
                            'value': v})
    return out


def relation_sets(affected_columns, values):
    """Columns a function writes and reads, for dependency ordering."""
    written = {e['column'] for e in normalize_results(values)
               if e['column'] is not None}
    if affected_columns:
        writes = set(affected_columns.keys()) | written
        reads = {s for ss in affected_columns.values() for s in ss}
    else:
        writes, reads = written, set(written)
    return {'writes': writes, 'reads': reads}