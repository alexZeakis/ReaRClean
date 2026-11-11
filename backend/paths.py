"""Graph routines for rule ordering.

The original implementations, unchanged apart from removing a rendering
helper that belonged in the frontend. find_path selects node-disjoint paths
of maximum weight over a column graph; topological_order sorts the functions
themselves so that every one of them still runs.
"""

import networkx as nx
import os

def build_graph(edges):
    # MultiDiGraph: two functions can share the same source->target pair, and a
    # DiGraph would silently keep only the last one
    G = nx.MultiDiGraph()
    for i, e in enumerate(edges):
        G.add_edge(
            e["source"],
            e["target"],
            weight=e["weight"],
            function=e["function"],
            id=i,
        )
    return G

def best_path_from_node(G, start):
    best_weight = float("-inf")
    best_path = []

    def dfs(node, visited, current_weight, path):
        nonlocal best_weight, best_path

        visited.add(node)

        # Explore neighbors
        for _, neighbor, data in G.out_edges(node, data=True):
            if neighbor in visited:
                continue  # avoid cycles

            dfs(
                neighbor,
                visited,
                current_weight + data.get("weight", 0),
                path + [(node, neighbor, data)]
            )

        # Update best when no more expansion (leaf or dead-end)
        if current_weight > best_weight:
            best_weight = current_weight
            best_path = path

        visited.remove(node)

    dfs(start, set(), 0, [])
    return best_path, best_weight

# def all_paths_from_node(G, start):
#     all_paths = []

#     def dfs(node, visited, path):
#         visited.add(node)

#         extended = False

#         for _, neighbor, data in G.out_edges(node, data=True):
#             if neighbor in visited:
#                 continue  # avoid cycles

#             extended = True
#             dfs(
#                 neighbor,
#                 visited,
#                 path + [(node, neighbor, data)]
#             )

#         # If no further expansion → this is a final path
#         if not extended:
#             all_paths.append(path)

#         visited.remove(node)

#     dfs(start, set(), [])
#     return all_paths

def all_paths_from_node(G, start):
    all_paths = []

    def dfs(node, path_nodes, path_edges):
        extended = False

        for _, neighbor, data in G.out_edges(node, data=True):

            # parallel edges are distinct rules, so identify by edge id
            if data.get("id") in {d.get("id") for _, _, d in path_edges}:
                continue

            new_nodes = path_nodes + [neighbor]
            new_edges = path_edges + [(node, neighbor, data)]

            dfs(neighbor, new_nodes, new_edges)
            extended = True

        if not extended:
            all_paths.append(path_edges)

    dfs(start, [start], [])
    return all_paths

EXACT_SEARCH_LIMIT = int(os.environ.get("REARCLEAN_EXACT_SEARCH_LIMIT", "20"))

def max_disjoint_sets(sets_with_weights, exact_limit=EXACT_SEARCH_LIMIT):
    # attach index
    indexed = [(i, nodes, weight) for i, (nodes, weight) in enumerate(sets_with_weights)]

    # Maximum-weight set packing is NP-hard and the "skip" branch never prunes,
    # so the exhaustive search is 2^n. Run it only for small inputs; otherwise
    # take the standard greedy approximation (heaviest first, keep if disjoint).
    if len(indexed) > exact_limit:
        print(f'  NOTE: {len(indexed)} candidate sets exceeds the exact-search limit '
              f'({exact_limit}); using greedy selection by weight')
        best_solution, best_weight, used = [], 0, set()
        for idx, nodes, weight in sorted(indexed, key=lambda t: -t[2]):
            if nodes.isdisjoint(used):
                best_solution.append((idx, nodes, weight))
                used |= set(nodes)
                best_weight += weight
        return best_solution, best_weight

    best_weight = 0
    best_solution = []

    # explicit stack instead of recursion: depth used to equal len(indexed),
    # which overflows once the graph yields many candidate paths
    stack = [(0, frozenset(), 0, [])]
    while stack:
        i, used_nodes, current_weight, chosen = stack.pop()

        if i == len(indexed):
            if current_weight > best_weight:
                best_weight = current_weight
                best_solution = chosen
            continue

        idx, nodes, weight = indexed[i]

        # skip
        stack.append((i + 1, used_nodes, current_weight, chosen))

        # take
        if nodes.isdisjoint(used_nodes):
            stack.append((
                i + 1,
                used_nodes | frozenset(nodes),
                current_weight + weight,
                chosen + [(idx, nodes, weight)]
            ))

    return best_solution, best_weight

# if __name__ == "__main__":

#     edges = [
#         {"source": "col1", "target": "col2", "weight": 10, "function": 0},
#         {"source": "col1", "target": "col3", "weight": 11, "function": 1},
#         {"source": "col2", "target": "col3", "weight": 20, "function": 2},
#         {"source": "col4", "target": "col5", "weight": 10, "function": 1},
#         {"source": "col2", "target": "col5", "weight": 25, "function": 3},
#         # {"source": "col3", "target": "col1", "weight": 25, "function": 4},
#         {"source": "col6", "target": "col7", "weight": 10, "function": 5},
#     ]

def topological_order(relations, weights):
    """Order every function by dependency instead of selecting among them.

    relations[i] = {'writes': set(cols), 'reads': set(cols)} for function i.
    An edge f -> g means g reads a column f writes, so f must run first.
    Cycles are broken by dropping their lowest-weight edge. Every function
    appears in the returned order exactly once.

    Returns (order, all_edges, kept_edges) where all_edges are dicts for
    display and kept_edges are (source, target, data) tuples.
    """
    print('Starting TOPOLOGICAL ORDERING:')
    n = len(relations)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    all_edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            shared = relations[i]['writes'] & relations[j]['reads']
            if shared:
                G.add_edge(i, j, weight=weights[i])
                all_edges.append((i, j, weights[i], sorted(shared)))

    print(f'  {n} function(s), {len(all_edges)} dependency edge(s)')
    for i, j, w, shared in all_edges:
        print(f'    F-{i} -> F-{j}  via {shared}  (weight {w})')

    removed = []
    while True:
        try:
            cycle = nx.find_cycle(G)
        except nx.NetworkXNoCycle:
            break
        u, v = min(((a, b) for a, b in [(e[0], e[1]) for e in cycle]),
                   key=lambda ab: G[ab[0]][ab[1]]['weight'])
        print(f'    cycle {[e[0] for e in cycle]}: dropping lowest-weight edge F-{u} -> F-{v}')
        G.remove_edge(u, v)
        removed.append((u, v))

    order = list(nx.lexicographical_topological_sort(G, key=lambda i: -weights[i]))
    print(f'  ORDER: {[f"F-{i}" for i in order]}')
    if removed:
        print(f'  edges removed to break cycles: {[(f"F-{u}", f"F-{v}") for u, v in removed]}')
    isolated = [i for i in range(n) if G.degree(i) == 0]
    if isolated:
        print(f'  independent functions (no dependencies): {[f"F-{i}" for i in isolated]}')

    edge_dicts = [{'source': f'F-{i}', 'target': f'F-{j}', 'weight': w, 'function': i}
                  for i, j, w, _ in all_edges]
    kept = [(f'F-{i}', f'F-{j}', {'weight': w, 'function': i})
            for i, j, w, _ in all_edges if G.has_edge(i, j)]
    return order, edge_dicts, kept


def find_path(edges):
    print('Starting PATH FINDING:')
    print(f'  INPUT: {len(edges)} edge(s)')
    for e in edges:
        print(f"    F-{e['function']}: {e['source']} -> {e['target']}  (weight {e['weight']})")
    in_funcs = {e['function'] for e in edges}
    in_cols = {c for e in edges for c in (e['source'], e['target'])}
    print(f'  functions in edges: {sorted(in_funcs)}')
    print(f'  columns in edges:   {sorted(in_cols)}')

    G = build_graph(edges)
    print(f'  GRAPH: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
    print(f'    nodes: {sorted(G.nodes)}')
    if G.number_of_edges() < len(edges):
        kept_in_graph = {d.get('function') for _, _, d in G.edges(data=True)}
        print(f'    NOTE: {len(edges) - G.number_of_edges()} edge(s) collapsed by DiGraph '
              f'(same source->target pair); functions surviving in graph: {sorted(kept_in_graph)}, '
              f'lost here: {sorted(in_funcs - kept_in_graph)}')

    total_all_paths = []
    for start in G.nodes:
        total_all_paths += all_paths_from_node(G, start)
    print(f'  CANDIDATE PATHS: {len(total_all_paths)}')
    for n, path in enumerate(total_all_paths):
        desc = ' -> '.join([path[0][0]] + [v for _, v, _ in path]) if path else '(empty)'
        funcs = sorted({d.get('function') for _, _, d in path})
        print(f'    [{n}] {desc}  funcs={funcs}  weight={sum(d.get("weight", 0) for _, _, d in path)}')

    sets_with_weights = []
    for path in total_all_paths:
        nodes = {n for u, v, _ in path for n in (u, v)}
        total_weight = sum([e[2]['weight'] for e in path])

        sets_with_weights.append((nodes, total_weight))
    print(f'  SETS: {[(sorted(ns), w) for ns, w in sets_with_weights]}')

    # the search below only reads (nodes, weight), so many paths covering the
    # same node set are redundant; keeping the heaviest per set is exact and
    # keeps the search from exploding
    best_by_nodes = {}
    for i, (nodes, w) in enumerate(sets_with_weights):
        key = frozenset(nodes)
        if key not in best_by_nodes or w > best_by_nodes[key][1]:
            best_by_nodes[key] = (i, w)
    index_map = [i for i, _ in best_by_nodes.values()]
    deduped = [(set(k), w) for k, (_, w) in best_by_nodes.items()]
    print(f'  DEDUPED: {len(sets_with_weights)} path(s) -> {len(deduped)} distinct node set(s)')

    solution, total = max_disjoint_sets(deduped)

    total_edges = []
    for i, s, w in solution:
        total_edges += total_all_paths[index_map[i]]

    print(f'  SELECTED: {len(solution)} disjoint path(s), total weight {total}')
    for i, s, w in solution:
        print(f'    path [{index_map[i]}] nodes={sorted(s)} weight={w}')
    out_funcs = {d.get('function') for _, _, d in total_edges}
    out_cols = {c for u, v, _ in total_edges for c in (u, v)}
    print(f'  KEPT functions:    {sorted(out_funcs)}')
    print(f'  DROPPED functions: {sorted(in_funcs - out_funcs)}')
    print(f'  DROPPED columns:   {sorted(in_cols - out_cols)}')
    return total_edges