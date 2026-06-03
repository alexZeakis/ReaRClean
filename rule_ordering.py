import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import tempfile

def build_graph(edges):
    G = nx.DiGraph()
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

            edge = (node, neighbor)

            # prevent infinite looping on same edge in same path
            if edge in [(u, v) for u, v, _ in path_edges]:
                continue

            new_nodes = path_nodes + [neighbor]
            new_edges = path_edges + [(node, neighbor, data)]

            dfs(neighbor, new_nodes, new_edges)
            extended = True

        if not extended:
            all_paths.append(path_edges)

    dfs(start, [start], [])
    return all_paths

def max_disjoint_sets(sets_with_weights):
    # attach index
    indexed = [(i, nodes, weight) for i, (nodes, weight) in enumerate(sets_with_weights)]

    best_weight = 0
    best_solution = []

    def backtrack(i, used_nodes, current_weight, chosen):
        nonlocal best_weight, best_solution

        if i == len(indexed):
            if current_weight > best_weight:
                best_weight = current_weight
                best_solution = chosen[:]
            return

        idx, nodes, weight = indexed[i]

        # skip
        backtrack(i + 1, used_nodes, current_weight, chosen)

        # take
        if nodes.isdisjoint(used_nodes):
            backtrack(
                i + 1,
                used_nodes | nodes,
                current_weight + weight,
                chosen + [(idx, nodes, weight)]
            )

    backtrack(0, set(), 0, [])
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

def find_path(edges):
    print('Starting PATH FINDING:')
    G = build_graph(edges)
    print(G)

    
    total_all_paths = []
    for start in G.nodes:
        total_all_paths += all_paths_from_node(G, start)
    print(total_all_paths)

    sets_with_weights = []
    for path in total_all_paths:
        nodes = {n for u, v, _ in path for n in (u, v)}
        total_weight = sum([e[2]['weight'] for e in path])

        sets_with_weights.append((nodes, total_weight))
    print(sets_with_weights)

    solution, total = max_disjoint_sets(sets_with_weights)

    total_edges = []
    for i, s, w in solution:
        total_edges += total_all_paths[i]

    # print("Total:", total)
    # print(total_edges)
    return total_edges


def visualize_graph(total_edges, selected_edges):
    elements = []

    # --- Normalize selected edges ---
    selected_set = set()
    for e in selected_edges:
        if isinstance(e, dict):
            selected_set.add((e["source"], e["target"]))
        else:  # tuple
            selected_set.add((e[0], e[1]))

    # --- Collect nodes ---
    nodes = set()
    for e in total_edges:
        if isinstance(e, dict):
            u, v = e["source"], e["target"]
        else:
            u, v = e[0], e[1]
        nodes.add(u)
        nodes.add(v)

    # --- Add nodes ---
    for n in nodes:
        elements.append({
            "data": {"id": n, "label": n}
        })

    # --- Add edges ---
    for e in total_edges:
        if isinstance(e, dict):
            u, v = e["source"], e["target"]
            weight = e["weight"]
            function = e["function"]
        else:
            u, v, data = e
            weight = data["weight"]
            function = data["function"]

        is_selected = (u, v) in selected_set

        elements.append({
            "data": {
                "source": u,
                "target": v,
                "label": f"F-{function}: W-{weight}",
                "id": f"{u}->{v}"
            },
            "classes": "selected" if is_selected else ""
        })

    # --- Styling ---
    stylesheet = [
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "width": 40,
                "height": 40,
                
                "font-size": 6,
                "background-color": "#6FB1FC",
                "text-valign": "center",
                "text-halign": "center",
            },
        },
        {
            "selector": "edge",
            "style": {
                "label": "data(label)",
                "color": "#FFFFFF",
                
                "width": 2,
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "line-color": "#999",
                "target-arrow-color": "#999",
                "font-size": 6,
                #"text-background-color": "white",
                #"text-background-opacity": 1,
                #"text-background-padding": 2,
                # --- label positioning tweaks ---
                "text-rotation": "autorotate",
                "text-margin-y": -10,   # move label slightly "above" edge
                
                # --- remove background ---
                "text-background-opacity": 0,
            },
        },
        {
            "selector": ".selected",
            "style": {
                "line-color": "red",
                "target-arrow-color": "red",
                "width": 4,
            },
        },
    ]

    return elements, stylesheet
