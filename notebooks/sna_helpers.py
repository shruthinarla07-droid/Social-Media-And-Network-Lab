"""Small helper functions for the SNA NetworkX lab (Amazon co-purchase network)."""
import os, random
import pandas as pd
import networkx as nx


def _data_dir():
    for d in ("../data", "data"):
        if os.path.exists(os.path.join(d, "amazon_edges.csv")):
            return d
    raise FileNotFoundError("Could not find the data folder. Open the notebook from the repository so ../data exists.")


def load_amazon():
    """Return (nodes_df, edges_df, G) where G is the DIRECTED co-purchase network with node attributes."""
    d = _data_dir()
    nodes = pd.read_csv(os.path.join(d, "amazon_nodes.csv"), dtype=str)   # dtype=str keeps leading zeros in product ids
    edges = pd.read_csv(os.path.join(d, "amazon_edges.csv"), dtype=str)
    G = nx.from_pandas_edgelist(edges, "Source", "Target", create_using=nx.DiGraph)
    G.add_nodes_from(nodes["Id"])                                          # keeps products that appear in no link (none here, but safe)
    nx.set_node_attributes(G, nodes.set_index("Id")["group"].to_dict(), "group")
    nx.set_node_attributes(G, nodes.set_index("Id")["Label"].to_dict(), "label")
    return nodes, edges, G


def find(G, text):
    """Products whose title contains `text` (case-insensitive)."""
    t = text.lower()
    return [(n, d["label"], d["group"]) for n, d in G.nodes(data=True) if t in d["label"].lower()]


def top(scores, G, n=10, name="score"):
    """Top-n nodes of a {node: score} dict as a table with title and group."""
    rows = sorted(scores.items(), key=lambda kv: -kv[1])[:n]
    return pd.DataFrame({"title": [G.nodes[k]["label"][:45] for k, _ in rows],
                         "group": [G.nodes[k]["group"] for k, _ in rows],
                         name: [v for _, v in rows]}, index=range(1, len(rows) + 1))


def sample_avg_path(H, k=200, seed=1):
    """Estimate the average shortest-path length from k random start nodes (exact is far too slow on 25,000 nodes)."""
    rnd = random.Random(seed)
    starts = rnd.sample(list(H.nodes), k)
    total = count = 0
    for s in starts:
        d = nx.single_source_shortest_path_length(H, s)
        total += sum(d.values())
        count += len(d) - 1
    return total / count


def directed_path_stats(G):
    """Exact path statistics for a DIRECTED graph, following arrows. Returns (diameter, average path length, share of pairs that are reachable).
    Only pairs that have a path are counted, like Gephi's Network Diameter."""
    longest = total = pairs = 0
    for s in G.nodes:
        d = nx.single_source_shortest_path_length(G, s)
        longest = max(longest, max(d.values()))
        total += sum(d.values())
        pairs += len(d) - 1
    n = G.number_of_nodes()
    return longest, total / pairs, pairs / (n * (n - 1))


def louvain_table(H, resolutions=(0.5, 1.0, 2.0), seeds=range(5)):
    """Run Louvain for several resolutions and seeds; return one row per run."""
    rows = []
    for r in resolutions:
        for s in seeds:
            c = nx.community.louvain_communities(H, resolution=r, seed=s)
            rows.append({"resolution": r, "seed": s, "communities": len(c),
                         "modularity": nx.community.modularity(H, c, resolution=1.0),
                         "largest": max(map(len, c))})
    return pd.DataFrame(rows)
