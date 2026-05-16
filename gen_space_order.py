import sys
import pickle
import random
from collections import defaultdict

import networkx as nx

assert len(sys.argv) == 2

pkl_file = sys.argv[1]

with open(pkl_file, "rb") as f:
    G = pickle.load(f)
print("G", G)

G_reduced2 = nx.transitive_reduction(G)
G_reduced = G.copy()
to_drop = []
for edge in G_reduced.edges:
    u, v = edge
    if edge not in G_reduced2.edges:
        to_drop.append(edge)
for edge in to_drop:
    u, v = edge
    G_reduced.remove_edge(u, v)
# print("G_reduced2", G_reduced2)

redundant_nodes = [
    n
    for n, attrs in G_reduced.nodes(data=True)
    if attrs["redundant"]
]
for n in redundant_nodes:
    G_reduced.remove_node(n)



print("G_reduced", G_reduced)

FOLD = True
# FOLD = False

if FOLD:
    def fold_small_spaces(G, min_size):
        """
        Fold chains of small spaces into groups.
    
        Returns:
            groups: list[list[node]]
            representative_of: dict[node -> representative]
        """
        # print("fold_small_spaces", G, min_size)
    
        G_gen = G.reverse(copy=True)
        # print("G_gen", G_gen)
    
        visited = set()
    
        groups = []
        representative_of = {}
    
        roots = [n for n in G_gen.nodes if G_gen.in_degree(n) == 0]
        # print("roots", roots)
    
        for root in roots:
            # print("root", root)
            cur = root
    
            group = []
    
            while True:
                # print("loop")
                # print("cur", cur)
                if cur in visited:
                    # print("break")
                    break
    
                visited.add(cur)
                group.append(cur)
    
                size = G.nodes[cur]["size"]
                # print("size", size)
    
                succs = list(G_gen.successors(cur))
                # print("succs", succs, len(succs))
    
                # stop conditions
                stop = False
    
                if size >= min_size:
                    # print("size >= min_size")
                    stop = True
    
                elif len(succs) > 1:
                    # print("multiple succs")
                    stop = True
                elif len(succs) == 0:
                    # print("no succs")
                    stop = True
    
                # print("stop", stop)
    
                if stop:
                    rep = cur
    
                    groups.append(group)
    
                    for n in group:
                        representative_of[n] = rep
    
                    # continue traversal upward if possible
                    if len(succs) == 1:
                        cur = succs[0]
                        group = []
                        continue
    
                    break
    
                cur = succs[0]
    
        return groups, representative_of
    
    
    MIN_SIZE = 20
    
    groups, representative_of = fold_small_spaces(G_reduced, MIN_SIZE)
    print("groups", groups)
    print("representative_of", representative_of)
    
    
    def build_contracted_graph(G, representative_of):
        """
        Build a quotient graph where nodes are representatives of folded groups.
        """
    
        H = nx.DiGraph()
    
        # invert mapping: rep -> members
        groups = defaultdict(list)
        for node, rep in representative_of.items():
            groups[rep].append(node)
    
        # add nodes with aggregated attributes
        for rep, members in groups.items():
            data = G.nodes[rep]
            H.add_node(
                rep,
                **data,
                # size=sum(G.nodes[n]["size"] for n in members),
                # num_nodes=len(members),
                # members=members,
            )
    
        # add edges (collapsed)
        for u, v, data in G.edges(data=True):
            ru = representative_of.get(u)
            rv = representative_of.get(v)
    
            if ru is None or rv is None:
                continue
    
            if ru == rv:
                continue  # internal edge inside collapsed group
    
            # H.add_edge(ru, rv)
            H.add_edge(ru, rv, **data)
    
        # optional: remove duplicates (nx already handles, but safe)
        H = nx.DiGraph(H)
    
        return H
    
    G_small = build_contracted_graph(G_reduced, representative_of)
    print("G_small", G_small)
    G_reduced = G_small

G_gen = G_reduced.reverse(copy=True)

print("G_gen", G_gen)
print("G_gen.nodes", G_gen.nodes)
print("G_gen.edges", G_gen.edges)

G_cur = G_gen.copy()


def get_available(G):
    available = [
        n for n in G.nodes
        if G.in_degree(n) == 0
    ]
    return available


available = get_available(G_cur)

print("available", available, len(available))

def pick_best(available, G, strategy="random"):
    print("pick_best", available, strategy)
    if strategy == "random":
        best = random.choice(available)
    elif strategy == "smallest":
        sorted_ids = list(sorted(available, key=lambda n: G.nodes[n]["size"]))
        best = sorted_ids[0]
    else:
        raise NotImplementedError(f"strategy={strategy}")
    return best

order = []
sizes = []

strategy = "smallest"

while available:
    best = pick_best(available, G_cur, strategy=strategy)
    print("best", best)

    order.append(best)
    if len(sizes) == 0:
        new_size = G.nodes[best]["size"]
        sizes.append(new_size)
    else:
        generated = set(order)
        # in_edges = list(G_gen.in_edges(best))
        in_edges = [
            e for e in G_gen.in_edges(best)
            if e[0] in generated
        ]
        print("in_edges", in_edges)
        # out_edges = list(G_gen.out_edges(best))
        # print("out_edges", out_edges)
        old_size = sizes[-1]
        print("old_size", old_size)

        if len(in_edges) == 0:
            added_size = G_gen.nodes[best]["size"]
            print("added_size", added_size)
            new_size = old_size + added_size
            print("new_size", new_size)
        elif len(in_edges) == 1:
            in_edge = in_edges[0]
            print("in_edge", in_edge)
            smaller = in_edge[0]
            print("smaller", smaller)
            edge = G_gen.edges[in_edge]
            print("edge", edge)
            containment_ratio = edge["containment_ratio"]
            print("containment_ratio", containment_ratio)
            sz = G_gen.nodes[best]["size"]
            print("sz", sz)
            added_size = round(sz * (1 - containment_ratio), 2)
            print("added_size", added_size)
            assert int(added_size) == added_size
            added_size = int(added_size)
            new_size = old_size + added_size
            print("new_size", new_size)
        elif len(in_edges) > 0:
            raise NotImplementedError
        sizes.append(new_size)


    G_cur.remove_node(best)

    available = get_available(G_cur)
    print("available", available, len(available))

print("order", order)
print("sizes", sizes)
sizes2 = [G.nodes[n]["size"] for n in order]
print("sizes2", sizes2)

active = set()
history = []

for n in order:
    # all smaller spaces contained in n
    contained = nx.descendants(G, n)

    # remove contained active spaces
    active -= contained

    # add new maximal space
    active.add(n)

    # snapshot
    history.append(set(active))

print("sets =", history)
