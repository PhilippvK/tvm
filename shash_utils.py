
def jaccard(a, b):
    return len(a & b) / len(a | b)


def drop_similar_shashs_nodes(G, space2shashs, threshold=0.98):
    keep = []
    seen = []

    for idx, s in space2shashs.items():
        redundant = False
        for prev in seen:
            if jaccard(s, prev) >= threshold:
                redundant = True
                break
        if not redundant:
            keep.append(idx)
            seen.append(s)


    print("keep", keep, len(keep))
    # print("seen", seen, len(seen))
    drop_nodes = set(list(G.nodes)) - set(keep)
    print("drop_nodes", drop_nodes)
    if len(drop_nodes) > 0:
        print(f"Dropping {len(drop_nodes)} nodes...")
        G.remove_nodes_from(drop_nodes)


def drop_similar_shashs_rows(df, threshold=0.98):
    keep = []
    seen = []

    for idx, s in df["shashs"].items():
        redundant = False
        for prev in seen:
            if jaccard(s, prev) >= threshold:
                redundant = True
                break
        if not redundant:
            keep.append(idx)
            seen.append(s)


    print("keep", keep, len(keep))
    # print("seen", seen, len(seen))
    df_filtered = df.loc[keep]
    return df_filtered
