import argparse
# from collections import defaultdict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_csv(path):
    df = pd.read_csv(path)
    if "label" not in df.columns:
        df["label"] = "default"
    labels = df["label"].unique()
    assert len(labels) == 1
    # label = labels[0]
    df.set_index(["label", "mode", "record"], inplace=True)
    return df


parser = argparse.ArgumentParser(description="TODO", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("input", nargs="+", help="TODO")
parser.add_argument("--col", default="flops_per_s", help="TODO")
parser.add_argument("--scale", default=1e6, help="TODO")
parser.add_argument("--cmap", default="crest", help="TODO")
parser.add_argument("--out", "-o", default=None, help="TODO")
parser.add_argument("--print", action="store_true", help="TODO")
parser.add_argument("--highlight", action="store_true", help="TODO")
parser.add_argument("--top-rows", type=int, default=3, help="TODO")
parser.add_argument("--top-cols", type=int, default=2, help="TODO")
parser.add_argument("--per-label", action="store_true", help="TODO")
args = parser.parse_args()

# label_dfs = defaultdict(list)
dfs = []

for i, path in enumerate(args.input):
    df_ = load_csv(path)
    df_ = df_[[args.col]]
    df_.rename(columns={args.col: i}, inplace=True)
    # label_dfs[label].append(df_)
    dfs.append(df_)

# label_df = {label: pd.concat(dfs, axis=1) for label, dfs in label_dfs.items()}
df = pd.concat(dfs, axis=1)

# for label, df in label_df.items():
#     if args.scale:
#         df = df / args.scale
#     df.insert(0, "label", label)
#     label_df[label] = df
if args.scale:
    df = df / args.scale

# df = pd.concat(label_df.values(), axis=0)

if args.print:
    print("DataFrame:")
    print(df)

# input("???")

num_labels = 3  # TODO
fig, axs = plt.subplots(num_labels, 1, figsize=(10, 3 * num_labels))
fig.suptitle(f"Col: {args.col}/{args.scale}")

if args.out:

    def plot(df, cmap="crest", title="Heatmap", ax=None):
        ax_ = sns.heatmap(df, annot=True, fmt=".1f", linewidth=.5, cmap=cmap, ax=ax)
        ax_.set_title(title)
        for x, y in highlights:
            w = 1
            h = 1
            ax_.add_patch(Rectangle((y, x), w, h, fill=False, edgecolor="crimson", lw=4, clip_on=False))
        ax_.set(xlabel="", ylabel="")
        # return ax_

    i = 0
    for label, label_df in df.groupby("label", dropna=True):
        print(">", label, label_df)
        label_df.dropna(axis=1, inplace=True)
        label_df = label_df.droplevel("label")
        highlights = []

        if args.highlight:
            n = args.top_rows
            m = args.top_cols
            temp = label_df.max(axis=1)
            print("temp")
            print(temp)
            nl_rows = temp.nlargest(n, keep="first")
            print("nl_rows")
            print(nl_rows)
            temp2 = label_df.loc[nl_rows.index]
            print("temp2")
            print(temp2)
            nl_cols = temp2.apply(lambda s: s.abs().nlargest(m).index.tolist(), axis=1)
            print("nl_cols")
            print(nl_cols)
            for idx, cols in nl_cols.items():
                print("idx", idx)
                for col in cols:
                    print("col", col)
                    # print(list(label_df.index.values))
                    hl = (label_df.index.get_loc(idx), label_df.columns.get_loc(col))
                    highlights.append(hl)

        print("highlights", highlights)


        plot(label_df, cmap=args.cmap, title=f"Label: {label}", ax=axs[i])
        i += 1
    axs[0].set_title("Heatmap")


    fig.subplots_adjust(left=0.3)
    fig.savefig(args.out)
