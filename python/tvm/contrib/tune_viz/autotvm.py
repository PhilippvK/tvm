
def visualize_progress(
    idx, title="AutoTVM Progress", si_prefix="G", keep_open=False, live=True, out_path=None
):
    """Display tuning progress in graph

    Parameters
    ----------
    idx: int
        Index of the current task.
    title: str
        Specify the title of the matplotlib figure.
    si_prefix: str
        SI prefix for flops
    keep_open: bool
        Wait until the matplotlib window was closed by the user.
    live: bool
        If false, the graph is only written to the file specified in out_path.
    out_path: str
        Path where the graph image should be written (if defined).
    """
    import matplotlib.pyplot as plt

    class _Context(object):
        """Context to store local variables"""

        def __init__(self):
            self.keep_open = keep_open
            self.live = live
            self.out_path = out_path
            self.best_flops = [0]
            self.all_flops = []
            if idx > 0:
                plt.figure(title)
            else:
                plt.figure(title).clear()
            self.color = plt.cm.tab10(idx)
            (self.p,) = plt.plot([0], [0], color=self.color, label=f"Task {idx}")
            plt.xlabel("Iterations")
            plt.ylabel(f"{si_prefix}FLOPS")
            plt.legend(loc="upper left")
            if self.live:
                plt.pause(0.05)

        def __del__(self):
            if self.out_path:
                print(f"Writing plot to file {self.out_path}...")
                plt.savefig(self.out_path)
            if self.live and self.keep_open:
                print("Close matplotlib window to continue...")
                plt.show()

    ctx = _Context()

    def _callback(_, inputs, results):

        flops = 0
        for inp, res in zip(inputs, results):
            m = "x"
            if res.error_no == 0:
                flops = inp.task.flop / np.mean(res.costs)
                m = "."

            flops = format_si_prefix(flops, si_prefix)
            ctx.all_flops.append(flops)
            best = max(flops, ctx.best_flops[-1])
            ctx.best_flops.append(best)

            axes = plt.gca()
            _, ymax = axes.get_ylim()
            _, xmax = axes.get_xlim()
            plt.axis([0, max(len(ctx.all_flops) + 1, xmax), 0, max(ctx.best_flops[-1] * 1.1, ymax)])
            plt.scatter(len(ctx.all_flops), flops, color=ctx.color, marker=m, s=15)
            ctx.p.set_data(list(range(0, len(ctx.all_flops) + 1)), ctx.best_flops)
        if live:
            plt.pause(0.05)

    return _callback
