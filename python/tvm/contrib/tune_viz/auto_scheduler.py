
class VisualizeProgress(TaskSchedulerCallback):
    """Callback that generates a plot to visualize the tuning progress.

    Parameters
    ----------
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

    def __init__(
        self,
        title="AutoScheduler Progress",
        si_prefix="G",
        keep_open=False,
        live=True,
        out_path=None,
    ):
        self.best_flops = {}
        self.all_cts = {}
        self.si_prefix = si_prefix
        self.keep_open = keep_open
        self.live = live
        self.out_path = out_path
        self.init_plot(title)

    def __del__(self):
        import matplotlib.pyplot as plt

        if self.out_path:
            print(f"Writing plot to file {self.out_path}...")
            plt.savefig(self.out_path)
        if self.live and self.keep_open:
            print("Close matplotlib window to continue...")
            plt.show()

    def pre_tune(self, task_scheduler, task_id):
        for i, task in enumerate(task_scheduler.tasks):
            cts = task_scheduler.task_cts[i]
            if cts == 0:
                self.all_cts[i] = [0]
                self.best_flops[i] = [0]
        self.update_plot()

    def post_tune(self, task_scheduler, task_id):
        for i, task in enumerate(task_scheduler.tasks):
            if task_scheduler.best_costs[i] < 1e9:
                flops = task_scheduler.tasks[i].compute_dag.flop_ct / task_scheduler.best_costs[i]
                flops = format_si_prefix(flops, self.si_prefix)
                cts = task_scheduler.task_cts[i]
                if cts not in self.all_cts[i]:
                    self.all_cts[i].append(cts)
                    self.best_flops[i].append(flops)
            else:
                flops = None
        self.update_plot()

    def init_plot(self, title):
        import matplotlib.pyplot as plt

        plt.figure(title)

    def update_plot(self):
        import matplotlib.pyplot as plt

        plt.clf()
        for i in range(len(self.all_cts)):
            if i in self.all_cts and i in self.best_flops:
                color = plt.cm.tab10(i)
                plt.plot(self.all_cts[i], self.best_flops[i], color=color, label=f"Task {i}")
        plt.legend(loc="upper left")
        plt.xlabel("Iterations")
        plt.ylabel(f"{self.si_prefix}FLOPS")
        if self.live:
            plt.pause(0.05)
