import time
from rich import print, box
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.text import Text


# print(Panel("Hello, [red]World!"))
# print(Panel.fit("Hello, [red]World!"))
# print(Panel("Hello, [red]World!", title="Welcome", subtitle="Thank you"))
# table = Table(title="Targets", box=box.MINIMAL_DOUBLE_HEAD)
table = Table(title="Targets")
targets_table = Table(title="Targets")
configs_table = Table(title="Configs")
mods_table = Table(title="Mods")
layouts_table = Table(title="Layouts")
workloads_table = Table(title="Workloads")
spaces_table = Table(title="Spaces")
tasks_table = Table(title="Tasks")
tuners_table = Table(title="Tuners")
tuning_table = Table()

targets_table.add_column("ID", justify="right", style="cyan", no_wrap=True)
targets_table.add_column("Kind", style="magenta")
targets_table.add_column("String", justify="right", style="green")
targets_table.add_row("T0", "llvm", "llvm -num-cpus=1")

configs_table.add_column("ID", justify="right", style="cyan", no_wrap=True)
configs_table.add_column("Pass Config", style="magenta")
configs_table.add_column("Disabled Passes", justify="right", style="green")
configs_table.add_column("Opt Level", justify="right", style="green")
configs_table.add_row("C0", "{}", "[]", "3")

mods_table.add_column("ID", justify="right", style="cyan", no_wrap=True)
mods_table.add_column("Name", style="magenta")
mods_table.add_column("Kind", justify="right", style="green")
mods_table.add_row("M0", "resnet8", "relay")

layouts_table.add_column("ID", justify="right", style="cyan", no_wrap=True)
layouts_table.add_column("Data Layout", style="magenta")
layouts_table.add_column("Kernel Layout", justify="right", style="green")
layouts_table.add_row("L0", "default", "default")

workloads_table.add_column("ID", justify="right", style="cyan", no_wrap=True)
workloads_table.add_column("Mapping", style="magenta")
workloads_table.add_row("W0", "(T0, C0, M0, L0)")

spaces_table.add_column("ID", justify="right", style="cyan", no_wrap=True)
spaces_table.add_column("Mapping", style="magenta")
spaces_table.add_row("S0", "(T0, C0, M0, L0)")

tasks_table.add_column("ID", justify="right", style="cyan", no_wrap=True)
tasks_table.add_column("Mapping", style="magenta")
tasks_table.add_row("I0", "(T0, C0, M0, L0)")

tuners_table.add_column("ID", justify="right", style="cyan", no_wrap=True)
tuners_table.add_column("Mapping", style="magenta")
tuners_table.add_row("?0", "(T0, C0, M0, L0)")

tuning_table.add_column("Task", justify="right", style="cyan", no_wrap=True)
tuning_table.add_column("Space", justify="right", style="cyan", no_wrap=True)
tuning_table.add_column("Space Size")
tuning_table.add_column("Subspaces")
tuning_table.add_column("Masked Size")
tuning_table.add_column("Latency (ms)")
tuning_table.add_column("Performance (GFLOPS)")
tuning_table.add_column("Trials")
tuning_table.add_column("Coverage [Masked]")
tuning_table.add_column("Status")
tuning_table.add_row("T0", "S0", "~11256", "8/22", "6644 (59%)", "0.2325", "11.65", "105", "1.0% [1.6%]", Text("✓", style="green"))
tuning_table.add_row("T0", "S1", "~11256", "3/3", "11256 (100%)", "0.2325", "11.65", "105", "1.0% [1.6%]", Spinner("dots", style="orange"))
tuning_table.add_row("T0", "S2", "~11256", "3/3", "11256 (100%)", "N/A", "N/A", "0", "0.0% [0.0%]", Spinner("clock", style="orange", speed=0.1))


# console = Console()
# console.print(targets_table)


layout = Layout()
# layout.split_column(
#     Layout(Panel("hiiii...")),
#     Layout(name="lower"),
#     Layout(name="lower2"),
# )
# layout["lower"].split_row(
layout.split_row(
    Layout(targets_table),
    Layout(configs_table),
    Layout(mods_table),
    Layout(layouts_table),
    # Layout(name="right"),
)
layout2 = Layout()
layout2.split_row(
    Layout(workloads_table),
    Layout(spaces_table),
    Layout(tasks_table),
    Layout(tuners_table),
    # Layout(name="right"),
)
# layout["lower2"].split_row(
#     Layout(table),
#     Layout(table),
#     Layout(table),
#     Layout(table),
#     # Layout(name="right"),
# )
# layout["right"].split(
#     Layout(Panel("Hello")),
#     Layout(Panel("World!"))
# )
full_layout = Layout()
full_layout.split(layout, layout2)
print(full_layout)
tuning_layout = Layout(tuning_table)
# print(tuning_layout)

# with Live(Panel(Columns([Spinner("dots", text=Text("text", style="green"))], column_first=True, expand=True), title="Title", border_style="blue"), refresh_per_second=20) as live:
with Live(Panel(tuning_table, title="Tuning", border_style="blue"), refresh_per_second=5) as live:
    while True:
        time.sleep(1)
