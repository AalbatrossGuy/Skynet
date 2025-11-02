import os
import json
import click
import pathlib
# from __future__ import annotations
from matplotlib import pyplot as plot


def _save_chart(dir_path: pathlib.Path):
    dir_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plot.tight_layout()
    plot.savefig(dir_path.as_posix())
    plot.close()


def plot_accuracy(
    plot_data: dict,
    dir: pathlib.Path,
    prefix: str
) -> pathlib.Path:
    rounds = [
        history["round"] for history in plot_data["history"]
    ]
    clients = sorted(
        tuple(plot_data["history"][0]["accuracy"].keys())
    )
    print(f"CLIENT = {clients}")
    accuracy_per_client = {
        client: [
            history["accuracy"][client]["accuracy"]
            for history in plot_data["history"]
        ] for client in clients
    }
    average_accuracy = [
        sum(
            history["accuracy"][client]["accuracy"] for client in clients
        ) / len(clients) for history in plot_data["history"]
    ]

    plot.figure(figsize=(10, 6))

    for client in clients:
        plot.plot(
            rounds,
            accuracy_per_client[client],
            marker="o",
            label=client
        )

    plot.plot(
        rounds,
        average_accuracy,
        marker="o",
        linestyle="--",
        label="Average"
    )

    plot.title("Client Accuracies per Round")
    plot.xlabel("Round")
    plot.ylabel("Accuracy")
    plot.grid(True, linestyle="--", alpha=0.5)
    plot.legend()
    output = dir / f"{prefix}accuracy_per_client.png"
    _save_chart(output)
    return output


def plot_weight_normalization(
    plot_data: dict,
    dir: pathlib.Path,
    prefix: str
) -> pathlib.Path:
    rounds = [
        history["round"] for history in plot_data["history"]
    ]

    weight_normalization = [
        history["weight_norm"] for history in plot_data["history"]
    ]

    plot.figure(figsize=(10, 6))
    plot.plot(
        rounds,
        weight_normalization,
        marker="o"
    )
    plot.title("Global Weight Normalization Update Over Rounds")
    plot.xlabel("Round")
    plot.ylabel("Weight Normalization")
    plot.grid(True, linestyle="--", alpha=0.5)
    output = dir / f"{prefix}weight_normalization.png"
    _save_chart(output)
    return output


def plot_final_weight(
    plot_data: dict,
    dir: pathlib.Path,
    prefix: str
) -> pathlib.Path:
    weights = list(plot_data["training_weights"])
    plot.figure(figsize=(10, 6))
    plot.bar(range(len(weights)), weights)
    plot.axhline(0, linewidth=1)
    plot.title("Final Training Weights [Index vs Value]")
    plot.xlabel("Weight Index")
    plot.ylabel("Weight Value")
    plot.tight_layout()
    output = dir / f"{prefix}final_weights.png"
    _save_chart(output)
    return output


@click.command()
@click.option("--file", "file_path", type=click.Path(exists=True, dir_okay=False, readable=True), default="export.json", show_default=True, help="Path to export.json")
@click.option("--outdir", type=click.Path(file_okay=False, writable=True), default="reports", show_default=True, help="Directory to save charts")
@click.option("--prefix", default="", show_default=False, help="Optional filename prefix (e.g., run1_)")
def cli(file_path: str, outdir: str, prefix: str):
    """Generate PNG charts from an export.json."""
    outdir_path = pathlib.Path(outdir)
    with open(file_path, "r") as f:
        data = json.load(f)

    a = plot_accuracy(data, outdir_path, prefix)
    b = plot_weight_normalization(data, outdir_path, prefix)
    c = plot_final_weight(data, outdir_path, prefix)

    click.echo("Charts saved:")
    click.echo(f" - {a}")
    click.echo(f" - {b}")
    click.echo(f" - {c}")


if __name__ == "__main__":
    cli()
