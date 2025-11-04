from pathlib import Path
from analytics.charts import (
    plot_accuracy,
    plot_weight_normalization,
    plot_final_weight
)

EXPORT_FIXTURE = {
  "round": 3,
  "feature_weight": 12,
  "training_weights": [0.1, -0.2, 0.05],
  "history": [
    {"round": 1, "weight_norm": 0.1, "accuracy": {"A": {"accuracy": 0.7},"B": {"accuracy": 0.8}}},
    {"round": 2, "weight_norm": 0.2, "accuracy": {"A": {"accuracy": 0.75},"B": {"accuracy": 0.82}}},
    {"round": 3, "weight_norm": 0.25, "accuracy": {"A": {"accuracy": 0.78},"B":{"accuracy": 0.83}}},
  ],
  "export_time": 0
}


def test_plot_functions_write_pngs(tmp_path: Path):
    outdir = tmp_path / "reports"
    outdir.mkdir(parents=True, exist_ok=True)

    a = plot_accuracy(EXPORT_FIXTURE, outdir, "t_")
    b = plot_weight_normalization(EXPORT_FIXTURE, outdir, "t_")
    c = plot_final_weight(EXPORT_FIXTURE, outdir, "t_")

    for p in (a, b, c):
        assert p.exists() and p.stat().st_size > 0
