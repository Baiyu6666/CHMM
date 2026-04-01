from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLOT_ROOT = PROJECT_ROOT / "outputs" / "plots"


def plot_root(plot_dir: object = None) -> Path:
    if plot_dir is None:
        path = DEFAULT_PLOT_ROOT
    else:
        path = Path(plot_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def learner_plot_dir(learner, plot_dir: object = None) -> Path:
    return plot_root(plot_dir if plot_dir is not None else getattr(learner, "plot_dir", None))


def save_figure(fig, path: Path, close: bool = True, dpi: int = 160) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    savefig_kwargs = {"dpi": dpi}
    if str(path).lower().endswith(".png"):
        # Favor faster write-out over maximum PNG compression. This keeps the
        # rendered figure identical while reducing time spent in file encoding.
        savefig_kwargs["pil_kwargs"] = {"compress_level": 1, "optimize": False}
    fig.savefig(path, **savefig_kwargs)
    if close:
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass
    return path
