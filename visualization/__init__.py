from .io import DEFAULT_PLOT_ROOT, learner_plot_dir, plot_root, save_figure
from .plot4panel import plot_feature_model_debug, plot_results_4panel
from .swcl_4panel import plot_swcl_results_4panel

PyBulletRenderer3D = None


def __getattr__(name):
    if name == "PyBulletRenderer3D":
        try:
            from .pybullet_renderer import PyBulletRenderer3D as _Renderer
        except ModuleNotFoundError:
            return None
        return _Renderer
    raise AttributeError(name)
