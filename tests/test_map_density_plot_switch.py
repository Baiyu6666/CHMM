from methods.cores.map import StageWiseMAPConstraintLearningModel


def _plot_calls(save_density_plots):
    model = object.__new__(StageWiseMAPConstraintLearningModel)
    model.map_save_density_plots = save_density_plots
    model.verbose = True
    model.loss_total = [1.0]
    calls = []
    model._plot_map_pooled_mode_density_diagnostics = (
        lambda *args, **kwargs: calls.append("pooled_density")
    )
    model._plot_map_mode_density_diagnostics = (
        lambda *args, **kwargs: calls.append("demo_density")
    )

    model._plot_map_final_pooled_diagnostics(0, [])
    return calls


def test_map_density_plots_are_disabled_by_default():
    assert _plot_calls(False) == ["pooled_density"]


def test_map_density_plots_can_be_enabled():
    assert _plot_calls(True) == ["pooled_density", "demo_density"]
