from pathlib import Path
from typing import List

from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).parent.parent

# files not ending in "tutorial.py" are not run by sphinx (see sphinx_gallery_conf in conf.py),
# so we create a list of files that look like tutorials but aren't run due to this.
#
# For every tutorial on this list, we should determine if it is ok to not run the tutorial (add a comment after
# the file name to explain why, like intro.html), or fix the tutorial (change the name to end with
# "tutorial.py" and remove it from this list).

NOT_RUN = [
    "basics/intro.html",  # no code
    "translation_transformer.html",
    "profiler.html",
    "saving_loading_models.html",
    "introyt/captumyt.html",
    "introyt/trainingyt.html",
    "examples_nn/polynomial_module.html",
    "examples_nn/dynamic_net.html",
    "examples_nn/polynomial_optim.html",
    "former_torchies/autograd_tutorial_old.html",
    "former_torchies/tensor_tutorial_old.html",
    "examples_autograd/polynomial_autograd.html",
    "examples_autograd/polynomial_custom_function.html",
    "forward_ad_usage.html",
    "parametrizations.html",
    "reinforcement_q_learning.html",
    "text_to_speech_with_torchaudio.html",
    "mnist_train_nas.html",  # used by ax_multiobjective_nas_tutorial.py
    "fx_conv_bn_fuser.html",
    "super_resolution_with_onnxruntime.html",
    "ddp_pipeline.html",  # requires 4 gpus
    "fx_graph_mode_ptq_dynamic.html",
    "vmap_recipe.html",
    "torchscript_freezing.html",
    "nestedtensor.html",
    "recipes/saving_and_loading_models_for_inference.html",
    "recipes/saving_multiple_models_in_one_file.html",
    "recipes/loading_data_recipe.html",
    "recipes/tensorboard_with_pytorch.html",
    "recipes/what_is_state_dict.html",
    "recipes/profiler_recipe.html",
    "recipes/save_load_across_devices.html",
    "recipes/warmstarting_model_using_parameters_from_a_different_model.html",
    "recipes/dynamic_quantization.html",
    "recipes/saving_and_loading_a_general_checkpoint.html",
    "recipes/benchmark.html",
    "recipes/tuning_guide.html",
    "recipes/zeroing_out_gradients.html",
    "recipes/defining_a_neural_network.html",
    "recipes/timer_quick_start.html",
    "recipes/amp_recipe.html",
    "recipes/Captum_Recipe.html",
]


def tutorial_source_dirs() -> List[Path]:
    return [
        p.relative_to(REPO_ROOT).with_name(p.stem[:-7])
        for p in REPO_ROOT.glob("*_source")
    ]


def main() -> None:
    docs_dir = REPO_ROOT / "docs"
    html_file_paths = []
    for tutorial_source_dir in tutorial_source_dirs():
        glob_path = f"{tutorial_source_dir}/**/*.html"
        html_file_paths += docs_dir.glob(glob_path)

    did_not_run = []
    for html_file_path in html_file_paths:
        with open(html_file_path, "r", encoding="utf-8") as html_file:
            html = html_file.read()
        html_soup = BeautifulSoup(html, "html.parser")
        elems = html_soup.find_all("p", {"class": "sphx-glr-timing"})
        for elem in elems:
            if (
                "Total running time of the script: ( 0 minutes  0.000 seconds)"
                in elem.text
                and not any(
                    html_file_path.match(file) for file in NOT_RUN
                )
            ):
                did_not_run.append(html_file_path.as_posix())

    if len(did_not_run) != 0:
        raise RuntimeError(
            "The following file(s) are not known bad but ran in 0.000 sec, meaning that any "
            + "python code in this tutorial probably didn't run:\n{}".format(
                "\n".join(did_not_run)
            )
        )


if __name__ == "__main__":
    main()
