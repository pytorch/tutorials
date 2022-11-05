from pathlib import Path
from typing import List

from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).parent.parent

# For every tutorial on this list, we should determine if it is ok to not run the tutorial (add a comment after
# the file name to explain why, like intro.html), or fix the tutorial and remove it from this list).

NOT_RUN = [
    "basics/intro",  # no code
    "translation_transformer",
    "profiler",
    "saving_loading_models",
    "introyt/captumyt",
    "introyt/trainingyt",
    "examples_nn/polynomial_module",
    "examples_nn/dynamic_net",
    "examples_nn/polynomial_optim",
    "former_torchies/autograd_tutorial_old",
    "former_torchies/tensor_tutorial_old",
    "examples_autograd/polynomial_autograd",
    "examples_autograd/polynomial_custom_function",
    "parametrizations",
    "mnist_train_nas",  # used by ax_multiobjective_nas_tutorial.py
    "fx_conv_bn_fuser",
    "super_resolution_with_onnxruntime",
    "ddp_pipeline",  # requires 4 gpus
    "fx_graph_mode_ptq_dynamic",
    "vmap_recipe",
    "torchscript_freezing",
    "nestedtensor",
    "recipes/saving_and_loading_models_for_inference",
    "recipes/saving_multiple_models_in_one_file",
    "recipes/loading_data_recipe",
    "recipes/tensorboard_with_pytorch",
    "recipes/what_is_state_dict",
    "recipes/profiler_recipe",
    "recipes/save_load_across_devices",
    "recipes/warmstarting_model_using_parameters_from_a_different_model",
    "recipes/dynamic_quantization",
    "recipes/saving_and_loading_a_general_checkpoint",
    "recipes/benchmark",
    "recipes/tuning_guide",
    "recipes/zeroing_out_gradients",
    "recipes/defining_a_neural_network",
    "recipes/timer_quick_start",
    "recipes/amp_recipe",
    "recipes/Captum_Recipe",
    "hyperparameter_tuning_tutorial",
    "flask_rest_api_tutorial",
    "text_to_speech_with_torchaudio",
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
