from pathlib import Path
from typing import List

from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).parent.parent

# For every tutorial on this list, we should determine if it is ok to not run the tutorial (add a comment after
# the file name to explain why, like intro.html), or fix the tutorial and remove it from this list).

NOT_RUN = [
    "beginner_source/basics/intro",  # no code
    "beginner_source/introyt/introyt_index", # no code
    "beginner_source/onnx/intro_onnx",
    "beginner_source/profiler",
    "beginner_source/saving_loading_models",
    "beginner_source/introyt/captumyt",
    "beginner_source/examples_nn/polynomial_module",
    "beginner_source/examples_nn/dynamic_net",
    "beginner_source/examples_nn/polynomial_optim",
    "beginner_source/former_torchies/autograd_tutorial_old",
    "beginner_source/former_torchies/tensor_tutorial_old",
    "beginner_source/examples_autograd/polynomial_autograd",
    "beginner_source/examples_autograd/polynomial_custom_function",
    "intermediate_source/mnist_train_nas",  # used by ax_multiobjective_nas_tutorial.py
    "intermediate_source/fx_conv_bn_fuser",
    "intermediate_source/_torch_export_nightly_tutorial",  # does not work on release
    "advanced_source/super_resolution_with_onnxruntime",
    "advanced_source/usb_semisup_learn", # fails with CUDA OOM error, should try on a different worker
    "prototype_source/fx_graph_mode_ptq_dynamic",
    "prototype_source/vmap_recipe",
    "prototype_source/torchscript_freezing",
    "prototype_source/nestedtensor",
    "recipes_source/recipes/saving_and_loading_models_for_inference",
    "recipes_source/recipes/saving_multiple_models_in_one_file",
    "recipes_source/recipes/tensorboard_with_pytorch",
    "recipes_source/recipes/what_is_state_dict",
    "recipes_source/recipes/profiler_recipe",
    "recipes_source/recipes/save_load_across_devices",
    "recipes_source/recipes/warmstarting_model_using_parameters_from_a_different_model",
    "recipes_source/recipes/dynamic_quantization",
    "recipes_source/recipes/saving_and_loading_a_general_checkpoint",
    "recipes_source/recipes/benchmark",
    "recipes_source/recipes/tuning_guide",
    "recipes_source/recipes/zeroing_out_gradients",
    "recipes_source/recipes/defining_a_neural_network",
    "recipes_source/recipes/timer_quick_start",
    "recipes_source/recipes/amp_recipe",
    "recipes_source/recipes/Captum_Recipe",
    "intermediate_source/flask_rest_api_tutorial",
    "intermediate_source/text_to_speech_with_torchaudio",
    "intermediate_source/tensorboard_profiler_tutorial", # reenable after 2.0 release.
    "advanced_source/semi_structured_sparse" # reenable after 3303 is fixed.
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

    should_not_run = [f'{x.replace("_source", "")}.html' for x in NOT_RUN]
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
                and not any(html_file_path.match(file) for file in should_not_run)
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
