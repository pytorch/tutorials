import sys
import glob
import re

from bs4 import BeautifulSoup

# files that are ok to have 0 min 0 sec time, probably because they don't have any python code
OK_TO_NOT_RUN = [
    "beginner/basics/intro.html",
]

KNOWN_BAD = [
    "beginner/translation_transformer.html",
    "beginner/torchtext_translation.html",
    "beginner/profiler.html",
    "beginner/saving_loading_models.html",
    "beginner/introyt/captumyt.html",
    "beginner/introyt/trainingyt.html",
    "beginner/examples_nn/polynomial_module.html",
    "beginner/examples_nn/two_layer_net_optim.html",
    "beginner/examples_nn/dynamic_net.html",
    "beginner/examples_nn/two_layer_net_module.html",
    "beginner/examples_nn/polynomial_optim.html",
    "beginner/examples_nn/polynomial_nn.html",
    "beginner/examples_nn/two_layer_net_nn.html",
    "beginner/examples_tensor/two_layer_net_tensor.html",
    "beginner/examples_tensor/two_layer_net_numpy.html",
    "beginner/examples_tensor/polynomial_numpy.html",
    "beginner/examples_tensor/polynomial_tensor.html",
    "beginner/former_torchies/autograd_tutorial_old.html",
    "beginner/former_torchies/tensor_tutorial_old.html",
    "beginner/examples_autograd/two_layer_net_autograd.html",
    "beginner/examples_autograd/polynomial_autograd.html",
    "beginner/examples_autograd/tf_two_layer_net.html",
    "beginner/examples_autograd/polynomial_custom_function.html",
    "beginner/examples_autograd/two_layer_net_custom_function.html",
    "intermediate/forward_ad_usage.html",
    "intermediate/parametrizations.html",
    "intermediate/reinforcement_q_learning.html",
    "intermediate/text_to_speech_with_torchaudio.html",
    "intermediate/mnist_train_nas.html",
    "intermediate/fx_conv_bn_fuser.html",
    "advanced/super_resolution_with_onnxruntime.html",
    "advanced/super_resolution_with_caffe2.html",
    "advanced/ddp_pipeline.html",
    "prototype/fx_graph_mode_ptq_dynamic.html",
    "prototype/vmap_recipe.html",
    "prototype/torchscript_freezing.html",
    "prototype/nestedtensor.html",
    "recipes/recipes/saving_and_loading_models_for_inference.html",
    "recipes/recipes/saving_multiple_models_in_one_file.html",
    "recipes/recipes/loading_data_recipe.html",
    "recipes/recipes/tensorboard_with_pytorch.html",
    "recipes/recipes/what_is_state_dict.html",
    "recipes/recipes/profiler_recipe.html",
    "recipes/recipes/save_load_across_devices.html",
    "recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html",
    "recipes/recipes/dynamic_quantization.html",
    "recipes/recipes/saving_and_loading_a_general_checkpoint.html",
    "recipes/recipes/benchmark.html",
    "recipes/recipes/tuning_guide.html",
    "recipes/recipes/custom_dataset_transforms_loader.html",
    "recipes/recipes/zeroing_out_gradients.html",
    "recipes/recipes/intel_extension_for_pytorch.html",
    "recipes/recipes/defining_a_neural_network.html",
    "recipes/recipes/timer_quick_start.html",
    "recipes/recipes/amp_recipe.html",
    "recipes/recipes/Captum_Recipe.html",
]


def main():
    build_dir = sys.argv[1]

    html_file_paths = []

    for difficulty in ["beginner", "intermediate", "advanced", "prototype", "recipes"]:
        glob_path = f"{build_dir}/{difficulty}/**/*.html"
        html_file_paths += glob.glob(glob_path, recursive=True)

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
                and not any(html_file_path.endswith(file) for file in KNOWN_BAD + OK_TO_NOT_RUN)
            ):
                did_not_run.append(html_file_path)

    if len(did_not_run) != 0:
        raise RuntimeError(
            f"File(s) {' '.join(did_not_run)} are not known bad but ran in 0.000 sec, meaning that any python code in this tutorial probably didn't run"
        )


if __name__ == "__main__":
    main()
