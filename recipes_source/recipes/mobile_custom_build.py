"""
PyTorch Mobile Custom Build
================================
To reduce the size of binaries you can do custom build of PyTorch Android
with only set of operators required by your model. This includes two steps:
preparing the list of operators from your model, rebuilding pytorch android
with specified list.

"""


######################################################################
# 1. Verify your PyTorch version is 1.4.0 or above. You can do that
# by checking the value of torch.__version__.
#


######################################################################
# 2. Preparation of the list of operators
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# List of operators of your serialized torchscript model can be prepared in
# yaml format using python api function torch.jit.export_opnames(). To dump
# the operators in your model, say MobileNetV2, run the following lines of
# Python code:
#

# Dump list of operators used by MobileNetV2:
import torch, yaml
model = torch.jit.load('MobileNetV2.pt')
ops = torch.jit.export_opnames(model)
with open('MobileNetV2.yaml', 'w') as output:
    yaml.dump(ops, output)


######################################################################
# Android Steps
# -----
# 3. Building PyTorch Android with prepared operators list.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To build PyTorch Android with the prepared yaml list of operators,
# specify it in the environment variable SELECTED_OP_LIST. Also in the
# arguments, specify which Android ABIs it should build; by default it
# builds all 4 Android ABIs.
#

# Build PyTorch Android library customized for MobileNetV2:
SELECTED_OP_LIST=MobileNetV2.yaml scripts/build_pytorch_android.sh arm64-v8a



######################################################################
# iOS Steps
# -----
# 3. Building PyTorch iOS with prepared operators list.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To run the iOS build script locally with the prepared yaml list of operators,
# pass in the yaml file generate from the last step into the environment
# variable SELECTED_OP_LIST. Also in the arguments, specify BUILD_PYTORCH_MOBILE=1
# as well as the platform/architechture type. Take the arm64 build for example,
# the command should be:

SELECTED_OP_LIST=MobileNetV2.yaml BUILD_PYTORCH_MOBILE=1 IOS_ARCH=arm64 ./scripts/build_ios.sh

######################################################################
# 4. Integrate the result libraries to your project by following the XCode
# Setup section above.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

######################################################################
# 5. C++ one line change
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The last step is to add a single line of C++ code before running forward.
# This is because by default JIT will do some optimizations on operators
# (fusion for example), which might break the consistency with the ops we dumped
# from the model.
#

torch::jit::GraphOptimizerEnabledGuard guard(false);
