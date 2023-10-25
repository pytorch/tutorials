import torch
import torchvision.models as models
model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
model.eval()
data = torch.rand(1, 3, 224, 224)
#################### code changes ####################
import intel_extension_for_pytorch as ipex
# Invoke the following API optionally, to apply frontend optimizations
model = ipex.optimize(model, weights_prepack=False)
compile_model = torch.compile(model, backend="ipex")
######################################################
with torch.no_grad():
    print(compile_model(data))
