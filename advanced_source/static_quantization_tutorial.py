from torchvision.datasets import ImageNet
import torchvision.transforms as transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

val_data = ImageNet('~/.data/',
    split='val',
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize]
    )
)

print(val_data)
print(type(val_data))
