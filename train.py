import torch
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEvice in use:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

trainval_dataset = OxfordIIITPet(
    root = "data",
    split = "trainval",
    target_types = "category",
    transform = transform,
    download = True
)

print("Trainval split loaded with", len(trainval_dataset), "images")
print("Classes:", trainval_dataset.classes)