import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from model import CNN

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use:", device)

normalise = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalise,
])

test_dataset = OxfordIIITPet(
    root = "data", split = "test",
    target_types = "category", transform = test_transform, download = True
)

test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)
print(f"Test size: {len(test_dataset)}")

num_classes = len(test_dataset.classes)
model = CNN(num_classes = num_classes).to(device)
model.load_state_dict(torch.load("model.pth", map_location = device))
model.eval()

correct = 0 
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct = correct + (preds == labels).sum().item()
        total = total + labels.size(0)

test_acc = correct / total
print