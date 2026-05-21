import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from model import CNN

#use GPU if available as an option else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use:", device)

#normalise using ImageNet mean and std
normalise = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
)

#test transformation with no augmentation
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalise,
])

#test split
test_dataset = OxfordIIITPet(
    root = "data", split = "test",
    target_types = "category", transform = test_transform, download = True
)

#load test data in batch size of 32
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)
print(f"Test size: {len(test_dataset)}")

#initialise model and load trained weights 
num_classes = len(test_dataset.classes)
model = CNN(num_classes = num_classes).to(device)
model.load_state_dict(torch.load("model.pth", map_location = device))
model.eval()        #set model to evaluation

correct = 0 
total = 0

#evaluating on test set 
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)     #forward pass
        _, preds = torch.max(outputs, 1)        #get predicted class
        correct = correct + (preds == labels).sum().item()
        total = total + labels.size(0)

#final test accuracy
test_acc = correct / total

print(f"Test Accuracy: {test_acc * 100:.2f}%")