import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, random_split
from model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use:", device)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale = (0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness = 0.2,
        contrast = 0.3,
        saturation = 0.2,
    ),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

first_dataset = OxfordIIITPet(
    root = "data",
    split = "trainval",
    target_types = "category",
    transform = None,
    download = True
)

train_size = int(0.80 * len(first_dataset))
val_size = len(first_dataset) - train_size

train_dataset, val_dataset = random_split(
    first_dataset,
    [train_size, val_size]
)

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

print("Train sample: ", type(train_dataset[0][0]))
print("Val sample: ", type(val_dataset[0][0]))

print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)

print("Dataloaders created")

num_classes = len(first_dataset.classes)
model = CNN(num_classes = num_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr = 0.0001)

def train_epoch(model, loader, loss_fn, optimiser):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimiser.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimiser.step()

        total_loss = total_loss + loss.item()

        _, preds = torch.max(outputs, 1)
        correct = correct + (preds == labels).sum().item()
        total = total + labels.size(0)
    
    avg_loss = total_loss / len(loader)
    train_acc = correct / total

    return avg_loss, train_acc

def validate(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss = total_loss + loss.item()

            _, preds = torch.max(outputs, 1)
            correct = correct + (preds == labels).sum().item()
            total = total + labels.size(0)

    avg_loss = total_loss / len(loader)
    val_acc = correct / total
    
    return avg_loss, val_acc

for epoch in range(30):
    train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimiser)
    val_loss, val_acc = validate(model, val_loader, loss_fn)

    print(f"Epoch number {epoch + 1}: "
          f"Train Loss = {train_loss:.4f} | Train Acc = {train_acc * 100:.2f}% | "
          f"Val Loss = {val_loss:.4f} | Val Acc = {val_acc * 100:.2f}%"
          )
