import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Subset
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

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale = (0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness = 0.1,
        contrast = 0.1,
        saturation = 0.1,
    ),
    transforms.ToTensor(),
    normalise,
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalise,
])

train_full_dataset = OxfordIIITPet(
    root = "data",
    split = "trainval",
    target_types = "category",
    transform = train_transform,
    download = True
)

val_full_dataset = OxfordIIITPet(
    root = "data",
    split = "trainval",
    target_types = "category",
    transform = val_transform,
    download = False
)

train_size = int(0.80 * len(train_full_dataset))
val_size = len(train_full_dataset) - train_size

indices = torch.randperm(len(train_full_dataset)).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(train_full_dataset, train_indices)
val_dataset = Subset(val_full_dataset, val_indices)

#print("Train sample: ", type(train_dataset[0][0]))
#print("Val sample: ", type(val_dataset[0][0]))

print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)

print("Dataloaders created")

num_classes = len(train_full_dataset.classes)
model = CNN(num_classes = num_classes).to(device)
loss_fn = nn.CrossEntropyLoss(label_smoothing = 0.1)
optimiser = optim.Adam(model.parameters(), lr = 0.001)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max = 45)

#sample_img, sample_label = train_dataset[0]
#print("Image type: ", type(sample_img))
#print("Image shape: ", sample_img.shape)
#print("Label: ", sample_label)
#print("Label type :", type(sample_label))

#labels = [train_dataset[i][1] for i in range(20)]
#print("First 20 labels: ", labels)
#print("Unique labels in first 20: ", set(labels))

#model.eval()
#dummy = torch.randn(4, 3, 224, 224).to(device)

#with torch.no_grad():
    #out = model(dummy)
#print("Output shape:", out.shape)
#print("Output sample:", out[0][:5])
#print("Output std:", out.std().item())

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
    lr_scheduler.step()

    print(f"Epoch number {epoch + 1}: "
          f"Train Loss = {train_loss:.4f} | Train Acc = {train_acc * 100:.2f}% | "
          f"Val Loss = {val_loss:.4f} | Val Acc = {val_acc * 100:.2f}% | "
          f"LR = {lr_scheduler.get_last_lr()[0]:.6f}"
          )
