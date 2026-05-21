import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from model import CNN

#set a fixed seed of 42 to ensure consistent values
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True       #ensures determinisitc CUDA operations
torch.backends.cudnn.benchmark = False      #disables auto-tuning for consistency

#use GPU if available as an option else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use:", device)

#normalise using ImageNet mean and std
normalise = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
)

#training transformation with data augmentation to imporve generalisation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale = (0.8, 1.0)),      #random crop and resize
    transforms.RandomHorizontalFlip(),      #random horizontal flip
    transforms.ColorJitter(     #random colour variation                         
        brightness = 0.1,
        contrast = 0.1,
        saturation = 0.1,
    ),
    transforms.ToTensor(),        #converts the PIL images to tensor
    normalise,      #normalises the pixel values
])

"""
=== REMOVED FOR FINAL RUN ===

#validation transformation with no augmentation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),      #resize to fixed size
    transforms.ToTensor(),              
    normalise,
])
"""

#load the entire trainval split 
train_full_dataset = OxfordIIITPet(
    root = "data",
    split = "trainval",
    target_types = "category",
    transform = train_transform,
    download = True
)

"""
=== REMOVED FOR FINAL RUN ===

val_full_dataset = OxfordIIITPet(
    root = "data",
    split = "trainval",
    target_types = "category",
    transform = val_transform,
    download = False
)

#split trainval into an 80/20 split with 80% on training and 20% on validation
train_size = int(0.80 * len(train_full_dataset))
val_size = len(train_full_dataset) - train_size

#generate random indices and split them
indices = torch.randperm(len(train_full_dataset)).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

#create subsets using the split indices
train_dataset = Subset(train_full_dataset, train_indices)
val_dataset = Subset(val_full_dataset, val_indices)
"""

print(f"Train size: {len(train_full_dataset)}")

#create dataloaders, train_loader shuffled for randomness
train_loader = DataLoader(train_full_dataset, batch_size = 32, shuffle = True)


"""
=== REMOVED FOR FINAL RUN ===

val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)
"""

print("Dataloaders created")

#initialise model, loss function, optimiser and LR scheduler
num_classes = len(train_full_dataset.classes)
model = CNN(num_classes = num_classes).to(device)
loss_fn = nn.CrossEntropyLoss(label_smoothing = 0.1)        #crossentropyloss with label smoothing to help reduce overconfidence
optimiser = optim.Adam(model.parameters(), lr = 0.0005)     #adam optimiser with low LR
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max = 45)        #cosine annealing smoothly decays LR and T_max = 45 to decay LR slowly

"""
Trains the model for one epoch and returns average loss and accuracy
"""
def train_epoch(model, loader, loss_fn, optimiser):
    model.train()       #set model to training mode
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:   
        #move data to GPU ir CPU
        images, labels = images.to(device), labels.to(device)

        optimiser.zero_grad()       #clear gradients from the previous step
        outputs = model(images)     #forward pass
        loss = loss_fn(outputs, labels)     #find loss
        loss.backward()         #backpropagation
        optimiser.step()        #update weights

        total_loss = total_loss + loss.item()

        #count the number of correct predictions
        _, preds = torch.max(outputs, 1)
        correct = correct + (preds == labels).sum().item()
        total = total + labels.size(0)
    
    avg_loss = total_loss / len(loader)
    train_acc = correct / total

    return avg_loss, train_acc

"""
=== REMOVED FOR FINAL RUN ===

def validate(model, loader, loss_fn):
    
    model.eval()        #set model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():       #disable gradient computation 
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
"""

#training loop running for 30 epochs
for epoch in range(30):
    train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimiser)
    lr_scheduler.step()     #update LR after each epoch

    print(f"Epoch number {epoch + 1}: "
          f"Train Loss = {train_loss:.4f} | Train Acc = {train_acc * 100:.2f}% | "
          f"LR = {lr_scheduler.get_last_lr()[0]:.6f}"
          )

#save the trained model weights for evaluation in test.py
torch.save(model.state_dict(), "model.pth")
print("Model saved")