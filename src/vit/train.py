import torch
from torchvision import datasets, transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .vit import VisionTransformer
from sklearn.metrics import accuracy_score, f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# MNIST
# -----------------------------

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="./src/data/MNIST", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./src/data/MNIST", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# -----------------------------
# CIFAR 10
# -----------------------------

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

#train_dataset = datasets.CIFAR10(root='./src/data/CIFAR10', train=True, download=True, transform=transform_test)
#test_dataset = datasets.CIFAR10(root='./src/data/CIFAR10', train=False, download=True, transform=transform_test)

#train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=128)

# -----------------------------
# Training
# -----------------------------
model = VisionTransformer(img_size=28, patch_size=4, in_channels=1, n_embd=36*4, n_head=6, n_layer=9, num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-2)


total_correct = 0
total_samples = 0

for epoch in range(5):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)

        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    train_accuracy = total_correct / total_samples

    total_correct = 0
    total_samples = 0

    model.eval()

    for images, labels in test_loader:
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)

            logits = model(images)

            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    test_accuracy = total_correct / total_samples

    print(f"Epoch {epoch} - Loss: {avg_loss:.4f} - Train Acc: {train_accuracy:.4f} - Test Acc: {test_accuracy:.4f}")
